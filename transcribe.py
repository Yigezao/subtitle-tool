"""用 faster-whisper 把音频/视频转成 SRT 字幕。

用法：
    python transcribe.py "音频.wav"
    python transcribe.py "音频.wav" --model medium --language zh
    python transcribe.py "音频.wav" --prompt "托特, 羊脂球, 普鲁士军官"

第一次运行会通过 hf-mirror.com 下载模型（medium 约 1.5GB）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Hugging Face 在国内经常连不上；默认走镜像，并关闭 Xet 传输。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# CTranslate2 需要 cuBLAS/cuDNN；本机 Ming 的 torch 自带这些 DLL。
DEFAULT_CUDA_LIB = r"D:\my_Software\Ming-3.0-TTS\venv\Lib\site-packages\torch\lib"


def add_cuda_dlls() -> str | None:
    candidates = [
        os.environ.get("WHISPER_CUDA_LIB", ""),
        DEFAULT_CUDA_LIB,
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if not path.is_dir():
            continue
        os.environ["PATH"] = str(path) + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(path))
        except (AttributeError, OSError):
            pass
        return str(path)
    return None


def fmt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def count_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def split_pieces(text: str, max_chars: int) -> list[str]:
    """把一段字幕拆成适合竖屏/横屏阅读的短句；每句不超过 max_chars 个字。"""
    clean = re.sub(r"\s+", "", text or "").strip()
    if not clean:
        return []
    raw = re.split(r"(?<=[，,。.！!？?；;：:])", clean)
    pieces: list[str] = []
    for part in raw:
        part = part.strip()
        while len(part) > max_chars:
            cut = max(part.rfind("，", 0, max_chars), part.rfind(",", 0, max_chars))
            if cut < 4:
                cut = max_chars - 1
            pieces.append(part[: cut + 1])
            part = part[cut + 1:]
        if part:
            pieces.append(part)
    groups: list[str] = []
    buffer = ""
    for piece in pieces:
        if buffer and count_chars(buffer + piece) > max_chars:
            groups.append(buffer)
            buffer = piece
        else:
            buffer += piece
    if buffer:
        groups.append(buffer)
    return groups


def write_srt(segments: list, out_path: Path, max_chars: int) -> int:
    cues: list[tuple[float, float, str]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        parts = split_pieces(text, max_chars)
        total = sum(count_chars(part) for part in parts) or 1
        duration = max(0.01, float(seg.end) - float(seg.start))
        cursor = float(seg.start)
        for part in parts:
            share = duration * count_chars(part) / total
            start = cursor
            end = min(float(seg.end), cursor + share)
            if end - start < 0.2:
                end = min(float(seg.end), start + 0.2)
            cues.append((start, end, part))
            cursor = end
    with out_path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        for index, (start, end, text) in enumerate(cues, start=1):
            handle.write(f"{index}\n{fmt_time(start)} --> {fmt_time(end)}\n{text}\n\n")
    return len(cues)


def load_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            return WhisperModel(model_name, device="cuda", compute_type="float16"), "cuda/float16"
        except Exception as exc:  # noqa: BLE001
            print(f"CUDA 不可用，改用 CPU：{exc}", flush=True)
            return WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu/int8"
    return WhisperModel(model_name, device=device, compute_type=compute_type), f"{device}/{compute_type}"


def main() -> int:
    parser = argparse.ArgumentParser(description="音频转 SRT 字幕（faster-whisper）")
    parser.add_argument("input", help="音频或视频文件")
    parser.add_argument("--model", default="medium", help="tiny/base/small/medium/large-v3，默认 medium")
    parser.add_argument("--language", default="zh", help="语言代码，默认 zh；自动检测用 auto")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--compute-type", default="float16", help="CUDA 用 float16；CPU 用 int8")
    parser.add_argument("--output", default="", help="输出 SRT 路径；默认与音频同目录")
    parser.add_argument("--prompt", default="", help="人名/书名提示，例如：托特, 羊脂球, 普鲁士军官")
    parser.add_argument("--max-chars", type=int, default=26, help="单条字幕最多多少字，默认 26")
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        print(f"找不到文件：{source}")
        return 2
    output = Path(args.output).expanduser().resolve() if args.output else source.with_suffix(".srt")

    lib = add_cuda_dlls()
    if lib:
        print(f"CUDA DLL：{lib}", flush=True)
    print(f"加载模型 {args.model}…（第一次会下载，medium 约 1.5GB）", flush=True)
    started = time.time()
    model, device_info = load_model(args.model, args.device, args.compute_type)
    print(f"模型就绪：{device_info}，用时 {time.time() - started:.1f}s", flush=True)
    print(f"开始识别：{source.name}", flush=True)

    started = time.time()
    segments_iter, info = model.transcribe(
        str(source),
        language=None if args.language == "auto" else args.language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
        initial_prompt=args.prompt or None,
    )
    segments = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(seg)
        print(f"[{fmt_time(seg.start)} → {fmt_time(seg.end)}] {text}", flush=True)

    if not segments:
        print("没有识别到语音。检查音频是否有声音、是否选错文件。")
        return 3
    cues = write_srt(segments, output, max(10, args.max_chars))
    txt_path = output.with_suffix(".txt")
    txt_path.write_text("\n".join((seg.text or "").strip() for seg in segments) + "\n", encoding="utf-8")
    print(f"\n完成：{cues} 条字幕，用时 {time.time() - started:.1f}s")
    print(f"SRT：{output}")
    print(f"TXT：{txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
