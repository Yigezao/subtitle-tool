"""已知文字 + 音频 → 精确 SRT（强制对齐，不做语音识别写稿）。

用法：
    python align_srt.py "第一段.wav" "第一段.txt"
    python align_srt.py "全文.wav" "final.txt" --model large-v3

文字用你已有的口播稿，识别只负责给每个字找时间；所以文字不会出现
“羊之囚”“托特纳加”这类错字，时间轴又跟着真实语音走。
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import time
from pathlib import Path

from transcribe import DEFAULT_CUDA_LIB, add_cuda_dlls, fmt_time, load_model

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def is_content(ch: str) -> bool:
    return ("一" <= ch <= "鿿") or ch.isalnum()


def read_text(path: str) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8-sig").strip()


def cue_ranges(text: str, max_chars: int) -> list[tuple[int, int]]:
    """按标点和字数切字幕，返回原文的 [start, end) 区间，保留标点。"""
    ranges: list[tuple[int, int]] = []
    start = 0
    count = 0
    for index, ch in enumerate(text):
        if not ch.isspace():
            count += 1
        end_punct = ch in "，,。.！!？?；;：:"
        if (end_punct or count >= max_chars) and index + 1 > start:
            if text[start:index + 1].strip():
                ranges.append((start, index + 1))
            start = index + 1
            count = 0
    if start < len(text) and text[start:].strip():
        ranges.append((start, len(text)))
    return ranges


def words_with_times(model, audio: str, language: str, prompt: str) -> tuple[list[tuple[str, float, float]], float]:
    segments, info = model.transcribe(
        audio,
        language=None if language == "auto" else language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
        initial_prompt=prompt or None,
        word_timestamps=True,
    )
    words: list[tuple[str, float, float]] = []
    for seg in segments:
        for word in seg.words or []:
            text = (word.word or "").strip()
            if text:
                words.append((text, float(word.start), float(word.end)))
    return words, float(getattr(info, "duration", 0.0) or 0.0)


def build_rec_chars(words: list[tuple[str, float, float]]) -> tuple[list[str], list[tuple[float, float]]]:
    chars: list[str] = []
    times: list[tuple[float, float]] = []
    for text, start, end in words:
        content = [ch for ch in text if is_content(ch)]
        if not content:
            continue
        span = max(0.01, end - start)
        for index, ch in enumerate(content):
            t0 = start + span * index / len(content)
            t1 = start + span * (index + 1) / len(content)
            chars.append(ch.lower() if ch.isascii() else ch)
            times.append((t0, t1))
    return chars, times


def char_times_for_text(text: str, rec_chars: list[str], rec_times: list[tuple[float, float]]) -> list[tuple[float, float] | None]:
    known_chars: list[str] = []
    known_positions: list[int] = []
    for index, ch in enumerate(text):
        if is_content(ch):
            known_chars.append(ch.lower() if ch.isascii() else ch)
            known_positions.append(index)
    times: list[tuple[float, float] | None] = [None] * len(text)
    if not known_chars or not rec_chars:
        return times
    matcher = difflib.SequenceMatcher(None, "".join(known_chars), "".join(rec_chars), autojunk=False)
    matched = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            known_index = block.a + offset
            rec_index = block.b + offset
            times[known_positions[known_index]] = rec_times[rec_index]
            matched += 1
    coverage = matched / max(1, len(known_chars))
    # 未匹配的字：用前后最近的有时间字插值
    last: tuple[float, float] | None = None
    for index in range(len(text)):
        if times[index] is not None:
            last = times[index]
            continue
        next_time = None
        for j in range(index + 1, len(text)):
            if times[j] is not None:
                next_time = times[j]
                break
        if last and next_time:
            blend = (last[1] + next_time[0]) / 2
            times[index] = (blend, blend)
        elif last:
            times[index] = (last[1], last[1])
        elif next_time:
            times[index] = (next_time[0], next_time[0])
    # 单调化
    prev_end = 0.0
    for index in range(len(text)):
        if times[index] is None:
            times[index] = (prev_end, prev_end)
            continue
        start, end = times[index]
        start = max(start, prev_end)
        end = max(end, start + 0.02)
        times[index] = (start, end)
        prev_end = end
    return times


def proportional_times(text: str, duration: float) -> list[tuple[float, float]]:
    chars = max(1, sum(1 for ch in text if not ch.isspace()))
    result = []
    cursor = 0.0
    for ch in text:
        if ch.isspace():
            result.append((cursor, cursor))
            continue
        step = duration / chars
        result.append((cursor, cursor + step))
        cursor += step
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="已知文字 + 音频 → 精确 SRT（强制对齐）")
    parser.add_argument("audio", help="音频/视频文件")
    parser.add_argument("text", help="与音频对应的口播稿 txt（不要多传）")
    parser.add_argument("--model", default="medium", help="tiny/base/small/medium/large-v3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--prompt", default="", help="人名/书名提示")
    parser.add_argument("--output", default="", help="输出 SRT；默认与音频同目录")
    parser.add_argument("--max-chars", type=int, default=26, help="单条字幕最多多少字")
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    text_path = Path(args.text).expanduser().resolve()
    if not audio.is_file() or not text_path.is_file():
        print("音频或文字文件不存在。")
        return 2
    text = read_text(str(text_path))
    if not text:
        print("文字是空的。")
        return 3
    output = Path(args.output).expanduser().resolve() if args.output else audio.with_suffix(".srt")

    lib = add_cuda_dlls()
    if lib:
        print(f"CUDA DLL：{lib}", flush=True)
    print(f"加载模型 {args.model}…", flush=True)
    started = time.time()
    model, device_info = load_model(args.model, args.device, args.compute_type)
    print(f"模型就绪：{device_info}，用时 {time.time() - started:.1f}s", flush=True)
    print("正在识别词级时间（只取时间，不用它写文字）…", flush=True)
    started = time.time()
    words, duration = words_with_times(model, str(audio), args.language, args.prompt)
    rec_chars, rec_times = build_rec_chars(words)
    times = char_times_for_text(text, rec_chars, rec_times)
    coverage = sum(1 for item in times if item and item[1] > item[0]) / max(1, len(text))
    if coverage < 0.3 or not words:
        print("词级对齐覆盖太低，改用按字数比例分配时间；文字仍然准确。", flush=True)
        times = proportional_times(text, duration or 1.0)
    else:
        print(f"词级对齐覆盖约 {coverage * 100:.0f}%", flush=True)

    cues = []
    for start_index, end_index in cue_ranges(text, max(10, args.max_chars)):
        cue_text = text[start_index:end_index].strip()
        if not cue_text:
            continue
        segment = [item for item in times[start_index:end_index] if item]
        if not segment:
            continue
        start = segment[0][0]
        end = max(segment[-1][1], start + 0.2)
        cues.append((start, end, cue_text))
    if not cues:
        print("没有生成字幕。")
        return 4
    with output.open("w", encoding="utf-8-sig", newline="\n") as handle:
        for index, (start, end, cue_text) in enumerate(cues, start=1):
            handle.write(f"{index}\n{fmt_time(start)} --> {fmt_time(end)}\n{cue_text}\n\n")
    txt_path = output.with_suffix(".txt")
    txt_path.write_text("\n".join(cue[2] for cue in cues) + "\n", encoding="utf-8")
    print(f"\n完成：{len(cues)} 条字幕，用时 {time.time() - started:.1f}s")
    print(f"SRT：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
