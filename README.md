# 音频转字幕（faster-whisper）

把音频或视频拖到 `转字幕.bat` 上，自动在同目录生成 `.srt` 和 `.txt`。

## 用哪个模型

- 默认 **medium**（约 1.5GB，中文清晰语音够用，RTX 2060 上很快）
- 人名/书名多、要求更高：`--model large-v3`（约 3GB，更慢）
- 只想快速看个大概：`--model small`

## 命令行

```bash
python transcribe.py "D:\某处\音频.wav"
python transcribe.py "音频.wav" --model large-v3 --language zh
python transcribe.py "音频.wav" --prompt "专有名词1, 专有名词2" --max-chars 20
```

输出默认放在音频旁边，同名 `.srt` / `.txt`。

## 环境说明

- 依赖 `faster-whisper` + `ctranslate2`
- 模型下载走 `hf-mirror.com`（Hugging Face 在国内连不上）
- CUDA 需要 cuBLAS/cuDNN；脚本从环境变量 `WHISPER_CUDA_LIB` 指定的目录加载，例如：
  `%VIRTUAL_ENV%\Lib\site-packages\torch\lib`
  未设置时可用 `--device cpu`（会慢很多）
- 模型缓存：`%USERPROFILE%\.cache\huggingface\hub\`

## 准确率提醒

Whisper 对清晰 TTS 效果不错，但人名、书名、生僻字仍可能错；导出后要过一遍。  
`--prompt` 可以给几个专有名词做提示，能减少错字。

## 已知文字 + 音频（比纯识别更准）

如果这段音频本来就是照着某个 txt 读的，直接对齐，文字不会错：

```bash
python align_srt.py "音频.wav" "对应版本的口播稿.txt" --model medium
```

也可以把**音频和 txt 两个文件一起拖到 `对齐字幕.bat` 上**。

注意：**txt 必须是这段音频用的那一版**。

例：稿件改过版、而音频是按旧版录的，两者开头对不上，直接拿新版去对齐就会错位。
这时改用与音频同版本的备份稿（如 `*.bak-<日期>`、`draft.vN.txt`），并只取与音频对应的那一段。

工具只负责给已知文字找时间，不会改写文字；所以人名、书名这类专有名词不会再被识别成近似字。
