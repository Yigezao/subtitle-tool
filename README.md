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
python transcribe.py "音频.wav" --prompt "托特, 羊脂球, 普鲁士军官" --max-chars 20
```

输出默认放在音频旁边，同名 `.srt` / `.txt`。

## 环境说明

- 已安装 `faster-whisper 1.2.1` + `ctranslate2 4.8.2`
- 模型下载走 `hf-mirror.com`（Hugging Face 在国内连不上）
- CUDA 需要 cuBLAS/cuDNN；脚本默认借用本机 Ming 的 torch 目录：
  `D:\my_Software\Ming-3.0-TTS\venv\Lib\site-packages\torch\lib`
  如果这个目录被移动，设置环境变量 `WHISPER_CUDA_LIB` 指向新的 `torch\lib`，或直接用 `--device cpu`（会慢很多）
- 模型缓存：`C:\Users\15844\.cache\huggingface\hub\`

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

例：
- 当前 `export\final.txt` 开头是「一辆马车，十个人…」
- 但 Audacity 里那段 wav 开头是「托特那家旅店的院子里…」
- 两版对不上，直接拿当前 final.txt 去对齐会错位。

这种情况下用旧版备份，例如：
`data\books\羊脂球\export\final.txt.bak-20261025-111921`
或者对应的 `draft.v10.txt` / `v11.txt`，而且只取与音频对应的那一段。

工具只负责给已知文字找时间，不会改写文字；所以「羊脂球」「托特」不会再被识别成「羊之囚」「托特纳加」。
