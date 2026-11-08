@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if "%~2"=="" (
    echo Drag the AUDIO file and the matching TXT file onto this .bat at the same time.
    echo Or run:
    echo   python align_srt.py "audio.wav" "script.txt" --model medium
    pause
    exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
    py -3 "%~dp0align_srt.py" "%~1" "%~2" --model medium
) else (
    python "%~dp0align_srt.py" "%~1" "%~2" --model medium
)
pause
