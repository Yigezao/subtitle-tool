@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if "%~1"=="" (
    echo Drag an audio or video file onto this .bat, or pass its path as an argument.
    pause
    exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
    py -3 "%~dp0transcribe.py" "%~1" --model medium
) else (
    python "%~dp0transcribe.py" "%~1" --model medium
)
pause
