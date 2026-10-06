@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pinch.app
) else if exist "%USERPROFILE%\Desktop\Experiment\.venv\Scripts\python.exe" (
  "%USERPROFILE%\Desktop\Experiment\.venv\Scripts\python.exe" -m pinch.app
) else (
  py -3.13 -m pinch.app
)
if errorlevel 1 pause
