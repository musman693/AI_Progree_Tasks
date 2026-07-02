@echo off
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)
start "Progree Tasks" cmd /k "%PYTHON_EXE%" app.py
