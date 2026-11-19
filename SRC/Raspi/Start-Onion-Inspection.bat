@echo off
cd /d "%~dp0"
set "PYW=C:\Program Files\Python313\pythonw.exe"
set "PY=C:\Program Files\Python313\python.exe"
if not exist "%PYW%" set "PYW=%PY%"
if not exist "%PYW%" (
  echo Python was not found.
  echo Install Python 3.13, then double-click this file again.
  pause
  exit /b 1
)
start "" "%PYW%" "%~dp0final.py"
exit /b 0
