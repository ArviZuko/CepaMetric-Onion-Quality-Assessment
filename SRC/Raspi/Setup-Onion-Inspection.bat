@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_desktop_icon.ps1"
if errorlevel 1 (
  echo Could not create the desktop icon.
  pause
  exit /b 1
)
echo.
echo The desktop icon "Onion AI Inspection" is ready.
echo Double-click it to open the application.
echo.
pause