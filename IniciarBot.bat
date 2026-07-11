@echo off
title Bot DB Legends
cd /d "%~dp0"
echo ============================================
echo   Bot de farmeo - Dragon Ball Legends
echo   (Ctrl+C para detenerlo)
echo ============================================
tools\python\python.exe -u bot.py
echo.
echo El bot se ha detenido.
pause
