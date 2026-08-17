@echo off
chcp 65001 >nul
title CANLI DEMO - Konusmadan Duygu Tanima
cd /d "C:\Users\user\Desktop\470 proje\speech-emotion-recognition"
echo.
echo   CANLI DEMO: rastgele 3 kayit iki yontemden geciriliyor...
echo   ==========================================================
".venv\Scripts\python.exe" -X utf8 final\demo.py --rastgele 3
echo.
echo   ==========================================================
echo   Kapatmak icin bir tusa bas.
pause >nul
