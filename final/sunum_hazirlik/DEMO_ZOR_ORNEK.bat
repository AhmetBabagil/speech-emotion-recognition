@echo off
chcp 65001 >nul
title CANLI DEMO - Zor Ornek (disgust)
cd /d "C:\Users\user\Desktop\470 proje\speech-emotion-recognition"
echo.
echo   ZOR ORNEK: modellerin yanildigi bir disgust kaydi...
echo   ==========================================================
".venv\Scripts\python.exe" -X utf8 final\demo.py final\demo_ornekleri\1013_TIE_DIS_XX.wav
echo.
echo   ==========================================================
echo   Kapatmak icin bir tusa bas.
pause >nul
