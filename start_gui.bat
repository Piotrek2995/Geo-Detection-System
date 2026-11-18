@echo off
REM Geo-Detection-System — uruchomienie GUI (Streamlit). Otwiera przegladarke na http://localhost:8501
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Brak srodowiska .venv — zobacz README, sekcja Instalacja.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app\main.py --server.headless false
pause
