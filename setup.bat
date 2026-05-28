@echo off
REM ============== Robotic Arm - Setup Script ==============
echo.
echo [*] Creating Virtual Environment...
python -m venv venv

echo.
echo [*] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo.
echo [*] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [*] Installing Dependencies...
pip install -r requirements.txt

echo.
echo [✓] Setup Complete!
echo.
echo To activate the virtual environment in future sessions, run:
echo   venv\Scripts\activate.bat
echo.
echo Then start the app with:
echo   python app.py
echo.
pause
