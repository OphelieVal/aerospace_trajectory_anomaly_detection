@echo off
set VENV_DIR=.venv

REM 1. Create venv if not exists
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM 2. Activate
call %VENV_DIR%\Scripts\activate

REM 3. Install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo Environment ready. Venv activated.
