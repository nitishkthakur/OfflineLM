@echo off
title OfflineLM Chat Application Launcher
color 0A
echo ===============================================
echo    OfflineLM Chat Application Launcher
echo ===============================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at venv\Scripts\activate.bat
    echo.
    echo Please create a virtual environment first:
    echo   1. python -m venv venv
    echo   2. venv\Scripts\activate
    echo   3. pip install -r requirements_streaming.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if FastAPI file exists
if not exist "fastapi_streaming_improved.py" (
    echo [ERROR] fastapi_streaming_improved.py not found in current directory
    echo Please ensure you're running this from the correct directory.
    pause
    exit /b 1
)

REM Check if requirements file exists and dependencies are installed
if exist "requirements_streaming.txt" (
    echo [INFO] Checking dependencies...
    python -c "import fastapi, uvicorn, ollama" 2>nul
    if errorlevel 1 (
        echo [WARNING] Some dependencies might be missing.
        echo Installing/updating dependencies...
        pip install -r requirements_streaming.txt
    )
)

REM Check if Ollama is running
echo [INFO] Checking if Ollama service is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama service doesn't seem to be running on localhost:11434
    echo Please make sure Ollama is installed and running.
    echo You can start it with: ollama serve
    echo.
    echo Continuing anyway... (server will start but may not work properly)
    timeout /t 3 >nul
)

REM Start the FastAPI server in background
echo [INFO] Starting FastAPI server...
start "OfflineLM Server" python fastapi_streaming_improved.py

REM Wait for server to start
echo [INFO] Waiting for server to start...
timeout /t 5 /nobreak >nul

REM Try to check if server is responding
echo [INFO] Checking server status...
curl -s http://localhost:8001 >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] Server is running!
) else (
    echo [INFO] Server starting... (may take a moment)
)

REM Wait a bit more before opening browser
timeout /t 2 /nobreak >nul

REM Try Firefox first, then fallback to default browser
echo [INFO] Opening application in browser...
firefox http://localhost:8001 2>nul
if errorlevel 1 (
    echo [INFO] Firefox not found, trying default browser...
    start http://localhost:8001
)

echo.
echo ===============================================
echo   OfflineLM Chat Application is running!
echo   Server URL: http://localhost:8001
echo   Server Process: Check "OfflineLM Server" window
echo ===============================================
echo.
echo Press any key to stop the server and exit...
pause >nul

REM Kill the server process
echo [INFO] Stopping server...
taskkill /f /fi "WindowTitle eq OfflineLM Server*" 2>nul
taskkill /f /im python.exe /fi "CommandLine eq *fastapi_streaming_improved.py*" 2>nul
echo [INFO] Server stopped.
echo.
echo Thank you for using OfflineLM Chat Application!
timeout /t 2 >nul
