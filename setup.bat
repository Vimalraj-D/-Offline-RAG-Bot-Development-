@echo off
REM Offline RAG Bot Setup Script for Windows
REM This script sets up the complete environment

echo ======================================
echo 🤖 Offline RAG Bot - Setup Script
echo ======================================
echo.

REM Check Python version
echo 📋 Checking Python version...
python --version
echo.

REM Create virtual environment
echo 🔧 Creating virtual environment...
if exist venv (
    echo    Virtual environment already exists
) else (
    python -m venv venv
    echo    ✅ Virtual environment created
)
echo.

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate
echo    ✅ Activated
echo.

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

REM Install requirements
echo 📦 Installing dependencies...
echo    This may take a few minutes...
pip install -r requirements.txt --quiet
echo    ✅ Dependencies installed
echo.

REM Create necessary directories
echo 📁 Creating directories...
if not exist data\documents mkdir data\documents
if not exist models\embeddings mkdir models\embeddings
if not exist models\llm mkdir models\llm
if not exist vector_db mkdir vector_db
if not exist static\css mkdir static\css
if not exist static\js mkdir static\js
if not exist templates mkdir templates
echo    ✅ Directories created
echo.

REM Download models
echo 📥 Downloading AI models...
echo    This will download ~1.3GB of models
set /p download_choice="   Do you want to download models now? (y/n): "

if /i "%download_choice%"=="y" (
    python download_models.py
) else (
    echo    ⚠️  Skipping model download
    echo    You can download later with: python download_models.py
)
echo.

echo ======================================
echo ✅ Setup Complete!
echo ======================================
echo.
echo 📝 Next steps:
echo    1. Activate the virtual environment:
echo       venv\Scripts\activate
echo.
echo    2. (Optional) Add your documents to:
echo       data\documents\
echo.
echo    3. Run the application:
echo       python app.py
echo.
echo    4. Open your browser to:
echo       http://localhost:5000
echo.
echo ======================================
echo Happy chatting! 🚀
echo ======================================
pause
