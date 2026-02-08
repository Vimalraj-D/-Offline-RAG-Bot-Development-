#!/bin/bash

# Offline RAG Bot Setup Script
# This script sets up the complete environment

set -e

echo "======================================"
echo "🤖 Offline RAG Bot - Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists"
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "📦 Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt --quiet
echo "   ✅ Dependencies installed"

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data/documents
mkdir -p models/embeddings
mkdir -p models/llm
mkdir -p vector_db
mkdir -p static/css
mkdir -p static/js
mkdir -p templates
echo "   ✅ Directories created"

# Download models
echo ""
echo "📥 Downloading AI models..."
echo "   This will download ~1.3GB of models"
read -p "   Do you want to download models now? (y/n): " download_choice

if [ "$download_choice" = "y" ] || [ "$download_choice" = "Y" ]; then
    python download_models.py
else
    echo "   ⚠️  Skipping model download"
    echo "   You can download later with: python download_models.py"
fi

# Create sample document if none exist
echo ""
echo "📄 Checking for documents..."
if [ -z "$(ls -A data/documents/*.{pdf,txt,md} 2>/dev/null)" ]; then
    echo "   No documents found. A sample document will be created on first run."
else
    echo "   ✅ Documents found"
fi

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "📝 Next steps:"
echo "   1. Activate the virtual environment:"
echo "      source venv/bin/activate"
echo ""
echo "   2. (Optional) Add your documents to:"
echo "      data/documents/"
echo ""
echo "   3. Run the application:"
echo "      python app.py"
echo ""
echo "   4. Open your browser to:"
echo "      http://localhost:5000"
echo ""
echo "======================================"
echo "Happy chatting! 🚀"
echo "======================================"
