"""
Model Download Script
Downloads required embedding and LLM models for offline use
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
EMBEDDING_DIR = MODELS_DIR / "embeddings"
LLM_DIR = MODELS_DIR / "llm"

# Create directories
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
LLM_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_embedding_model():
    """Download SentenceTransformer embedding model"""
    print("\n📦 Downloading Embedding Model...")
    print("Model: all-MiniLM-L6-v2 (~80MB)")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # This will download the model to the default cache
        # Then we'll note where it is
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Embedding model downloaded successfully!")
        print(f"📁 Model will be loaded from HuggingFace cache")
        return True
    except Exception as e:
        print(f"❌ Error downloading embedding model: {e}")
        return False

def download_llm_model():
    """Download Llama model in GGUF format"""
    print("\n📦 Downloading LLM Model...")
    print("Model: Llama-3.2-1B-Instruct-Q4_K_M (~1.2GB)")
    print("This may take several minutes depending on your connection...")
    
    try:
        # Download from Hugging Face - using correct filename
        model_file = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        repo_id = "bartowski/Llama-3.2-1B-Instruct-GGUF"
        
        destination = LLM_DIR / model_file
        
        if destination.exists():
            print(f"✅ Model already exists at {destination}")
            return True
        
        print(f"Downloading from Hugging Face: {repo_id}")
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            local_dir=LLM_DIR,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ LLM model downloaded successfully!")
        print(f"📁 Saved to: {downloaded_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading LLM model: {e}")
        print("\nAlternative: You can manually download from:")
        print("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
        print(f"And place it in: {LLM_DIR}")
        return False

def verify_models():
    """Verify that all required models are available"""
    print("\n🔍 Verifying models...")
    
    # Check embedding model
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model: OK")
        embedding_ok = True
    except:
        print("❌ Embedding model: NOT FOUND")
        embedding_ok = False
    
    # Check LLM model
    llm_files = list(LLM_DIR.glob("*.gguf"))
    if llm_files:
        print(f"✅ LLM model: OK ({llm_files[0].name})")
        llm_ok = True
    else:
        print("❌ LLM model: NOT FOUND")
        llm_ok = False
    
    return embedding_ok and llm_ok

def main():
    """Main download function"""
    print("=" * 60)
    print("🤖 Offline RAG Bot - Model Downloader")
    print("=" * 60)
    
    print(f"\n📂 Models will be saved to: {MODELS_DIR}")
    
    # Download embedding model
    embedding_success = download_embedding_model()
    
    # Download LLM model
    llm_success = download_llm_model()
    
    # Verify everything
    print("\n" + "=" * 60)
    all_ok = verify_models()
    print("=" * 60)
    
    if all_ok:
        print("\n🎉 All models downloaded successfully!")
        print("You can now run: python app.py")
    else:
        print("\n⚠️  Some models are missing. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()