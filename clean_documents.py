"""
Clean Documents Script
Fixes encoding issues and recreates clean sample documents
"""

import sys
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

def clean_documents():
    """Remove problematic documents and create clean ones"""
    print("\n" + "="*60)
    print("🧹 Cleaning Documents Directory")
    print("="*60)
    
    # Create directory if it doesn't exist
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Remove old sample.txt if it exists
    old_sample = DOCUMENTS_DIR / "sample.txt"
    if old_sample.exists():
        print(f"\n🗑️  Removing old sample.txt with encoding issues...")
        old_sample.unlink()
        print("   ✅ Removed")
    
    # Create clean sample documents
    print("\n📝 Creating clean sample documents...")
    
    # Document 1: AI Overview
    doc1 = DOCUMENTS_DIR / "ai_overview.txt"
    doc1.write_text("""Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI has become increasingly important in modern technology.

Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Deep Learning is a subset of machine learning based on artificial neural networks. The learning process is deep because the structure of neural networks consists of multiple input, output, and hidden layers. Each layer extracts different features from the input data.

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with machine learning and deep learning models.

Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.

AI systems can be classified into narrow AI and general AI. Narrow AI is designed to perform specific tasks, while general AI would have the ability to understand, learn, and apply knowledge across various domains.
""", encoding='utf-8')
    print(f"   ✅ Created: {doc1.name}")
    
    # Document 2: Machine Learning
    doc2 = DOCUMENTS_DIR / "machine_learning.txt"
    doc2.write_text("""Machine Learning Fundamentals

Machine Learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention.

Supervised Learning uses labeled training data to learn the relationship between input and output. The algorithm learns from examples where the correct answer is known. Common algorithms include linear regression, logistic regression, decision trees, random forests, and neural networks.

Unsupervised Learning works with unlabeled data to find hidden patterns or structures. The algorithm tries to find structure in the data without being told what to look for. Clustering and dimensionality reduction are common unsupervised learning techniques.

Reinforcement Learning is about taking suitable actions to maximize reward in a particular situation. The algorithm learns through trial and error, receiving rewards for correct actions and penalties for incorrect ones. This approach is used in robotics, gaming, and autonomous vehicles.

Training Data is crucial for machine learning models. The quality and quantity of training data directly impacts the model's performance and accuracy. Data preprocessing, cleaning, and augmentation are important steps in preparing training data.

Model Evaluation involves testing the trained model on new, unseen data to assess its performance. Common metrics include accuracy, precision, recall, F1 score, and mean squared error depending on the type of problem.
""", encoding='utf-8')
    print(f"   ✅ Created: {doc2.name}")
    
    # Document 3: AI Applications
    doc3 = DOCUMENTS_DIR / "ai_applications.txt"
    doc3.write_text("""AI Applications in the Real World

Healthcare: AI is revolutionizing healthcare through improved diagnostics, personalized treatment plans, drug discovery, and patient care. Medical imaging analysis using deep learning can detect diseases like cancer earlier and more accurately than traditional methods.

Finance: AI powers fraud detection systems, algorithmic trading, credit scoring, and risk assessment. Financial institutions use machine learning to analyze market trends and make investment decisions. Chatbots handle customer inquiries and account management.

Transportation: Self-driving cars use computer vision, sensor fusion, and deep learning to navigate roads safely. AI optimizes traffic flow, predicts maintenance needs, and improves logistics. Ride-sharing services use AI for route optimization and demand prediction.

Customer Service: Chatbots and virtual assistants powered by NLP provide 24/7 customer support, handle routine queries, and escalate complex issues to human agents. This improves response times and customer satisfaction while reducing costs.

Education: AI enables personalized learning experiences, automated grading, intelligent tutoring systems, and adaptive learning platforms that adjust to individual student needs. Virtual teaching assistants can answer student questions and provide feedback.

Manufacturing: AI improves quality control through computer vision inspection systems, predictive maintenance to prevent equipment failures, supply chain optimization, and robotic automation in manufacturing processes.

Retail: AI is used for personalized product recommendations, inventory management, price optimization, and demand forecasting. Visual search allows customers to find products by uploading images.

Agriculture: AI helps farmers optimize crop yields through precision agriculture, monitors crop health using drones and satellite imagery, predicts weather patterns, and automates harvesting with robotic systems.
""", encoding='utf-8')
    print(f"   ✅ Created: {doc3.name}")
    
    # Document 4: Python Programming
    doc4 = DOCUMENTS_DIR / "python_basics.txt"
    doc4.write_text("""Python Programming Basics

Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.

Variables in Python are created when you assign a value to them. Python is dynamically typed, meaning you don't need to declare variable types explicitly. Common data types include integers, floats, strings, lists, tuples, dictionaries, and sets.

Control Flow statements include if-else conditionals for decision making and loops like for and while for iteration. Python uses indentation to define code blocks, which makes the code more readable.

Functions are reusable blocks of code that perform specific tasks. They are defined using the def keyword followed by the function name and parameters. Functions can return values using the return statement.

Lists are ordered, mutable collections that can contain items of different types. They are created using square brackets and support operations like append, insert, remove, and sort.

Dictionaries are unordered collections of key-value pairs. They provide fast lookup times and are created using curly braces. Dictionaries are useful for storing related data.

Object-Oriented Programming in Python allows you to create classes that bundle data and functionality together. Classes are blueprints for creating objects, and they support inheritance, encapsulation, and polymorphism.

Python has a rich ecosystem of libraries and frameworks for various purposes including NumPy for numerical computing, Pandas for data analysis, Flask and Django for web development, and TensorFlow and PyTorch for machine learning.
""", encoding='utf-8')
    print(f"   ✅ Created: {doc4.name}")
    
    print("\n" + "="*60)
    print("✅ Document Cleaning Complete!")
    print("="*60)
    
    # List all documents
    all_docs = list(DOCUMENTS_DIR.glob("*.txt")) + \
               list(DOCUMENTS_DIR.glob("*.pdf")) + \
               list(DOCUMENTS_DIR.glob("*.md"))
    
    print(f"\n📚 Total documents in directory: {len(all_docs)}")
    for doc in all_docs:
        size = doc.stat().st_size
        print(f"   - {doc.name} ({size} bytes)")
    
    print("\n🎉 Ready to rebuild vector store!")
    print("Run: python rebuild_vectorstore.py")
    
    return True

if __name__ == "__main__":
    try:
        success = clean_documents()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
