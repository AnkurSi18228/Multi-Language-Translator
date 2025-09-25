# 🌍 Advanced Multi-Language Translator

An AI-powered multilingual translation tool supporting 50+ languages with direct translation capabilities. Built with Streamlit and powered by Facebook's M2M100 transformer model.

## ✨ Features

- 50+ Language Support: Direct translation between multiple language pairs
- AI-Powered: Uses Facebook's M2M100 transformer model for accurate translations
- Real-time Translation: Instant translation with caching for improved performance
- Translation History: Keep track of your previous translations
- Favorite Languages: Quick access to frequently used language pairs
- Responsive Design: Modern, user-friendly interface with dark/light theme support
- GPU Acceleration: Automatic CUDA support for faster translations

## 🚀 Supported Languages

English, Hindi, French, German, Spanish, Chinese (Simplified), Japanese, Korean, Arabic, Russian, Portuguese, Italian, Dutch, Swedish, Norwegian, Danish, Finnish, Polish, Czech, Hungarian, Turkish, Greek, Hebrew, Thai, Vietnamese, Indonesian, Malay, Tamil, Telugu, Bengali, Gujarati, Marathi, Punjabi, Urdu, Swahili, Amharic, and many more!

## 📋 Prerequisites

- Python 3.11 or higher
- 4GB+ RAM (8GB+ recommended)
- Internet connection (for initial model download)
- Optional: NVIDIA GPU with CUDA support for faster performance

## 🛠️ Installation

### 1. Clone the Repository
git clone https://github.com/AnkurSi18228/Multi-Language-Translator.git


### 2. Create Virtual Environment (Recommended)
Create a virtual environment
python -m venv translator_env

Activate virtual environment
On Windows:
translator_env\Scripts\activate

On macOS/Linux:
source translator_env/bin/activate


### 3. Install Dependencies
pip install -r requirements.txt


## 🚀 Running the Application
### Local Development
streamlit run multitranslator.py


The application will start and be available at http://localhost:8501

### Production Deployment
streamlit run multitranslator.py --server.port 8501 --server.address 0.0.0.0

