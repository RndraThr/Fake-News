# 🔍 Fake News Detection System

Sistem deteksi berita palsu menggunakan Machine Learning dengan metodologi **Knowledge Discovery in Databases (KDD)** dan algoritma **Logistic Regression**.

## 📋 Overview

Project ini mengimplementasikan sistem klasifikasi berita palsu yang dapat membedakan antara berita asli dan berita palsu menggunakan teknik Natural Language Processing dan Machine Learning.

### ✨ Features
- 🤖 **Machine Learning**: Logistic Regression dengan TF-IDF features
- 📊 **KDD Methodology**: 5-stage preprocessing pipeline
- 🌐 **Web Interface**: Interactive Streamlit application
- 🔧 **Production Ready**: FastAPI REST API
- 📈 **Evaluation**: Comprehensive metrics dan visualization

## 🗂️ Project Structure

```
fake-news-detection/
├── 📁 src/                          # Source code utama
│   ├── preprocessing.py             # Data preprocessing (5 tahap KDD)
│   ├── tfidf_calculator.py          # TF-IDF calculation
│   ├── model.py                     # Logistic Regression model
│   └── evaluation.py                # Pattern evaluation
├── 📁 data/
│   ├── raw/                         # Dataset mentah
│   │   ├── train.csv                # Training data (24,353 records)
│   │   ├── test.csv                 # Test data (8,117 records)
│   │   └── evaluation.csv           # Evaluation data (8,117 records)
│   └── processed/                   # Data after preprocessing
├── 📁 app/                          # Web applications
│   ├── streamlit_app.py             # Streamlit web interface
│   └── fastapi_app.py               # FastAPI REST API
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Data exploration
│   ├── 02_preprocessing.ipynb       # Data preprocessing
│   ├── 03_model_training.ipynb      # Model training
│   └── 04_evaluation.ipynb          # Model evaluation
├── 📁 models/                       # Trained models (auto-generated)
│   ├── fake_news_model.pkl          # Trained Logistic Regression
│   ├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│   └── pipeline_info.pkl            # Model metadata
├── main_pipeline.py                 # Complete training pipeline
├── simple_demo.py                   # Simple command-line demo
├── standalone_demo.py               # Standalone demo
└── requirements.txt                 # Python dependencies
```

## 🔬 KDD Methodology

### 1. Data Selection
- **Dataset**: Kaggle Fake News Classification dataset
- **Training**: 24,353 samples (54.4% fake, 45.6% real)
- **Test**: 8,117 samples
- **Language**: Indonesian and English

### 2. Data Preprocessing (5 Stages)
1. **Case Folding**: Convert text to lowercase
2. **Punctuation Removal**: Remove special characters
3. **Tokenization**: Split text into tokens
4. **Stopword Removal**: Remove common words
5. **Stemming**: Indonesian stemming using Sastrawi

### 3. Data Transformation
- **TF-IDF Vectorization**: Convert text to numerical features
- **Feature Selection**: Top-k features based on TF-IDF scores

### 4. Data Mining
- **Algorithm**: Logistic Regression
- **Optimization**: Custom gradient descent implementation
- **Regularization**: L2 regularization support

### 5. Pattern Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion Matrix, ROC Curve
- **Cross-validation**: K-fold validation

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd fake-news-detection

# Create virtual environment
python -m venv fake_news_env
source fake_news_env/bin/activate  # Linux/Mac
# atau
fake_news_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model

```bash
# Run complete KDD pipeline
python main_pipeline.py
```

Output:
- Trained model saved to `models/fake_news_model.pkl`
- TF-IDF vectorizer saved to `models/tfidf_vectorizer.pkl`
- Training metrics and evaluation results

### 3. Run Applications

#### 🌐 Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```
- Access: http://localhost:8501
- Features: Interactive prediction, file upload, sample texts

#### 🔌 FastAPI REST API
```bash
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000
```
- Access: http://localhost:8000
- Docs: http://localhost:8000/docs
- Swagger UI for API testing

#### 💻 Command Line Demo
```bash
# Simple demo
python simple_demo.py

# Standalone demo (more detailed)
python standalone_demo.py
```

## 📊 Model Performance

### Training Results
- **Accuracy**: ~85-90% (varies by sample size)
- **Precision**: High precision for both classes
- **Recall**: Balanced recall across classes
- **F1-Score**: Optimal balance of precision/recall

### Sample Predictions

| Text Sample | Prediction | Confidence |
|-------------|------------|------------|
| "Menteri Kesehatan melaporkan data resmi..." | REAL | 89.2% |
| "Vaksin mengandung chip 5G berbahaya!" | FAKE | 94.7% |
| "Pemerintah umumkan program vaksinasi..." | REAL | 87.3% |
| "Obat ajaib sembuhkan semua penyakit!" | FAKE | 91.8% |

## 🛠️ Development

### Project Dependencies

```txt
pandas==2.3.2          # Data manipulation
numpy==2.3.3           # Numerical computing
scikit-learn==1.7.2    # Machine learning
matplotlib==3.10.6     # Plotting
seaborn==0.13.2        # Statistical visualization
nltk==3.9.1            # Natural language processing
Sastrawi==1.0.1        # Indonesian stemming
streamlit==1.49.1      # Web application framework
plotly==6.3.0          # Interactive plotting
jupyter==1.1.1         # Jupyter notebooks
ipykernel==6.30.1      # Jupyter kernel
```

### Code Structure

#### Core Modules
- **`preprocessing.py`**: Text preprocessing pipeline
- **`tfidf_calculator.py`**: TF-IDF vectorization (custom + sklearn)
- **`model.py`**: Logistic regression implementation
- **`evaluation.py`**: Model evaluation metrics

#### Applications
- **`streamlit_app.py`**: Web interface with interactive features
- **`fastapi_app.py`**: REST API for integration
- **`main_pipeline.py`**: Complete training workflow

### Custom Implementation Features

1. **Flexible TF-IDF**: Both custom implementation and sklearn wrapper
2. **Indonesian Support**: Sastrawi stemmer for Indonesian text
3. **Configurable Pipeline**: Adjustable preprocessing steps
4. **Real-time Prediction**: Optimized for live inference
5. **Model Persistence**: Pickle serialization for deployment

## 📈 Usage Examples

### Python API
```python
from src.preprocessing import TextPreprocessor
from src.model import FakeNewsLogisticRegression
import joblib

# Load trained components
preprocessor = TextPreprocessor()
model = FakeNewsLogisticRegression()
model.load_model('models/fake_news_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Make prediction
text = "Breaking news: miracle cure discovered!"
tokens = preprocessor.preprocess_text(text)
processed_text = ' '.join(tokens)
features = tfidf.transform([processed_text])
prediction = model.predict(features.toarray())
probability = model.predict_proba(features.toarray())

print(f"Prediction: {'FAKE' if prediction[0] == 1 else 'REAL'}")
print(f"Confidence: {max(probability[0]) * 100:.1f}%")
```

### REST API
```bash
# POST request for prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Menteri kesehatan mengumumkan program vaksinasi"}'

# Response
{
  "prediction": "REAL",
  "confidence": 89.2,
  "probabilities": {
    "REAL": 89.2,
    "FAKE": 10.8
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Kaggle Fake News Classification
- Indonesian Stemming: Sastrawi library
- Methodology: Knowledge Discovery in Databases (KDD)
- Framework: Streamlit for web interface
- API: FastAPI for REST endpoints

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Happy Fake News Detection! 🔍✨**