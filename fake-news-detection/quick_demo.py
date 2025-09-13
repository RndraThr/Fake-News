"""
Quick Demo - Fake News Detection tanpa preprocessing lambat
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("    QUICK FAKE NEWS DETECTION DEMO    ")
    print("    Tanpa preprocessing lambat    ")
    print("=" * 60)

    # Sample data langsung
    sample_data = [
        ("Vaksin COVID mengandung chip 5G berbahaya untuk kesehatan!", 1),  # FAKE
        ("Pemerintah umumkan program vaksinasi gratis untuk warga", 0),     # REAL
        ("Obat ajaib sembuhkan semua penyakit dalam sekali minum!", 1),     # FAKE
        ("Menteri kesehatan laporkan penurunan kasus COVID resmi", 0),      # REAL
        ("Rahasia dokter yang disembunyikan pemerintah terungkap!", 1),     # FAKE
        ("WHO setujui protokol kesehatan baru untuk pencegahan", 0),        # REAL
        ("Air putih bisa mengobati kanker stadium 4 tanpa obat!", 1),       # FAKE
        ("Rumah sakit rujukan tambah fasilitas ICU COVID-19", 0),          # REAL
        ("Konspirasi besar dibalik vaksin yang harus kamu tahu!", 1),       # FAKE
        ("Kementerian kesehatan buka layanan telemedicine gratis", 0),      # REAL
    ] * 10  # Duplikasi untuk lebih banyak data

    print(f"\n>> Loading sample data: {len(sample_data)} records")

    # Prepare data
    texts = [item[0] for item in sample_data]
    labels = [item[1] for item in sample_data]

    print(">> Quick preprocessing (simple cleaning)...")
    # Simple preprocessing
    processed_texts = []
    for text in texts:
        # Basic cleaning only
        clean_text = text.lower().strip()
        processed_texts.append(clean_text)

    print(">> TF-IDF vectorization...")
    # TF-IDF with sklearn (fast)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(processed_texts)
    y = np.array(labels)

    print(">> Splitting data...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(">> Training model...")
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    print(">> Evaluating model...")
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n" + "=" * 50)
    print("    RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Demo predictions
    test_texts = [
        "Vaksin berbahaya mengandung zat kimia beracun!",
        "Pemerintah umumkan kebijakan kesehatan baru",
        "Obat herbal ajaib sembuhkan diabetes instant!",
        "Menteri kesehatan sampaikan laporan resmi"
    ]

    print(f"\n" + "=" * 50)
    print("    DEMO PREDICTIONS")
    print("=" * 50)

    for i, text in enumerate(test_texts, 1):
        # Simple preprocessing
        clean_text = text.lower().strip()

        # Vectorize
        features = vectorizer.transform([clean_text])

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        label = "FAKE" if prediction == 1 else "REAL"
        confidence = max(probability) * 100

        print(f"\n{i}. Text: {text}")
        print(f"   Prediction: {label} (Confidence: {confidence:.1f}%)")

    print(f"\n" + "=" * 50)
    print("    DEMO COMPLETE! ⚡")
    print("=" * 50)

if __name__ == "__main__":
    main()