"""
Standalone Demo - Fake News Detection KDD Pipeline
Demonstrasi sederhana tanpa dependency issues
"""

import numpy as np
import pandas as pd
from math import log
from collections import Counter
import re
import string

class SimpleFakeNewsDemo:
    """
    Demonstrasi sederhana implementasi KDD pipeline
    """
    
    def __init__(self):
        print("=" * 60)
        print("FAKE NEWS DETECTION - KDD PIPELINE DEMO")
        print("Sesuai metodologi dalam dokumen penelitian")
        print("=" * 60)
        
    def simple_preprocessing(self, text):
        """
        Preprocessing sederhana tanpa external dependencies
        """
        if not text:
            return []
        
        # 1. Case folding
        text = text.lower()
        
        # 2. Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # 3. Tokenization
        tokens = text.split()
        
        # 4. Simple stopword removal
        stopwords = {'dan', 'atau', 'dengan', 'dalam', 'untuk', 'pada', 'dari', 'ke', 'yang', 'adalah', 'ini', 'itu', 'tidak', 'ada', 'akan', 'dapat', 'sudah', 'telah', 'juga', 'saja', 'hanya'}
        tokens = [token for token in tokens if token not in stopwords and len(token) >= 3]
        
        return tokens
    
    def calculate_tf_idf_simple(self, documents):
        """
        Implementasi TF-IDF sederhana sesuai rumus dokumen
        """
        # Build vocabulary
        vocab = set()
        for doc in documents:
            vocab.update(doc)
        vocab = sorted(list(vocab))
        
        n_docs = len(documents)
        
        # Calculate IDF
        idf = {}
        for term in vocab:
            df = sum(1 for doc in documents if term in doc)
            idf[term] = log(n_docs / df) if df > 0 else 0
        
        # Calculate TF-IDF matrix
        tfidf_matrix = []
        for doc in documents:
            term_counts = Counter(doc)
            tfidf_vector = []
            
            for term in vocab:
                # TF calculation: 1 + log(f) jika f > 0, else 0
                tf = 1 + log(term_counts[term]) if term_counts[term] > 0 else 0
                tfidf_val = tf * idf[term]
                tfidf_vector.append(tfidf_val)
            
            # Normalization
            norm = sum(val**2 for val in tfidf_vector) ** 0.5
            if norm > 0:
                tfidf_vector = [val/norm for val in tfidf_vector]
            
            tfidf_matrix.append(tfidf_vector)
        
        return np.array(tfidf_matrix), vocab
    
    def sigmoid(self, z):
        """
        Sigmoid function: 1 / (1 + e^(-z))
        """
        z = max(-500, min(500, z))  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def demonstrate_pipeline(self):
        """
        Demonstrasi complete pipeline dengan sample data
        """
        # Sample data dari dokumen penelitian
        sample_data = [
            {
                'text': 'Vaksin COVID19 mengandung chip berbahaya yang bisa mengontrol pikiran manusia. Jangan divaksin karena pemerintah ingin menguasai rakyat!',
                'label': 1  # FAKE
            },
            {
                'text': 'Pemerintah mengumumkan program vaksinasi COVID19 gratis untuk seluruh warga negara sesuai protokol kesehatan WHO dan telah melalui uji klinis ketat.',
                'label': 0  # REAL
            },
            {
                'text': 'Ilmuwan menemukan obat ajaib yang bisa menyembuhkan semua penyakit dalam satu hari tanpa efek samping apapun!',
                'label': 1  # FAKE
            },
            {
                'text': 'Menteri Kesehatan melaporkan penurunan kasus COVID19 sebesar 15 persen dalam minggu ini berdasarkan data surveilans resmi.',
                'label': 0  # REAL
            },
            {
                'text': 'BREAKING NEWS: Alien sudah mendarat di Jakarta dan akan mengambil alih pemerintahan Indonesia dalam 24 jam!',
                'label': 1  # FAKE
            },
            {
                'text': 'Badan Meteorologi melaporkan cuaca cerah hingga berawan di sebagian besar wilayah Indonesia untuk minggu depan.',
                'label': 0  # REAL
            }
        ]
        
        print("\n=== SAMPLE DATA ===")
        for i, item in enumerate(sample_data):
            label_text = "FAKE" if item['label'] == 1 else "REAL"
            print(f"{i+1}. [{label_text}] {item['text'][:80]}...")
        
        # 1. PREPROCESSING
        print("\n=== PREPROCESSING PHASE ===")
        processed_docs = []
        
        for i, item in enumerate(sample_data):
            tokens = self.simple_preprocessing(item['text'])
            processed_docs.append(tokens)
            print(f"Doc {i+1}: {tokens[:10]}...")
        
        # 2. FEATURE EXTRACTION (TF-IDF)
        print("\n=== FEATURE EXTRACTION (TF-IDF) ===")
        tfidf_matrix, vocabulary = self.calculate_tf_idf_simple(processed_docs)
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(vocabulary)}")
        print(f"Sample vocabulary: {vocabulary[:15]}...")
        
        # 3. SIMPLE LOGISTIC REGRESSION SIMULATION
        print("\n=== LOGISTIC REGRESSION SIMULATION ===")
        
        # Simple manual weights untuk demo (biasanya dari training)
        np.random.seed(42)
        weights = np.random.randn(len(vocabulary)) * 0.1
        bias = 0.1
        
        # Manual prediction
        predictions = []
        probabilities = []
        
        for i, features in enumerate(tfidf_matrix):
            z = np.dot(features, weights) + bias
            prob = self.sigmoid(z)
            pred = 1 if prob > 0.5 else 0
            
            predictions.append(pred)
            probabilities.append(prob)
            
            true_label = sample_data[i]['label']
            pred_text = "FAKE" if pred == 1 else "REAL"
            true_text = "FAKE" if true_label == 1 else "REAL"
            correct = "✓" if pred == true_label else "✗"
            
            print(f"Doc {i+1}: Z={z:.3f}, P={prob:.3f}, Pred={pred_text}, True={true_text} {correct}")
        
        # 4. EVALUATION
        print("\n=== EVALUATION ===")
        true_labels = [item['label'] for item in sample_data]
        
        # Confusion Matrix calculation
        tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0) 
        fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
        
        print("CONFUSION MATRIX:")
        print("                 Predicted")
        print("              Fake    Real")
        print(f"Actual  Fake   {tp:4d}    {fn:4d}")
        print(f"        Real   {fp:4d}    {tn:4d}")
        
        # Metrics sesuai rumus dokumen
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMETRICS (Manual Calculation):")
        print(f"Accuracy  = (TP + TN) / (TP + TN + FP + FN)")
        print(f"          = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})")
        print(f"          = {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"Precision = {precision:.4f} ({precision*100:.1f}%)")
        print(f"Recall    = {recall:.4f} ({recall*100:.1f}%)")
        print(f"F1-Score  = {f1_score:.4f} ({f1_score*100:.1f}%)")
        
        # 5. DEMO PREDICTION
        print("\n=== DEMO NEW PREDICTION ===")
        new_text = "Vitamin C dosis tinggi bisa mencegah COVID19 100 persen tanpa perlu vaksin!"
        
        print(f"New text: {new_text}")
        
        # Preprocess
        new_tokens = self.simple_preprocessing(new_text)
        print(f"Processed: {new_tokens}")
        
        # Convert to TF-IDF (simplified)
        new_features = [0.0] * len(vocabulary)
        token_counts = Counter(new_tokens)
        
        for i, term in enumerate(vocabulary):
            if term in token_counts:
                tf = 1 + log(token_counts[term])
                # Use average IDF for simplicity
                new_features[i] = tf * 0.5
        
        # Normalize
        norm = sum(val**2 for val in new_features) ** 0.5
        if norm > 0:
            new_features = [val/norm for val in new_features]
        
        # Predict
        z_new = np.dot(new_features, weights) + bias
        prob_new = self.sigmoid(z_new)
        pred_new = 1 if prob_new > 0.5 else 0
        
        print(f"\\nPrediction Result:")
        print(f"Z-value: {z_new:.4f}")
        print(f"Probability: {prob_new:.4f}")
        print(f"Classification: {'FAKE' if pred_new == 1 else 'REAL'}")
        print(f"Confidence: {max(prob_new, 1-prob_new)*100:.1f}%")
        
        print("\\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("Semua komponen KDD pipeline telah didemonstrasikan")
        print("=" * 60)

def main():
    """
    Main function
    """
    demo = SimpleFakeNewsDemo()
    demo.demonstrate_pipeline()

if __name__ == "__main__":
    main()