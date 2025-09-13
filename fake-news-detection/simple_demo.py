"""
Simple Demo - Fake News Detection KDD Pipeline
"""

import numpy as np
from math import log
from collections import Counter
import re

class SimpleFakeNewsDemo:
    def __init__(self):
        print("=" * 60)
        print("FAKE NEWS DETECTION - KDD PIPELINE DEMO")
        print("Sesuai metodologi dalam dokumen penelitian")
        print("=" * 60)
        
    def simple_preprocessing(self, text):
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
        z = max(-500, min(500, z))
        return 1 / (1 + np.exp(-z))
    
    def demonstrate_pipeline(self):
        # Sample data
        sample_data = [
            {'text': 'Vaksin COVID19 mengandung chip berbahaya yang bisa mengontrol pikiran manusia. Jangan divaksin karena pemerintah ingin menguasai rakyat!', 'label': 1},
            {'text': 'Pemerintah mengumumkan program vaksinasi COVID19 gratis untuk seluruh warga negara sesuai protokol kesehatan WHO dan telah melalui uji klinis ketat.', 'label': 0},
            {'text': 'Ilmuwan menemukan obat ajaib yang bisa menyembuhkan semua penyakit dalam satu hari tanpa efek samping apapun!', 'label': 1},
            {'text': 'Menteri Kesehatan melaporkan penurunan kasus COVID19 sebesar 15 persen dalam minggu ini berdasarkan data surveilans resmi.', 'label': 0}
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
        
        # 3. LOGISTIC REGRESSION SIMULATION
        print("\n=== LOGISTIC REGRESSION SIMULATION ===")
        
        np.random.seed(42)
        weights = np.random.randn(len(vocabulary)) * 0.1
        bias = 0.1
        
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
            correct = "OK" if pred == true_label else "NO"
            
            print(f"Doc {i+1}: Z={z:.3f}, P={prob:.3f}, Pred={pred_text}, True={true_text} {correct}")
        
        # 4. EVALUATION
        print("\n=== EVALUATION ===")
        true_labels = [item['label'] for item in sample_data]
        
        tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0) 
        fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
        
        print("CONFUSION MATRIX:")
        print("                 Predicted")
        print("              Fake    Real")
        print(f"Actual  Fake   {tp:4d}    {fn:4d}")
        print(f"        Real   {fp:4d}    {tn:4d}")
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMETRICS:")
        print(f"Accuracy  = {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"Precision = {precision:.4f} ({precision*100:.1f}%)")
        print(f"Recall    = {recall:.4f} ({recall*100:.1f}%)")
        print(f"F1-Score  = {f1_score:.4f} ({f1_score*100:.1f}%)")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("Pipeline KDD telah didemonstrasikan sesuai dokumen")
        print("=" * 60)

def main():
    demo = SimpleFakeNewsDemo()
    demo.demonstrate_pipeline()

if __name__ == "__main__":
    main()