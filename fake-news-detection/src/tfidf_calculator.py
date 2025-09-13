"""
TF-IDF Calculator
Implementasi sesuai dengan rumus dalam dokumen penelitian
"""

import numpy as np
import pandas as pd
from math import log
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class CustomTFIDFCalculator:
    """
    Implementasi TF-IDF manual sesuai rumus dalam dokumen
    """
    
    def __init__(self):
        self.vocabulary = None
        self.idf_values = None
        self.tf_idf_matrix = None
    
    def calculate_tf(self, document_tokens):
        """
        Perhitungan Term Frequency sesuai rumus 2.2:
        tf(t,d) = 1 + 10*log(f(t,d))
        dimana f(t,d) adalah frekuensi term t dalam dokumen d
        """
        tf_dict = {}
        total_terms = len(document_tokens)
        term_counts = Counter(document_tokens)
        
        for term, count in term_counts.items():
            if count > 0:
                # Rumus dari dokumen: tf = 1 + 10*log(f)
                tf_dict[term] = 1 + 10 * log(count)
            else:
                tf_dict[term] = 0
                
        return tf_dict
    
    def calculate_idf(self, documents_tokens):
        """
        Perhitungan Inverse Document Frequency sesuai rumus 2.3:
        idf(t) = 10*log(n/df(t))
        dimana n adalah total dokumen, df(t) adalah jumlah dokumen yang mengandung term t
        """
        n_documents = len(documents_tokens)
        
        # Hitung document frequency untuk setiap term
        term_doc_count = {}
        
        for document_tokens in documents_tokens:
            # Dapatkan unique terms dalam dokumen ini
            unique_terms = set(document_tokens)
            for term in unique_terms:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1
        
        # Hitung IDF
        idf_dict = {}
        for term, doc_freq in term_doc_count.items():
            # Rumus dari dokumen: idf = 10*log(n/df)
            idf_dict[term] = 10 * log(n_documents / doc_freq)
            
        return idf_dict
    
    def calculate_tfidf_matrix(self, documents_tokens):
        """
        Hitung TF-IDF matrix untuk semua dokumen
        """
        print("=== CALCULATING TF-IDF MATRIX ===")
        print("Menggunakan rumus dari dokumen penelitian\n")
        
        # Buat vocabulary dari semua dokumen
        all_terms = set()
        for doc_tokens in documents_tokens:
            all_terms.update(doc_tokens)
        
        self.vocabulary = sorted(list(all_terms))
        vocab_size = len(self.vocabulary)
        n_documents = len(documents_tokens)
        
        print(f"Vocabulary size: {vocab_size}")
        print(f"Number of documents: {n_documents}")
        
        # Hitung IDF untuk semua terms
        print("Calculating IDF values...")
        self.idf_values = self.calculate_idf(documents_tokens)
        
        # Buat TF-IDF matrix
        print("Building TF-IDF matrix...")
        tfidf_matrix = []
        
        for i, doc_tokens in enumerate(documents_tokens):
            if i % 1000 == 0:
                print(f"Processing document {i}/{n_documents}")
            
            # Hitung TF untuk dokumen ini
            tf_dict = self.calculate_tf(doc_tokens)
            
            # Buat vector TF-IDF untuk dokumen ini
            tfidf_vector = []
            for term in self.vocabulary:
                tf_val = tf_dict.get(term, 0)
                idf_val = self.idf_values.get(term, 0)
                tfidf_val = tf_val * idf_val
                tfidf_vector.append(tfidf_val)
            
            tfidf_matrix.append(tfidf_vector)
        
        self.tf_idf_matrix = np.array(tfidf_matrix)
        
        # Normalisasi sesuai rumus 2.5
        print("Applying normalization...")
        self.tf_idf_matrix = self.normalize_tfidf_matrix(self.tf_idf_matrix)
        
        print("TF-IDF calculation completed!")
        return self.tf_idf_matrix
    
    def normalize_tfidf_matrix(self, tfidf_matrix):
        """
        Normalisasi TF-IDF sesuai rumus 2.5:
        w(t,d) / sqrt(sum(w(t,d)^2))
        """
        normalized_matrix = []
        
        for row in tfidf_matrix:
            # Hitung norm (sqrt of sum of squares)
            norm = np.sqrt(np.sum(row ** 2))
            
            # Hindari pembagian dengan nol
            if norm > 0:
                normalized_row = row / norm
            else:
                normalized_row = row
                
            normalized_matrix.append(normalized_row)
        
        return np.array(normalized_matrix)
    
    def get_feature_names(self):
        """Return vocabulary/feature names"""
        return self.vocabulary if self.vocabulary else []
    
    def transform_new_document(self, document_tokens):
        """
        Transform dokumen baru menggunakan vocabulary dan IDF yang sudah ada
        """
        if self.vocabulary is None or self.idf_values is None:
            raise ValueError("Model belum difit. Jalankan calculate_tfidf_matrix terlebih dahulu.")
        
        # Hitung TF untuk dokumen baru
        tf_dict = self.calculate_tf(document_tokens)
        
        # Buat vector TF-IDF
        tfidf_vector = []
        for term in self.vocabulary:
            tf_val = tf_dict.get(term, 0)
            idf_val = self.idf_values.get(term, 0)
            tfidf_val = tf_val * idf_val
            tfidf_vector.append(tfidf_val)
        
        # Normalisasi
        tfidf_vector = np.array(tfidf_vector)
        norm = np.sqrt(np.sum(tfidf_vector ** 2))
        if norm > 0:
            tfidf_vector = tfidf_vector / norm
            
        return tfidf_vector.reshape(1, -1)

class SKLearnTFIDFWrapper:
    """
    Wrapper untuk TF-IDF sklearn untuk perbandingan
    """
    
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=False,  # sudah dilakukan di preprocessing
            token_pattern=None,
            tokenizer=lambda x: x.split(),  # tokens sudah dipisah spasi
            stop_words=None  # stopwords sudah dihapus di preprocessing
        )
        
    def fit_transform(self, processed_texts):
        """
        Fit dan transform text menggunakan sklearn TF-IDF
        """
        return self.vectorizer.fit_transform(processed_texts)
    
    def transform(self, processed_texts):
        """
        Transform text baru
        """
        return self.vectorizer.transform(processed_texts)
    
    def get_feature_names(self):
        """
        Get feature names
        """
        return self.vectorizer.get_feature_names_out()

# Demo penggunaan
if __name__ == "__main__":
    # Sample data sesuai dengan contoh di dokumen
    sample_docs = [
        ['vaksin', 'covid19', 'kandung', 'chip', 'bisa', 'kontrol', 'pikiran', 'manusia', 'hatihati', 'pemerintah', 'ingin', 'kuasa', 'rakyat'],
        ['pemerintah', 'umum', 'vaksin', 'covid19', 'aman', 'guna', 'uji', 'klinis', 'ketat', 'bpom']
    ]
    
    print("=== DEMO TF-IDF CALCULATION ===")
    print("Menggunakan sample data dari dokumen\n")
    
    # Manual calculation
    manual_calc = CustomTFIDFCalculator()
    tfidf_matrix = manual_calc.calculate_tfidf_matrix(sample_docs)
    
    print(f"\nTF-IDF Matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary: {manual_calc.get_feature_names()}")
    print(f"\nTF-IDF Matrix:")
    print(tfidf_matrix)
    
    # Sklearn comparison
    print("\n=== SKLEARN COMPARISON ===")
    sklearn_calc = SKLearnTFIDFWrapper()
    processed_texts = [' '.join(doc) for doc in sample_docs]
    sklearn_matrix = sklearn_calc.fit_transform(processed_texts)
    
    print(f"Sklearn TF-IDF Matrix shape: {sklearn_matrix.shape}")
    print(f"Sklearn TF-IDF Matrix:")
    print(sklearn_matrix.toarray())