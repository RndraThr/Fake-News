"""
Data Preprocessing Module untuk Fake News Detection
Implementasi sesuai dengan metodologi dalam dokumen penelitian

5 Tahap Preprocessing:
1. Case Folding - mengubah ke huruf kecil
2. Punctuation Removal - menghapus tanda baca  
3. Tokenization - memecah teks menjadi kata
4. Stopword Removal - menghapus kata umum
5. Stemming - mengubah ke kata dasar
"""

import re
import string
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class TextPreprocessor:
    def __init__(self):
        # Download NLTK requirements
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize stemmer untuk bahasa Indonesia
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Set stopwords (gabungan English + Indonesian)
        self.stop_words_en = set(stopwords.words('english'))
        self.stop_words_id = set(stopwords.words('indonesian'))
        self.stop_words = self.stop_words_en.union(self.stop_words_id)
        
        # Tambah custom stopwords sesuai domain berita
        custom_stopwords = {
            'said', 'says', 'told', 'news', 'report', 'reports', 'according', 'sources',
            'reuters', 'ap', 'cnn', 'bbc', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        self.stop_words.update(custom_stopwords)
    
    def case_folding(self, text):
        """
        Tahap 1: Case Folding
        Mengubah semua huruf menjadi huruf kecil
        """
        if pd.isna(text):
            return ""
        return str(text).lower()
    
    def remove_punctuation(self, text):
        """
        Tahap 2: Punctuation Removal  
        Menghapus tanda baca dan karakter khusus
        """
        if not text:
            return ""
        
        # Hapus URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Hapus email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        
        # Hapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Hapus karakter khusus dan unicode
        text = re.sub(r'[^\w\s]', '', text)
        
        # Hapus extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tahap 3: Tokenization
        Memecah teks menjadi list kata-kata
        """
        if not text:
            return []
        
        # Gunakan NLTK word tokenizer
        tokens = word_tokenize(text)
        
        # Filter token yang terlalu pendek (< 2 karakter)
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Tahap 4: Stopword Removal
        Menghapus kata-kata umum yang tidak bermakna
        """
        if not tokens:
            return []
        
        # Filter stopwords
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        return filtered_tokens
    
    def stemming(self, tokens):
        """
        Tahap 5: Stemming
        Mengubah kata ke bentuk dasar menggunakan Sastrawi
        """
        if not tokens:
            return []
        
        # Stem setiap token
        stemmed_tokens = []
        for token in tokens:
            try:
                stemmed = self.stemmer.stem(token)
                if stemmed:  # pastikan hasil stemming tidak kosong
                    stemmed_tokens.append(stemmed)
            except:
                # Jika error, gunakan token asli
                stemmed_tokens.append(token)
        
        return stemmed_tokens
    
    def preprocess_text(self, text):
        """
        Jalankan semua tahap preprocessing secara berurutan
        """
        # Tahap 1: Case Folding
        text = self.case_folding(text)
        
        # Tahap 2: Punctuation Removal
        text = self.remove_punctuation(text)
        
        # Tahap 3: Tokenization
        tokens = self.tokenize(text)
        
        # Tahap 4: Stopword Removal
        tokens = self.remove_stopwords(tokens)
        
        # Tahap 5: Stemming
        tokens = self.stemming(tokens)
        
        return tokens
    
    def preprocess_dataset(self, df, text_column='text', show_progress=True):
        """
        Preprocessing seluruh dataset
        """
        print("=== STARTING DATA PREPROCESSING ===")
        print("Sesuai 5 tahap dalam metodologi KDD:\n")
        
        # Copy dataframe
        processed_df = df.copy()
        
        # Preprocessing setiap teks
        processed_texts = []
        total_texts = len(df)
        start_time = time.time()

        # Optimized progress display (less frequent updates)
        progress_interval = max(1, total_texts // 20)  # Update 20 times max

        for idx, text in enumerate(df[text_column]):
            # Process text
            processed_tokens = self.preprocess_text(text)
            processed_texts.append(processed_tokens)

            # Show progress less frequently for speed
            if show_progress and (idx % progress_interval == 0 or idx == total_texts - 1):
                elapsed = time.time() - start_time
                progress = (idx + 1) / total_texts * 100

                if idx > 0 and elapsed > 0:
                    eta = elapsed / (idx + 1) * (total_texts - idx - 1)
                    eta_str = f"ETA: {int(eta//60)}m {int(eta%60)}s"
                    speed = (idx + 1) / elapsed
                else:
                    eta_str = "Calculating..."
                    speed = 0

                # Progress bar
                bar_length = 25
                filled = int(bar_length * progress / 100)
                bar = "=" * filled + "-" * (bar_length - filled)

                print(f"\r>> Processing: [{bar}] {progress:.1f}% ({idx+1:,}/{total_texts:,}) | {eta_str} | Speed: {speed:.1f} texts/sec", end="", flush=True)

        if show_progress:
            total_elapsed = time.time() - start_time
            print(f"\n>> Preprocessing complete! Total time: {int(total_elapsed//60)}m {int(total_elapsed%60)}s")
        
        # Simpan hasil preprocessing
        processed_df['processed_tokens'] = processed_texts
        processed_df['processed_text'] = processed_df['processed_tokens'].apply(lambda x: ' '.join(x))
        
        # Statistik hasil preprocessing
        print(f"\n=== PREPROCESSING COMPLETED ===")
        print(f"Original texts: {len(df)}")
        print(f"Processed texts: {len(processed_df)}")
        
        # Hapus teks kosong setelah preprocessing
        before_filter = len(processed_df)
        processed_df = processed_df[processed_df['processed_text'].str.len() > 0]
        after_filter = len(processed_df)
        
        if before_filter != after_filter:
            print(f"Removed {before_filter - after_filter} empty texts after preprocessing")
        
        return processed_df

# Contoh penggunaan untuk testing
if __name__ == "__main__":
    # Test preprocessing dengan sample text
    preprocessor = TextPreprocessor()
    
    sample_text = "COVID-19 Vaccines Are DANGEROUS! According to FAKE news sources, vaccines contain chips that control your mind. This is absolutely FALSE information!"
    
    print("=== SAMPLE PREPROCESSING DEMO ===")
    print(f"Original: {sample_text}")
    print()
    
    # Tahap demi tahap
    step1 = preprocessor.case_folding(sample_text)
    print(f"1. Case Folding: {step1}")
    
    step2 = preprocessor.remove_punctuation(step1)
    print(f"2. Punctuation Removal: {step2}")
    
    step3 = preprocessor.tokenize(step2)
    print(f"3. Tokenization: {step3}")
    
    step4 = preprocessor.remove_stopwords(step3)
    print(f"4. Stopword Removal: {step4}")
    
    step5 = preprocessor.stemming(step4)
    print(f"5. Stemming: {step5}")
    
    # Full preprocessing
    final_result = preprocessor.preprocess_text(sample_text)
    print(f"\nFinal Result: {final_result}")