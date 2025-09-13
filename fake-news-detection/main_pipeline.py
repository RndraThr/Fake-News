"""
Complete KDD Pipeline untuk Fake News Detection
Implementasi end-to-end sesuai metodologi dalam dokumen penelitian
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

# Import custom modules
from src.preprocessing import TextPreprocessor
from src.tfidf_calculator import CustomTFIDFCalculator, SKLearnTFIDFWrapper  
from src.model import FakeNewsLogisticRegression
from src.evaluation import ModelEvaluator

class FakeNewsDetectionPipeline:
    """
    Complete pipeline untuk fake news detection menggunakan KDD methodology
    """
    
    def __init__(self, use_custom_tfidf=True):
        self.preprocessor = TextPreprocessor()
        self.use_custom_tfidf = use_custom_tfidf
        
        if use_custom_tfidf:
            self.tfidf_calculator = CustomTFIDFCalculator()
        else:
            self.tfidf_calculator = SKLearnTFIDFWrapper(max_features=5000)
            
        self.model = FakeNewsLogisticRegression()
        self.evaluator = ModelEvaluator()
        
        self.is_trained = False
    
    def load_dataset(self, train_path, test_path=None, evaluation_path=None):
        """
        Load dataset dari file CSV
        """
        print("=" * 60)
        print("LOADING DATASET")
        print("=" * 60)
        
        # Load training data
        print("Loading training data...")
        self.train_df = pd.read_csv(train_path, sep=';', on_bad_lines='skip', encoding='utf-8')
        print(f"Training data loaded: {len(self.train_df)} records")
        print(f"Columns: {list(self.train_df.columns)}")
        
        # Label distribution
        label_dist = self.train_df['label'].value_counts()
        print(f"\nLabel distribution:")
        print(f"  REAL news (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(self.train_df)*100:.1f}%)")
        print(f"  FAKE news (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(self.train_df)*100:.1f}%)")
        
        # Load test data if provided
        if test_path and os.path.exists(test_path):
            print(f"\nLoading test data...")
            self.test_df = pd.read_csv(test_path, sep=';', on_bad_lines='skip', encoding='utf-8')
            print(f"Test data loaded: {len(self.test_df)} records")
        else:
            self.test_df = None
            
        # Load evaluation data if provided
        if evaluation_path and os.path.exists(evaluation_path):
            print(f"\nLoading evaluation data...")
            self.eval_df = pd.read_csv(evaluation_path, sep=';', on_bad_lines='skip', encoding='utf-8')
            print(f"Evaluation data loaded: {len(self.eval_df)} records")
        else:
            self.eval_df = None
        
        print("\nDataset loading completed!")
        return self.train_df
    
    def preprocess_data(self, sample_size=None):
        """
        Preprocessing semua dataset menggunakan 5 tahap KDD
        """
        print("\n" + "=" * 60)
        print("        DATA PREPROCESSING PHASE        ")
        print("=" * 60)
        
        # Sample data jika diminta (untuk testing)
        if sample_size and sample_size < len(self.train_df):
            print(f"Using sample of {sample_size} records for faster processing...")
            self.train_df_processed = self.train_df.sample(n=sample_size, random_state=42)
        else:
            self.train_df_processed = self.train_df.copy()
        
        # Preprocessing training data
        print("Preprocessing training data...")
        self.train_df_processed = self.preprocessor.preprocess_dataset(
            self.train_df_processed, 
            text_column='text', 
            show_progress=True
        )
        
        # Preprocessing test data jika ada
        if self.test_df is not None:
            print("\nPreprocessing test data...")
            self.test_df_processed = self.preprocessor.preprocess_dataset(
                self.test_df,
                text_column='text',
                show_progress=True
            )
        
        print("\nData preprocessing completed!")
        return self.train_df_processed
    
    def extract_features(self):
        """
        Feature extraction menggunakan TF-IDF
        """
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION PHASE")
        print("=" * 60)
        
        if self.use_custom_tfidf:
            print("Using custom TF-IDF implementation...")
            
            # Convert tokens to documents for TF-IDF
            train_documents = self.train_df_processed['processed_tokens'].tolist()
            
            # Calculate TF-IDF matrix
            self.X_train = self.tfidf_calculator.calculate_tfidf_matrix(train_documents)
            self.feature_names = self.tfidf_calculator.get_feature_names()
            
        else:
            print("Using sklearn TF-IDF implementation...")
            
            # Convert to text format for sklearn
            train_texts = self.train_df_processed['processed_text'].tolist()
            
            # Fit and transform
            self.X_train = self.tfidf_calculator.fit_transform(train_texts).toarray()
            self.feature_names = self.tfidf_calculator.get_feature_names()
        
        # Labels
        self.y_train = self.train_df_processed['label'].values
        
        print(f"\nFeature extraction completed!")
        print(f"Feature matrix shape: {self.X_train.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        return self.X_train, self.y_train
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Model training menggunakan Logistic Regression
        """
        print("\n" + "=" * 60)
        print("         MODEL TRAINING PHASE         ")
        print("=" * 60)
        
        # Split data untuk validasi
        self.X_train_split, self.X_val_split, self.y_train_split, self.y_val_split = train_test_split(
            self.X_train, self.y_train, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y_train
        )
        
        print(f"Training set: {self.X_train_split.shape[0]} samples")
        print(f"Validation set: {self.X_val_split.shape[0]} samples")
        
        # Training model
        training_result = self.model.train(
            self.X_train_split, 
            self.y_train_split,
            self.X_val_split,
            self.y_val_split
        )
        
        self.is_trained = True
        print("\nModel training completed!")
        return training_result
    
    def evaluate_model(self):
        """
        Evaluasi model menggunakan confusion matrix dan metrik
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION PHASE")
        print("=" * 60)
        
        if not self.is_trained:
            raise ValueError("Model belum ditraining!")
        
        # Prediksi pada validation set
        y_pred = self.model.predict(self.X_val_split)
        
        # Generate evaluation report
        metrics = self.evaluator.print_evaluation_report(
            self.y_val_split, 
            y_pred, 
            model_name="Logistic Regression"
        )
        
        # Compare dengan penelitian lain
        self.evaluator.compare_with_research(metrics)
        
        return metrics
    
    def predict_new_text(self, text, explain=True):
        """
        Prediksi untuk teks baru
        """
        if not self.is_trained:
            raise ValueError("Model belum ditraining!")
        
        print(f"\n=== PREDICTION FOR NEW TEXT ===")
        print(f"Input text: {text[:200]}...")
        
        # Preprocess text
        processed_tokens = self.preprocessor.preprocess_text(text)
        
        if self.use_custom_tfidf:
            # Transform menggunakan custom TF-IDF
            tfidf_vector = self.tfidf_calculator.transform_new_document(processed_tokens)
        else:
            # Transform menggunakan sklearn
            processed_text = ' '.join(processed_tokens)
            tfidf_vector = self.tfidf_calculator.transform([processed_text]).toarray()
        
        # Prediksi
        prediction = self.model.predict(tfidf_vector)[0]
        probability = self.model.predict_proba(tfidf_vector)[0]
        
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': probability[prediction] * 100,
            'probabilities': {
                'REAL': probability[0] * 100,
                'FAKE': probability[1] * 100
            }
        }
        
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Probabilities: REAL={result['probabilities']['REAL']:.2f}%, FAKE={result['probabilities']['FAKE']:.2f}%")
        
        if explain:
            # Detailed explanation
            self.model.explain_prediction(
                tfidf_vector[0], 
                feature_names=self.feature_names[:20],  # Top 20 features only
                top_features=10
            )
        
        return result
    
    def save_pipeline(self, save_dir="models"):
        """
        Simpan seluruh pipeline
        """
        if not self.is_trained:
            raise ValueError("Model belum ditraining!")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "fake_news_model.pkl")
        self.model.save_model(model_path)
        
        # Save TF-IDF vectorizer
        tfidf_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
        joblib.dump(self.tfidf_calculator, tfidf_path)
        
        # Save pipeline info
        pipeline_info = {
            'use_custom_tfidf': self.use_custom_tfidf,
            'feature_count': len(self.feature_names),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'training_samples': len(self.X_train_split),
            'validation_samples': len(self.X_val_split)
        }
        
        info_path = os.path.join(save_dir, "pipeline_info.pkl")
        joblib.dump(pipeline_info, info_path)
        
        print(f"\nPipeline saved to {save_dir}/")
        return save_dir

def main():
    """
    Demonstrasi complete pipeline
    """
    print("=" * 80)
    print("    FAKE NEWS DETECTION - COMPLETE KDD PIPELINE    ")
    print("    Sesuai metodologi dalam dokumen penelitian    ")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = FakeNewsDetectionPipeline(use_custom_tfidf=True)
    
    # 1. Load dataset
    dataset_paths = {
        'train': 'data/raw/train.csv',
        'test': 'data/raw/test.csv', 
        'evaluation': 'data/raw/evaluation.csv'
    }
    
    pipeline.load_dataset(
        train_path=dataset_paths['train'],
        test_path=dataset_paths['test'],
        evaluation_path=dataset_paths['evaluation']
    )
    
    # 2. Preprocess data (sample untuk demo)
    pipeline.preprocess_data(sample_size=2000)  # Sample sesuai rencana awal
    
    # 3. Extract features
    pipeline.extract_features()
    
    # 4. Train model
    pipeline.train_model()
    
    # 5. Evaluate model
    metrics = pipeline.evaluate_model()
    
    # 6. Demo prediction
    sample_texts = [
        "Vaksin COVID-19 mengandung chip yang bisa mengontrol pikiran manusia! Jangan divaksin!",
        "Pemerintah mengumumkan vaksinasi COVID-19 gratis untuk semua warga negara sesuai protokol kesehatan WHO."
    ]
    
    print("\n" + "=" * 60)
    print("           DEMO PREDICTIONS           ")
    print("=" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Sample Prediction {i} ---")
        result = pipeline.predict_new_text(text, explain=False)
    
    # 7. Save pipeline
    pipeline.save_pipeline()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()