"""
Logistic Regression Model untuk Fake News Detection
Implementasi sesuai dengan metodologi dalam dokumen penelitian
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class FakeNewsLogisticRegression:
    """
    Implementasi Logistic Regression untuk klasifikasi berita palsu
    Sesuai dengan rumus sigmoid dalam dokumen
    """
    
    def __init__(self, random_state=42):
        # Inisialisasi model dengan parameter optimal
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear',  # baik untuk dataset kecil-menengah
            C=1.0  # regularization parameter
        )
        self.is_fitted = False
        self.feature_names = None
        
    def sigmoid(self, z):
        """
        Implementasi Sigmoid Function sesuai rumus 2.1:
        sigmoid(z) = 1 / (1 + e^(-z))
        """
        # Clip z untuk menghindari overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_probability_manual(self, X, coefficients, intercept):
        """
        Manual prediction menggunakan rumus sigmoid
        P(y=1|X) = sigmoid(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)
        """
        # Hitung z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ  
        z = np.dot(X, coefficients) + intercept
        
        # Aplikasikan sigmoid function
        probabilities = self.sigmoid(z)
        
        return probabilities
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Training model Logistic Regression
        """
        print("=== TRAINING LOGISTIC REGRESSION MODEL ===")
        print("Sesuai metodologi dalam dokumen penelitian\n")
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels distribution:")
        print(f"  Real news (0): {np.sum(y_train == 0)}")
        print(f"  Fake news (1): {np.sum(y_train == 1)}")
        
        # Training model
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        # Jika ada test data, evaluasi juga
        if X_test is not None and y_test is not None:
            y_test_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            print(f"Test accuracy: {test_accuracy:.4f}")
            
            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'model': self.model
            }
        
        return {
            'train_accuracy': train_accuracy,
            'model': self.model
        }
    
    def predict(self, X):
        """
        Prediksi kelas (0 atau 1)
        """
        if not self.is_fitted:
            raise ValueError("Model belum ditraining. Jalankan train() terlebih dahulu.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Prediksi probabilitas
        """
        if not self.is_fitted:
            raise ValueError("Model belum ditraining. Jalankan train() terlebih dahulu.")
        
        return self.model.predict_proba(X)
    
    def get_model_coefficients(self):
        """
        Dapatkan coefficients (β) dan intercept (β₀) dari model
        """
        if not self.is_fitted:
            raise ValueError("Model belum ditraining.")
        
        return {
            'coefficients': self.model.coef_[0],
            'intercept': self.model.intercept_[0],
            'feature_importance': np.abs(self.model.coef_[0])
        }
    
    def explain_prediction(self, X_sample, feature_names=None, top_features=10):
        """
        Penjelasan prediksi berdasarkan kontribusi fitur
        """
        if not self.is_fitted:
            raise ValueError("Model belum ditraining.")
        
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        # Prediksi
        prediction = self.predict(X_sample)[0]
        probability = self.predict_proba(X_sample)[0]
        
        # Coefficients
        coeffs = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        
        # Kontribusi setiap fitur
        contributions = X_sample[0] * coeffs
        
        # z value sebelum sigmoid
        z_value = np.sum(contributions) + intercept
        
        # Sigmoid manual
        sigmoid_prob = self.sigmoid(z_value)
        
        print(f"=== PREDICTION EXPLANATION ===")
        print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
        print(f"Probability: {probability}")
        print(f"Z-value (before sigmoid): {z_value:.4f}")
        print(f"Sigmoid probability: {sigmoid_prob:.4f}")
        
        if feature_names is not None:
            # Top contributing features
            feature_contributions = list(zip(feature_names, contributions))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\nTop {top_features} contributing features:")
            for i, (feature, contrib) in enumerate(feature_contributions[:top_features]):
                print(f"{i+1:2d}. {feature:<20} : {contrib:8.4f}")
        
        return {
            'prediction': prediction,
            'probability': probability,
            'z_value': z_value,
            'sigmoid_probability': sigmoid_prob,
            'contributions': contributions
        }
    
    def save_model(self, filepath):
        """
        Simpan model ke file
        """
        if not self.is_fitted:
            raise ValueError("Model belum ditraining.")
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model dari file
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")

def demonstrate_logistic_regression():
    """
    Demonstrasi Logistic Regression dengan sample data dari dokumen
    """
    print("=== LOGISTIC REGRESSION DEMONSTRATION ===")
    print("Menggunakan sample data dari dokumen penelitian\n")
    
    # Sample TF-IDF data dari dokumen (simplified)
    X_sample = np.array([
        [0.0, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231, 0.0231, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Hoax
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0301, 0.0301, 0.0301, 0.0301, 0.0301, 0.0301, 0.0301, 0.0, 0.0, 0.0, 0.0, 0.0]  # Real
    ])
    
    y_sample = np.array([1, 0])  # 1=Fake, 0=Real
    
    # Inisialisasi model
    model = FakeNewsLogisticRegression()
    
    # Training (dengan data sample yang sangat kecil)
    print("Training dengan sample data...")
    result = model.train(X_sample, y_sample)
    
    # Prediksi
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)
    
    print(f"\nPrediksi:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        true_label = "FAKE" if y_sample[i] == 1 else "REAL"
        pred_label = "FAKE" if pred == 1 else "REAL"
        print(f"Sample {i+1}: True={true_label}, Predicted={pred_label}, Prob={prob}")
    
    # Model coefficients
    coeffs = model.get_model_coefficients()
    print(f"\nModel Coefficients:")
    print(f"Intercept (beta_0): {coeffs['intercept']:.4f}")
    print(f"Number of features: {len(coeffs['coefficients'])}")
    
    # Demonstrasi manual sigmoid calculation
    print(f"\n=== MANUAL SIGMOID CALCULATION ===")
    sample_z = 0.1617  # dari contoh di dokumen
    manual_sigmoid = model.sigmoid(sample_z)
    print(f"Z = {sample_z}")
    print(f"Sigmoid(Z) = {manual_sigmoid:.4f}")
    print(f"Classification: {'FAKE' if manual_sigmoid > 0.5 else 'REAL'}")

if __name__ == "__main__":
    demonstrate_logistic_regression()