"""
Pattern Evaluation untuk Fake News Detection
Implementasi Confusion Matrix dan metrik evaluasi sesuai dokumen penelitian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)

class ModelEvaluator:
    """
    Evaluasi model menggunakan Confusion Matrix dan metrik sesuai rumus dalam dokumen
    """
    
    def __init__(self):
        self.confusion_matrix = None
        self.metrics = {}
    
    def calculate_confusion_matrix(self, y_true, y_pred):
        """
        Hitung Confusion Matrix
        Format sesuai dokumen:
        
                 Predicted
              Positive  Negative  
        Actual
        Positive   TP      FN
        Negative   FP      TN
        """
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrix = cm
        
        # Extract TP, TN, FP, FN
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError("Confusion matrix harus berukuran 2x2 untuk klasifikasi biner")
        
        return {
            'TP': tp,  # True Positive (prediksi FAKE benar)
            'TN': tn,  # True Negative (prediksi REAL benar) 
            'FP': fp,  # False Positive (prediksi FAKE salah)
            'FN': fn   # False Negative (prediksi REAL salah)
        }
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Hitung metrik evaluasi sesuai rumus dalam dokumen:
        
        1. Accuracy = (TP + TN) / (TP + TN + FP + FN)     - Rumus 2.6
        2. Precision = TP / (TP + FP)                     - Rumus 2.7  
        3. Recall = TP / (TP + FN)                        - Rumus 2.8
        4. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)  - Rumus 2.9
        """
        
        # Hitung confusion matrix elements
        cm_elements = self.calculate_confusion_matrix(y_true, y_pred)
        tp, tn, fp, fn = cm_elements['TP'], cm_elements['TN'], cm_elements['FP'], cm_elements['FN']
        
        # Hitung metrik sesuai rumus dokumen
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Verifikasi dengan sklearn
        sklearn_accuracy = accuracy_score(y_true, y_pred)
        sklearn_precision = precision_score(y_true, y_pred, zero_division=0)
        sklearn_recall = recall_score(y_true, y_pred, zero_division=0)
        sklearn_f1 = f1_score(y_true, y_pred, zero_division=0)
        
        self.metrics = {
            'confusion_matrix_elements': cm_elements,
            'manual_metrics': {
                'accuracy': accuracy,
                'precision': precision, 
                'recall': recall,
                'f1_score': f1
            },
            'sklearn_metrics': {
                'accuracy': sklearn_accuracy,
                'precision': sklearn_precision,
                'recall': sklearn_recall,
                'f1_score': sklearn_f1
            }
        }
        
        return self.metrics
    
    def print_evaluation_report(self, y_true, y_pred, model_name="Logistic Regression"):
        """
        Print laporan evaluasi lengkap sesuai format dokumen
        """
        print("=" * 60)
        print(f"PATTERN EVALUATION REPORT - {model_name}")
        print("=" * 60)
        print("Sesuai metodologi KDD dalam dokumen penelitian\n")
        
        # Hitung metrik
        metrics = self.calculate_metrics(y_true, y_pred)
        cm_elements = metrics['confusion_matrix_elements']
        manual_metrics = metrics['manual_metrics']
        sklearn_metrics = metrics['sklearn_metrics']
        
        # Print Confusion Matrix
        print("CONFUSION MATRIX:")
        print("-" * 40)
        print("                 Predicted")
        print("              Fake    Real")
        print(f"Actual  Fake   {cm_elements['TP']:4d}    {cm_elements['FN']:4d}")
        print(f"        Real   {cm_elements['FP']:4d}    {cm_elements['TN']:4d}")
        print()
        
        # Print Confusion Matrix Elements
        print("CONFUSION MATRIX ELEMENTS:")
        print("-" * 40)
        print(f"True Positive (TP)  : {cm_elements['TP']:4d} (Fake correctly predicted as Fake)")
        print(f"True Negative (TN)  : {cm_elements['TN']:4d} (Real correctly predicted as Real)")
        print(f"False Positive (FP) : {cm_elements['FP']:4d} (Real incorrectly predicted as Fake)")
        print(f"False Negative (FN) : {cm_elements['FN']:4d} (Fake incorrectly predicted as Real)")
        print()
        
        # Print Manual Metrics (sesuai rumus dokumen)
        print("EVALUATION METRICS (Manual Calculation):")
        print("-" * 40)
        print(f"Accuracy  = (TP + TN) / (TP + TN + FP + FN)")
        print(f"          = ({cm_elements['TP']} + {cm_elements['TN']}) / ({cm_elements['TP']} + {cm_elements['TN']} + {cm_elements['FP']} + {cm_elements['FN']})")
        print(f"          = {manual_metrics['accuracy']:.4f} ({manual_metrics['accuracy']*100:.2f}%)")
        print()
        
        print(f"Precision = TP / (TP + FP)")
        print(f"          = {cm_elements['TP']} / ({cm_elements['TP']} + {cm_elements['FP']})")
        print(f"          = {manual_metrics['precision']:.4f} ({manual_metrics['precision']*100:.2f}%)")
        print()
        
        print(f"Recall    = TP / (TP + FN)")
        print(f"          = {cm_elements['TP']} / ({cm_elements['TP']} + {cm_elements['FN']})")
        print(f"          = {manual_metrics['recall']:.4f} ({manual_metrics['recall']*100:.2f}%)")
        print()
        
        print(f"F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)")
        print(f"          = 2 * ({manual_metrics['precision']:.4f} * {manual_metrics['recall']:.4f}) / ({manual_metrics['precision']:.4f} + {manual_metrics['recall']:.4f})")
        print(f"          = {manual_metrics['f1_score']:.4f} ({manual_metrics['f1_score']*100:.2f}%)")
        print()
        
        # Verifikasi dengan sklearn
        print("VERIFICATION (Sklearn):")
        print("-" * 40)
        print(f"Accuracy : {sklearn_metrics['accuracy']:.4f}")
        print(f"Precision: {sklearn_metrics['precision']:.4f}")
        print(f"Recall   : {sklearn_metrics['recall']:.4f}")
        print(f"F1-Score : {sklearn_metrics['f1_score']:.4f}")
        print()
        
        # Performance interpretation
        self.interpret_performance(manual_metrics)
        
        return metrics
    
    def interpret_performance(self, metrics):
        """
        Interpretasi performa model
        """
        print("PERFORMANCE INTERPRETATION:")
        print("-" * 40)
        
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']
        
        # Interpretasi accuracy
        if accuracy >= 0.95:
            acc_level = "Excellent"
        elif accuracy >= 0.90:
            acc_level = "Very Good"
        elif accuracy >= 0.80:
            acc_level = "Good"
        elif accuracy >= 0.70:
            acc_level = "Fair"
        else:
            acc_level = "Poor"
        
        print(f"Overall Accuracy: {acc_level} ({accuracy*100:.2f}%)")
        
        # Interpretasi precision
        print(f"Precision: {precision*100:.2f}% of predicted fake news are actually fake")
        
        # Interpretasi recall  
        print(f"Recall: {recall*100:.2f}% of actual fake news are correctly identified")
        
        # Interpretasi F1-score
        print(f"F1-Score: {f1*100:.2f}% (balanced measure of precision and recall)")
        
        # Rekomendasi
        if precision > recall:
            print("\nModel is more precise but may miss some fake news (low recall)")
        elif recall > precision:
            print("\nModel catches more fake news but has more false alarms (low precision)")
        else:
            print("\nModel has balanced precision and recall")
        
        print()
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", save_path=None):
        """
        Plot confusion matrix visualization
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('Actual Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def compare_with_research(self, metrics, research_results=None):
        """
        Bandingkan hasil dengan penelitian lain dari dokumen
        """
        if research_results is None:
            # Hasil dari studi literatur dalam dokumen
            research_results = {
                "Sudhakar & Kaliyamurthie (2022)": {"accuracy": 0.9876, "method": "Logistic Regression"},
                "Hasma (2024)": {"accuracy": 0.9388, "method": "Logistic Regression"},
                "Fahmuddin et al. (2023)": {"accuracy": 0.783, "method": "Logistic Regression"},
                "Wahid et al. (2024)": {"accuracy": 0.952, "method": "Logistic Regression (tuned)"},
                "Singh et al. (2021)": {"accuracy": 0.94, "method": "Logistic Regression"}
            }
        
        current_accuracy = metrics['manual_metrics']['accuracy']
        
        print("COMPARISON WITH PREVIOUS RESEARCH:")
        print("-" * 50)
        print(f"{'Study':<35} {'Accuracy':<10} {'Method'}")
        print("-" * 50)
        
        for study, result in research_results.items():
            print(f"{study:<35} {result['accuracy']:<10.4f} {result['method']}")
        
        print("-" * 50)
        print(f"{'Current Implementation':<35} {current_accuracy:<10.4f} {'Logistic Regression'}")
        print()
        
        # Ranking
        accuracies = [(study, result['accuracy']) for study, result in research_results.items()]
        accuracies.append(("Current Implementation", current_accuracy))
        accuracies.sort(key=lambda x: x[1], reverse=True)
        
        print("RANKING BY ACCURACY:")
        print("-" * 30)
        for i, (study, acc) in enumerate(accuracies, 1):
            marker = " <-- Current" if study == "Current Implementation" else ""
            print(f"{i}. {study}: {acc:.4f}{marker}")
        
        print()

# Demo evaluation
if __name__ == "__main__":
    # Sample predictions sesuai contoh dari dokumen
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # 1=Fake, 0=Real
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])  # Prediksi model
    
    print("=== PATTERN EVALUATION DEMONSTRATION ===")
    print("Menggunakan sample data untuk demonstrasi\n")
    
    # Inisialisasi evaluator
    evaluator = ModelEvaluator()
    
    # Generate evaluation report
    metrics = evaluator.print_evaluation_report(y_true, y_pred)
    
    # Compare with research
    evaluator.compare_with_research(metrics)