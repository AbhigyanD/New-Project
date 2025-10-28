"""
Credit Risk Analysis using Machine Learning
This script implements various ML models for credit risk prediction using financial data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CreditRiskAnalyzer:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load and preprocess the financial fraud dataset."""
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("amitkedia/Financial-Fraud-Dataset")
        self.data = dataset['train'].to_pandas()
        
        print("\nDataset shape:", self.data.shape)
        print("\nFirst few rows of the dataset:")
        print(self.data.head())
        
        # Basic data info
        print("\nDataset information:")
        print(self.data.info())
        
        # Check for missing values
        print("\nMissing values per column:")
        print(self.data.isnull().sum())
    
    def preprocess_data(self):
        """Preprocess the data for modeling."""
        print("\nPreprocessing data...")
        
        # Drop any duplicate rows
        self.data = self.data.drop_duplicates()
        
        # Handle missing values if any
        # For simplicity, we'll fill numerical columns with median and categorical with mode
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            else:
                self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Convert categorical variables to numerical
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.le_dict = {}
        
        for col in self.categorical_cols:
            self.le_dict[col] = LabelEncoder()
            self.data[col] = self.le_dict[col].fit_transform(self.data[col].astype(str))
        
        # Separate features and target
        self.X = self.data.drop('isFraud', axis=1)
        self.y = self.data['isFraud']
        
        # Handle class imbalance using SMOTE
        print("\nClass distribution before SMOTE:")
        print(self.y.value_counts(normalize=True))
        
        smote = SMOTE(random_state=42)
        self.X_resampled, self.y_resampled = smote.fit_resample(self.X, self.y)
        
        print("\nClass distribution after SMOTE:")
        print(pd.Series(self.y_resampled).value_counts(normalize=True))
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_resampled, self.y_resampled, test_size=0.2, random_state=42, stratify=self.y_resampled
        )
        
        # Scale numerical features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
    
    def train_models(self):
        """Train multiple ML models for comparison."""
        print("\nTraining models...")
        
        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0, thread_count=-1)
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            # Print metrics
            print(f"\n{name} Performance:")
            for metric, value in self.results[name].items():
                print(f"{metric}: {value:.4f}")
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        print("\nPlotting feature importance...")
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Get feature importance
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot
                plt.figure(figsize=(12, 6))
                plt.title(f"{name} - Feature Importance")
                plt.bar(range(self.X_train.shape[1]), importances[indices][:20])
                plt.xticks(range(self.X_train.shape[1]), 
                          [self.X.columns[i] for i in indices[:20]], 
                          rotation=90)
                plt.tight_layout()
                plt.savefig(f"{name.lower().replace(' ', '_')}_feature_importance.png")
                plt.close()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        print("\nPlotting ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.savefig('roc_curves.png')
        plt.close()
    
    def analyze_results(self):
        """Analyze and compare model results."""
        print("\nAnalyzing results...")
        
        # Convert results to DataFrame for better visualization
        results_df = pd.DataFrame(self.results).T
        print("\nModel Comparison:")
        print(results_df)
        
        # Save results to CSV
        results_df.to_csv('model_results.csv')
        print("\nResults saved to 'model_results.csv'")
        
        # Plot performance metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        plt.figure(figsize=(12, 8))
        for metric in metrics:
            plt.plot(results_df.index, results_df[metric], 'o-', label=metric)
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()

    def run_analysis(self):
        """Run the complete credit risk analysis pipeline."""
        print("Starting Credit Risk Analysis...")
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Run the analysis pipeline
        self.load_data()
        self.preprocess_data()
        self.train_models()
        self.plot_feature_importance()
        self.plot_roc_curves()
        self.analyze_results()
        
        print("\nCredit Risk Analysis completed successfully!")

if __name__ == "__main__":
    analyzer = CreditRiskAnalyzer()
    analyzer.run_analysis()
