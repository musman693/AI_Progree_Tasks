#!/usr/bin/env python3
"""
Install packages and run sentiment classifier
"""
import subprocess
import sys
import os

def install_packages():
    """Install required packages"""
    packages = ["numpy", "pandas", "scikit-learn", "nltk"]
    
    print("Installing packages...")
    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package} already installed")
        except ImportError:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"  ✓ {package} installed")

if __name__ == "__main__":
    install_packages()
    
    # Now import and run the sentiment classifier
    print("\nRunning sentiment classifier...\n")
    
    from sentiment_classifier import SentimentClassifier, create_sample_dataset
    from sklearn.model_selection import train_test_split
    
    # Create sample dataset
    texts, labels = create_sample_dataset()
    print(f"✓ Sample dataset loaded: {len(texts)} samples")
    print(f"✓ Classes: {set(labels)}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"✓ Data split - Train: {len(X_train)}, Test: {len(X_test)}\n")
    
    # Train Logistic Regression
    print("=" * 80)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 80)
    lr_classifier = SentimentClassifier(model_type='logistic_regression')
    lr_classifier.train(X_train, y_train)
    print()
    
    # Evaluate LR
    print("Evaluating Logistic Regression model...")
    lr_metrics = lr_classifier.evaluate(X_test, y_test)
    print(f"✓ Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"✓ F1-Score (Weighted): {lr_metrics['f1_weighted']:.4f}")
    print(f"✓ F1-Score (Macro): {lr_metrics['f1_macro']:.4f}")
    print(f"✓ F1-Score (Micro): {lr_metrics['f1_micro']:.4f}\n")
    
    print("Classification Report (Logistic Regression):")
    print(lr_metrics['classification_report'])
    
    # Train Random Forest
    print("=" * 80)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 80)
    rf_classifier = SentimentClassifier(model_type='random_forest')
    rf_classifier.train(X_train, y_train)
    print()
    
    # Evaluate RF
    print("Evaluating Random Forest model...")
    rf_metrics = rf_classifier.evaluate(X_test, y_test)
    print(f"✓ Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"✓ F1-Score (Weighted): {rf_metrics['f1_weighted']:.4f}")
    print(f"✓ F1-Score (Macro): {rf_metrics['f1_macro']:.4f}")
    print(f"✓ F1-Score (Micro): {rf_metrics['f1_micro']:.4f}\n")
    
    print("Classification Report (Random Forest):")
    print(rf_metrics['classification_report'])
    
    # Test predictions
    print("=" * 80)
    print("TEST PREDICTIONS WITH NEW SAMPLES")
    print("=" * 80 + "\n")
    
    test_samples = [
        "This is absolutely wonderful and fantastic!",
        "I hate this, it is terrible and awful",
        "The product works as described",
        "Amazing quality and excellent service",
        "Worst purchase I have ever made"
    ]
    
    for sample in test_samples:
        lr_pred = lr_classifier.predict([sample])[0]
        rf_pred = rf_classifier.predict([sample])[0]
        lr_proba = lr_classifier.predict_proba([sample])[0]
        
        print(f"Text: '{sample}'")
        print(f"  LR Prediction: {lr_pred} (Confidence: {max(lr_proba):.4f})")
        print(f"  RF Prediction: {rf_pred}\n")
    
    print("=" * 80)
    print("✓ SENTIMENT CLASSIFIER EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
