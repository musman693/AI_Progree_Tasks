#!/usr/bin/env python3
"""
Test script for sentiment classifier - outputs results to file
"""

from sentiment_classifier import SentimentClassifier, create_sample_dataset
from sklearn.model_selection import train_test_split

with open('test_results.txt', 'w') as f:
    # Redirect print to file
    import sys
    
    # Create sample dataset
    texts, labels = create_sample_dataset()
    f.write(f"[OK] Sample dataset loaded: {len(texts)} samples\n")
    f.write(f"[OK] Classes: {set(labels)}\n\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    f.write(f"[OK] Data split - Train: {len(X_train)}, Test: {len(X_test)}\n\n")
    
    # Train Logistic Regression
    f.write("=" * 80 + "\n")
    f.write("TRAINING LOGISTIC REGRESSION MODEL\n")
    f.write("=" * 80 + "\n\n")
    
    lr_classifier = SentimentClassifier(model_type='logistic_regression')
    lr_classifier.train(X_train, y_train)
    
    # Evaluate LR
    lr_metrics = lr_classifier.evaluate(X_test, y_test)
    f.write(f"[OK] Accuracy: {lr_metrics['accuracy']:.4f}\n")
    f.write(f"[OK] F1-Score (Weighted): {lr_metrics['f1_weighted']:.4f}\n")
    f.write(f"[OK] F1-Score (Macro): {lr_metrics['f1_macro']:.4f}\n")
    f.write(f"[OK] F1-Score (Micro): {lr_metrics['f1_micro']:.4f}\n\n")
    
    f.write("Classification Report:\n")
    f.write(lr_metrics['classification_report'] + "\n\n")
    
    # Train Random Forest
    f.write("=" * 80 + "\n")
    f.write("TRAINING RANDOM FOREST MODEL\n")
    f.write("=" * 80 + "\n\n")
    
    rf_classifier = SentimentClassifier(model_type='random_forest')
    rf_classifier.train(X_train, y_train)
    
    # Evaluate RF
    rf_metrics = rf_classifier.evaluate(X_test, y_test)
    f.write(f"[OK] Accuracy: {rf_metrics['accuracy']:.4f}\n")
    f.write(f"[OK] F1-Score (Weighted): {rf_metrics['f1_weighted']:.4f}\n")
    f.write(f"[OK] F1-Score (Macro): {rf_metrics['f1_macro']:.4f}\n")
    f.write(f"[OK] F1-Score (Micro): {rf_metrics['f1_micro']:.4f}\n\n")
    
    f.write("Classification Report:\n")
    f.write(rf_metrics['classification_report'] + "\n\n")
    
    # Test predictions
    f.write("=" * 80 + "\n")
    f.write("TEST PREDICTIONS\n")
    f.write("=" * 80 + "\n\n")
    
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
        
        f.write(f"Text: '{sample}'\n")
        f.write(f"  LR Prediction: {lr_pred} (Confidence: {max(lr_proba):.4f})\n")
        f.write(f"  RF Prediction: {rf_pred}\n\n")
    
    f.write("[OK] TEST COMPLETED SUCCESSFULLY\n")

print("Test results written to test_results.txt")
