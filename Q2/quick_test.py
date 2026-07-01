#!/usr/bin/env python3
"""
Simple test to see if packages are available
"""
import sys

try:
    import numpy
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import numpy: {e}")
    sys.exit(1)

try:
    import pandas
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pandas: {e}")
    sys.exit(1)

try:
    import sklearn
    print("✓ scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ Failed to import scikit-learn: {e}")
    sys.exit(1)

try:
    import nltk
    print("✓ nltk imported successfully")
except ImportError as e:
    print(f"✗ Failed to import nltk: {e}")
    sys.exit(1)

print("\nAll packages available! Now running sentiment classifier...\n")

from sentiment_classifier import SentimentClassifier, create_sample_dataset
from sklearn.model_selection import train_test_split

# Create sample dataset
texts, labels = create_sample_dataset()
print(f"✓ Loaded {len(texts)} text samples with {len(set(labels))} classes")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"✓ Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

# Train model
print("\n" + "="*60)
print("Training Logistic Regression Model...")
print("="*60)
lr_classifier = SentimentClassifier(model_type='logistic_regression')
lr_classifier.train(X_train, y_train)

# Evaluate
lr_metrics = lr_classifier.evaluate(X_test, y_test)
print(f"\nAccuracy: {lr_metrics['accuracy']:.4f}")
print(f"F1-Score (Weighted): {lr_metrics['f1_weighted']:.4f}")
print(f"F1-Score (Macro): {lr_metrics['f1_macro']:.4f}")
print(f"F1-Score (Micro): {lr_metrics['f1_micro']:.4f}")

# Test predictions
print("\n" + "="*60)
print("Test Predictions")
print("="*60)
test_samples = [
    "This is absolutely wonderful!",
    "I hate this terrible product",
    "Product works as described",
]

for sample in test_samples:
    pred = lr_classifier.predict([sample])[0]
    prob = lr_classifier.predict_proba([sample])[0]
    print(f"\nText: '{sample}'")
    print(f"Prediction: {pred} (Confidence: {max(prob):.4f})")

print("\n✓ SUCCESS: Sentiment Classifier is working correctly!")
