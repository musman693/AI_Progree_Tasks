"""
Intelligent Multi-Class Natural Language Text Sentiment Classifier
This module implements an NLP data classifier for mapping and grouping unstructured text statements.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """
    Handles text preprocessing including:
    - Stop-word removal
    - Lemmatization
    - Text normalization
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean text by removing special characters and converting to lowercase
        """
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stop-words from text
        """
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def lemmatize_text(self, text):
        """
        Apply lemmatization to text
        """
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized_tokens)
    
    def preprocess(self, text):
        """
        Apply full preprocessing pipeline
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text


class SentimentClassifier:
    """
    Multi-class sentiment classifier using TF-IDF and machine learning models
    """
    
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize classifier
        Args:
            model_type: 'logistic_regression' or 'random_forest'
        """
        self.preprocessor = TextPreprocessor()
        self.model_type = model_type
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        if model_type == 'logistic_regression':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be 'logistic_regression' or 'random_forest'")
        
        self.pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('classifier', self.classifier)
        ])
        
        self.classes_ = None
        self.is_trained = False
    
    def preprocess_texts(self, texts):
        """
        Preprocess multiple texts
        """
        return [self.preprocessor.preprocess(text) for text in texts]
    
    def train(self, X_train, y_train):
        """
        Train the classifier
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        X_train_processed = self.preprocess_texts(X_train)
        self.pipeline.fit(X_train_processed, y_train)
        self.classes_ = self.pipeline.named_steps['classifier'].classes_
        self.is_trained = True
        print(f"[OK] Model trained successfully on {len(X_train)} samples")
        print(f"  Classes: {self.classes_}")
    
    def predict(self, texts):
        """
        Predict sentiment for texts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        texts_processed = self.preprocess_texts(texts)
        return self.pipeline.predict(texts_processed)
    
    def predict_proba(self, texts):
        """
        Get prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        texts_processed = self.preprocess_texts(texts)
        return self.pipeline.predict_proba(texts_processed)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        Returns dictionary with various metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred
        }
        
        return metrics


def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    Returns tuple of (texts, labels)
    """
    texts = [
        # Positive sentiments
        "This product is absolutely amazing and I love it!",
        "Excellent service, very satisfied with my purchase",
        "Great quality and fast delivery, highly recommend",
        "Fantastic experience, will definitely buy again",
        "Outstanding customer support, very helpful",
        
        # Negative sentiments
        "Terrible quality, waste of money",
        "Very disappointed with this product, poor quality",
        "Bad experience, customer service was rude",
        "Awful quality, broke after one day",
        "Horrible product, would not recommend",
        
        # Neutral sentiments
        "It is a product that exists",
        "The item was delivered on time",
        "This is a standard product",
        "Average quality, nothing special",
        "Product arrived in good condition",
        
        # Positive sentiments (more examples)
        "I am very happy with my purchase",
        "Perfect product, exactly what I wanted",
        "Amazing value for money",
        "Love everything about this",
        "Best purchase ever made",
        
        # Negative sentiments (more examples)
        "Worst experience ever",
        "Not worth the price at all",
        "Complete disaster, totally unhappy",
        "Extremely disappointed",
        "Absolute garbage, stay away",
        
        # Neutral sentiments (more examples)
        "The package arrived yesterday",
        "Product specifications are listed correctly",
        "Item matches the description",
        "Standard shipping took two days",
        "Available in multiple colors"
    ]
    
    labels = [
        # Positive
        'positive', 'positive', 'positive', 'positive', 'positive',
        # Negative
        'negative', 'negative', 'negative', 'negative', 'negative',
        # Neutral
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        # Positive
        'positive', 'positive', 'positive', 'positive', 'positive',
        # Negative
        'negative', 'negative', 'negative', 'negative', 'negative',
        # Neutral
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
    ]
    
    return texts, labels


def main():
    """
    Main execution function demonstrating the sentiment classifier
    """
    print("=" * 80)
    print("INTELLIGENT MULTI-CLASS NLP SENTIMENT CLASSIFIER")
    print("=" * 80)
    print()
    
    # 1. Create sample dataset
    print("1. Loading sample dataset...")
    texts, labels = create_sample_dataset()
    print(f"   [OK] Loaded {len(texts)} text samples")
    print(f"   [OK] Classes: {set(labels)}")
    print()
    
    # 2. Split data
    print("2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   [OK] Training set: {len(X_train)} samples")
    print(f"   [OK] Testing set: {len(X_test)} samples")
    print()
    
    # 3. Train Logistic Regression model
    print("3. Training Logistic Regression model...")
    lr_classifier = SentimentClassifier(model_type='logistic_regression')
    lr_classifier.train(X_train, y_train)
    print()
    
    # 4. Evaluate Logistic Regression
    print("4. Evaluating Logistic Regression model...")
    lr_metrics = lr_classifier.evaluate(X_test, y_test)
    print(f"   [OK] Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"   [OK] F1-Score (Weighted): {lr_metrics['f1_weighted']:.4f}")
    print(f"   [OK] F1-Score (Macro): {lr_metrics['f1_macro']:.4f}")
    print(f"   [OK] F1-Score (Micro): {lr_metrics['f1_micro']:.4f}")
    print()
    print("   Classification Report (Logistic Regression):")
    print("   " + "\n   ".join(lr_metrics['classification_report'].split('\n')))
    print()
    
    # 5. Train Random Forest model
    print("5. Training Random Forest model...")
    rf_classifier = SentimentClassifier(model_type='random_forest')
    rf_classifier.train(X_train, y_train)
    print()
    
    # 6. Evaluate Random Forest
    print("6. Evaluating Random Forest model...")
    rf_metrics = rf_classifier.evaluate(X_test, y_test)
    print(f"   [OK] Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"   [OK] F1-Score (Weighted): {rf_metrics['f1_weighted']:.4f}")
    print(f"   [OK] F1-Score (Macro): {rf_metrics['f1_macro']:.4f}")
    print(f"   [OK] F1-Score (Micro): {rf_metrics['f1_micro']:.4f}")
    print()
    print("   Classification Report (Random Forest):")
    print("   " + "\n   ".join(rf_metrics['classification_report'].split('\n')))
    print()
    
    # 7. Test with new samples
    print("7. Testing with new text samples...")
    test_samples = [
        "This is absolutely wonderful and fantastic!",
        "I hate this, it is terrible",
        "The product works as described"
    ]
    
    for sample in test_samples:
        lr_pred = lr_classifier.predict([sample])[0]
        rf_pred = rf_classifier.predict([sample])[0]
        lr_proba = lr_classifier.predict_proba([sample])[0]
        
        print(f"\n   Text: '{sample}'")
        print(f"   [OK] Logistic Regression: {lr_pred}")
        print(f"   [OK] Random Forest: {rf_pred}")
        print(f"   [OK] Confidence (LR): {max(lr_proba):.4f}")
    
    print()
    print("=" * 80)
    print("SENTIMENT CLASSIFIER EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
