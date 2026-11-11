"""
train_model.py
==============
Script to generate the student dataset, train a CART (Classification and Regression Tree)
Decision Tree Classifier, evaluate its performance, generate visual plots,
and save the trained model using Joblib.

Machine Learning Topic: CART Algorithm (Unit 5 - Combining Models)
Criterion: Gini Impurity
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating plots without a GUI window
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'metrics.json')
STATIC_IMG_DIR = os.path.join(BASE_DIR, 'static', 'images')

os.makedirs(STATIC_IMG_DIR, exist_ok=True)

FEATURE_NAMES = ['Attendance (%)', 'Study Hours / Day', 'Internal Marks', 'Assignment Score']
FEATURE_KEYS = ['attendance', 'study_hours', 'internal_marks', 'assignment_score']


def generate_dataset(n_samples=250, random_seed=42):
    """
    Generates a realistic synthetic student academic dataset.
    Features:
    1. attendance: 40% to 100%
    2. study_hours: 0.5 to 12.0 hours per day
    3. internal_marks: 20 to 100
    4. assignment_score: 25 to 100
    Target:
    Pass (1) or Fail (0) based on academic thresholds with realistic variance.
    """
    np.random.seed(random_seed)

    # Realistic distributions
    attendance = np.random.uniform(42, 98, n_samples)
    study_hours = np.random.uniform(0.5, 10.5, n_samples)
    internal_marks = np.random.uniform(25, 95, n_samples)
    assignment_score = np.random.uniform(30, 98, n_samples)

    # Academic Performance Index (weighted formula representing real academic evaluation)
    # Weights: Internal Marks (35%), Attendance (25%), Assignment Score (20%), Study Hours (20% normalized)
    normalized_study = (study_hours / 10.0) * 100.0
    composite_score = (
        0.35 * internal_marks +
        0.25 * attendance +
        0.20 * assignment_score +
        0.20 * normalized_study
    )

    # Add minor noise (+/- 4 marks) to simulate unpredictable exam day factors
    noise = np.random.normal(0, 3.5, n_samples)
    final_score = composite_score + noise

    # Passing standard: composite threshold of 50/100, with minimum attendance and internal requirement
    target = []
    for i in range(n_samples):
        # A student passes if composite score >= 50 and they don't have catastrophic internal failure (< 30)
        if final_score[i] >= 50.0 and internal_marks[i] >= 28.0 and attendance[i] >= 45.0:
            target.append('Pass')
        else:
            target.append('Fail')

    df = pd.DataFrame({
        'attendance': np.round(attendance, 1),
        'study_hours': np.round(study_hours, 1),
        'internal_marks': np.round(internal_marks, 1),
        'assignment_score': np.round(assignment_score, 1),
        'result': target
    })

    df.to_csv(DATASET_PATH, index=False)
    print(f"[OK] Generated {n_samples} student records and saved to '{DATASET_PATH}'.")
    return df


def load_or_create_dataset():
    """Loads existing dataset or creates a new one if missing."""
    if os.path.exists(DATASET_PATH):
        print(f"[INFO] Loading existing dataset from '{DATASET_PATH}'...")
        df = pd.read_csv(DATASET_PATH)
    else:
        print("[INFO] Dataset not found. Generating fresh dataset...")
        df = generate_dataset()
    return df


def plot_decision_tree(model):
    """Generates and saves a high-resolution decision tree visualization."""
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    
    # Custom color palette for classes
    plot_tree(
        model,
        feature_names=FEATURE_NAMES,
        class_names=['Fail', 'Pass'],
        filled=True,
        rounded=True,
        impurity=True,   # Displays Gini impurity at every node
        proportion=True, # Displays proportion of samples
        fontsize=10,
        ax=ax
    )
    
    ax.set_title("CART Decision Tree Visualization (Gini Impurity Criterion)\nStudent Performance Classification", 
                 fontsize=15, fontweight='bold', pad=15, color='#0f172a')
    plt.tight_layout()
    output_path = os.path.join(STATIC_IMG_DIR, 'decision_tree.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Decision tree plot saved to '{output_path}'.")


def plot_confusion_matrix(cm):
    """Generates and saves a clean, styled Confusion Matrix visualization."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Custom navy/purple styling for the heatmap
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
    
    # Add colorbar
    fig.colorbar(cax, shrink=0.8)

    labels = ['Fail', 'Pass']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color='#1e293b')
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold', color='#1e293b')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10, color='#0f172a')
    ax.set_ylabel('Actual True Label', fontsize=12, fontweight='bold', labelpad=10, color='#0f172a')
    ax.set_title('Confusion Matrix (Test Set)', fontsize=13, fontweight='bold', pad=20, color='#0f172a')

    # Annotate numbers with counts and percentage
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total) * 100
            text_color = "white" if count > (total * 0.3) else "#0f172a"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                    color=text_color, fontsize=13, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(STATIC_IMG_DIR, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix plot saved to '{output_path}'.")


def plot_feature_importance(model):
    """Generates and saves an informative horizontal bar chart of feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    colors = ['#818cf8', '#6366f1', '#4f46e5', '#312e81']
    
    y_pos = np.arange(len(importances))
    bars = ax.barh(y_pos, importances[indices] * 100, color=[colors[i] for i in indices], edgecolor='#1e1b4b', height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in indices], fontsize=11, fontweight='bold', color='#1e293b')
    ax.set_xlabel('Relative Importance (%)', fontsize=11, fontweight='bold', color='#0f172a', labelpad=8)
    ax.set_title('CART Feature Importance (Gini Index Contribution)', fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_xlim(0, max(importances * 100) * 1.25)

    # Annotate values on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.0, bar.get_y() + bar.get_height()/2.0, f"{width:.1f}%", 
                va='center', ha='left', fontsize=10, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    output_path = os.path.join(STATIC_IMG_DIR, 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Feature importance plot saved to '{output_path}'.")


def train_and_evaluate():
    """Main function to train CART Decision Tree, evaluate, and save artifacts."""
    df = load_or_create_dataset()

    X = df[FEATURE_KEYS]
    y = df['result'].apply(lambda x: 1 if x == 'Pass' else 0)

    # 80% Training, 20% Testing split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[INFO] Training set size: {len(X_train)} samples")
    print(f"[INFO] Testing set size:  {len(X_test)} samples")

    # CART uses criterion='gini' (Gini Impurity)
    # Using max_depth=3 or 4 prevents overfitting and makes the decision tree cleanly interpretable
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, target_names=['Fail', 'Pass'], output_dict=True)

    print("\n" + "="*50)
    print("MODEL PERFORMANCE REPORT (CART / GINI IMPURITY)")
    print("="*50)
    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Fail', 'Pass']))
    print("="*50 + "\n")

    # Save model using Joblib
    joblib.dump(model, MODEL_PATH)
    print(f"[OK] Model successfully saved to '{MODEL_PATH}'.")

    # Save metrics metadata for the web application
    metrics = {
        'train_accuracy': round(float(train_acc * 100), 2),
        'test_accuracy': round(float(test_acc * 100), 2),
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'pass_count': int((df['result'] == 'Pass').sum()),
        'fail_count': int((df['result'] == 'Fail').sum()),
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        },
        'classification_report': {
            'fail': {
                'precision': round(report['Fail']['precision'] * 100, 1),
                'recall': round(report['Fail']['recall'] * 100, 1),
                'f1': round(report['Fail']['f1-score'] * 100, 1),
                'support': int(report['Fail']['support'])
            },
            'pass': {
                'precision': round(report['Pass']['precision'] * 100, 1),
                'recall': round(report['Pass']['recall'] * 100, 1),
                'f1': round(report['Pass']['f1-score'] * 100, 1),
                'support': int(report['Pass']['support'])
            }
        },
        'feature_importances': {
            name: round(float(val * 100), 2)
            for name, val in zip(FEATURE_NAMES, model.feature_importances_)
        }
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Metrics saved to '{METRICS_PATH}'.")

    # Generate and save all plots
    plot_decision_tree(model)
    plot_confusion_matrix(cm)
    plot_feature_importance(model)

    return model, metrics


if __name__ == '__main__':
    train_and_evaluate()
