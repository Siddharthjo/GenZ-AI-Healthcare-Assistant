"""
Evaluation utilities: classification report, confusion matrix, feature importance.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def print_report(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return acc


def plot_confusion_matrix(model, X_test, y_test, le, figsize=(22, 18)):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = le.classes_

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix — Test Set', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, mlb, top_n=30):
    if not hasattr(model, 'feature_importances_'):
        print(f"{type(model).__name__} does not expose feature_importances_.")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    features = [mlb.classes_[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(features[::-1], importances[indices][::-1], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Most Predictive Symptoms')
    plt.tight_layout()
    return fig


def plot_cv_comparison(results):
    names = list(results.keys())
    means = [results[n]['cv_mean'] for n in names]
    stds = [results[n]['cv_std'] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6, color='steelblue', alpha=0.8)
    ax.set_ylabel('CV Accuracy (5-fold)')
    ax.set_title('Model Comparison — Cross-Validation Accuracy')
    ax.set_ylim(0, 1.05)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    return fig
