import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Global settings
SEED = 19
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Data loading

print("Loading AG News dataset...")
dataset = load_dataset("ag_news")

train_full = dataset["train"]
test_data = dataset["test"]

def combine_text(example):
    return example["text"]

X_train_full = [combine_text(x) for x in train_full]
y_train_full = [x["label"] for x in train_full]

X_test = [combine_text(x) for x in test_data]
y_test = [x["label"] for x in test_data]

# Data training
X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    stratify=y_train_full,
    random_state=SEED
)

print(f"Train size: {len(X_train)}")
print(f"Dev size: {len(X_dev)}")
print(f"Test size: {len(X_test)}")

# TF-IDF vectorizer, for preprocessing
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2,
    max_features=50000
)

vectorizer.fit(X_train)

X_train_vec = vectorizer.transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)
X_test_vec = vectorizer.transform(X_test)

# Model training and tuning
def evaluate_model(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    return acc, macro_f1

def tune_logistic_regression():
    print("\nTuning Logistic Regression...")
    best_score = 0
    best_model = None
    best_c = None

    for c in [0.1, 1, 5, 10]:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            random_state=SEED
        )
        model.fit(X_train_vec, y_train)
        acc, macro_f1 = evaluate_model(model, X_dev_vec, y_dev)

        print(f"C={c} | Dev Accuracy={acc:.4f} | Dev Macro-F1={macro_f1:.4f}")

        if macro_f1 > best_score:
            best_score = macro_f1
            best_model = model
            best_c = c

    print(f"Best C for Logistic Regression: {best_c}")
    return best_c

def tune_svm():
    print("\nTuning Linear SVM...")
    best_score = 0
    best_model = None
    best_c = None

    for c in [0.1, 1, 5, 10]:
        model = LinearSVC(
            C=c,
            random_state=SEED
        )
        model.fit(X_train_vec, y_train)
        acc, macro_f1 = evaluate_model(model, X_dev_vec, y_dev)

        print(f"C={c} | Dev Accuracy={acc:.4f} | Dev Macro-F1={macro_f1:.4f}")

        if macro_f1 > best_score:
            best_score = macro_f1
            best_model = model
            best_c = c

    print(f"Best C for Linear SVM: {best_c}")
    return best_c


best_c_lr = tune_logistic_regression()
best_c_svm = tune_svm()

# Retrain on Train+dev
print("\nRetraining best models on full training data...")

X_train_full_vec = vectorizer.fit_transform(X_train_full)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
final_lr = LogisticRegression(
    C=best_c_lr,
    max_iter=1000,
    random_state=SEED
)
final_lr.fit(X_train_full_vec, y_train_full)

# Linear SVM
final_svm = LinearSVC(
    C=best_c_svm,
    random_state=SEED
)
final_svm.fit(X_train_full_vec, y_train_full)

# Final test evaluation
def final_evaluation(model, model_name):
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"\n{model_name} TEST RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_confusion_matrix.png")
    plt.close()

    # Save misclassified examples
    misclassified = []
    for text, true, pred in zip(X_test, y_test, preds):
        if true != pred:
            misclassified.append({
                "text": text,
                "true_label": CLASS_NAMES[true],
                "predicted_label": CLASS_NAMES[pred]
            })

    df_mis = pd.DataFrame(misclassified)
    df_mis.head(20).to_csv(
        f"{RESULTS_DIR}/{model_name}_misclassified_top20.csv",
        index=False
    )

    print(f"Saved confusion matrix and misclassified examples for {model_name}")

    return acc, macro_f1


lr_acc, lr_f1 = final_evaluation(final_lr, "LogisticRegression")
svm_acc, svm_f1 = final_evaluation(final_svm, "LinearSVM")

# Summary
print("Final test results summary")
print(f"Logistic Regression  | Accuracy={lr_acc:.4f} | Macro-F1={lr_f1:.4f}")
print(f"Linear SVM           | Accuracy={svm_acc:.4f} | Macro-F1={svm_f1:.4f}")
