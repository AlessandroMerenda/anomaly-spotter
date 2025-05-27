import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

def evaluate_threshold(y_true, y_scores, threshold):
    """
    Applica una soglia ai punteggi e calcola precision, recall, F1 e confusion matrix.
    """
    y_pred = (y_scores > threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

def find_best_threshold_recall_1(y_true, y_scores, n_steps=200):
    """
    Trova la soglia più alta possibile che garantisce recall = 1.0.
    Restituisce None se non è mai raggiunto.
    """
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_steps)
    best = None

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if recall == 1.0:
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            best = {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }

    return best

def find_best_threshold_min_recall(y_true, y_scores, min_recall=0.95, n_steps=200):
    """
    Trova la soglia che massimizza l'F1 score tra quelle con recall >= min_recall.
    Restituisce None se nessuna soglia soddisfa la condizione.
    """
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_steps)
    best = None
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if recall >= min_recall and f1 > best_f1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            best = {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }
            best_f1 = f1

    return best
