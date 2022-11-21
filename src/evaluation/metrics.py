import warnings
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def multi_class_accuracy(preds: np.ndarray, 
                        labels: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc_acc = {key : None for key in ['Negative','Neutral','Positive']}
        
        positive_mask = labels == 2
        neutral_mask = labels == 1
        negative_mask = labels == 0
        
        pos_labels = labels[positive_mask]
        neut_labels = labels[neutral_mask]
        neg_labels = labels[negative_mask]
        
        pos_preds = preds[positive_mask]
        neut_preds = preds[neutral_mask]
        neg_preds = preds[negative_mask]
        
        mc_acc['Positive'] = accuracy_score(pos_labels, pos_preds)
        mc_acc['Neutral'] = accuracy_score(neut_labels, neut_preds)
        mc_acc['Negative'] = accuracy_score(neg_labels, neg_preds)

    return mc_acc
    
def multi_class_acc_via_cm(preds : np.ndarray,
                        labels : np.ndarray) -> dict:
    mc_acc = {key : None for key in ['Negative','Neutral','Positive']}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = confusion_matrix(labels, preds)
        accuracies = matrix.diagonal()/matrix.sum(axis=1)

    mc_acc['Positive'] = accuracies[2]
    mc_acc['Neutral'] = accuracies[1]
    mc_acc['Negative'] = accuracies[0]
    return mc_acc

def multi_class_f1_score(preds : np.ndarray,
             labels : np.ndarray) -> float:
    F1 = f1_score(y_pred = preds, y_true = labels, average = "weighted")
    return F1
