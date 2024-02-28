import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    Gather metrics for multi-label classification:
        - f1-micro
        - accuracy
        - ROC AUC
    """
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
            'roc_auc': roc_auc,
            'accuracy': accuracy}
    return metrics

def single_label_metrics_cls(predictions, labels):
    """
    Gather metrics for single-label classification:
        - f1-micro
        - accuracy
        - f1-macro
        - precision
        - recall
        - confusion matrix
        - classification report
    """
    y_pred = np.argmax(predictions, axis=1)
    y_true = labels
    classification_results = classification_report(y_true, y_pred, output_dict=True)
    confusion_results = confusion_matrix(y_true, y_pred).tolist()
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'f1-micro': f1_micro_average,
        'f1': f1_macro_average,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'classification_report': classification_results,
        'confusion_matrix': confusion_results
    }

def single_label_metrics_reg(predictions, labels):
    """
    Gather metrics for predictions with single-label regression, by applying tanh and rounding to
    nearest integer:
        - f1-micro
        - accuracy
    """
    tanh = torch.nn.Tanh()
    regs = tanh(torch.Tensor(predictions))
    y_pred = np.round(regs)
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    return {'f1': f1_micro_average,
               'accuracy': accuracy}