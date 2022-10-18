import numpy as np
import torch
from evaluation.metrics import multi_class_accuracy

def calculate_epoch_accuracies(output, epoch) -> dict:
    epoch_preds, epoch_labels = np.array([]), np.array([])
    for batch_number, batch in enumerate(output[f'epoch_{epoch}']):
            batch_logits = output[f'epoch_{epoch}'][batch]['outputs']['logits']
            batch_preds = torch.argmax(batch_logits, dim = -1).detach().numpy()
            batch_labels = output[f'epoch_{epoch}'][batch]['labels'].detach().numpy()
            
            epoch_preds = np.append(epoch_preds, batch_preds)
            epoch_labels = np.append(epoch_labels, batch_labels)
    acc = multi_class_accuracy(preds = epoch_preds, labels = epoch_labels)
    return(acc)


