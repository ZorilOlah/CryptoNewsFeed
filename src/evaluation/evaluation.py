import torch
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix 
import json
import warnings

class Results:
    def __init__(self,
                 output : dict) -> None:
        self.output = output
        self.results = {key : {} for key in self.output.keys()}
        self.total_preds = np.array([])
        self.total_labels = np.array([])   

        self._calculate_batch_accuracies_and_total_preds_labels()

        self.total_acc = self.multi_class_accuracy(self.total_preds, self.total_labels)

    def multi_class_accuracy(self, preds: np.array, 
                            labels: np.array) -> dict:
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
    
    def multi_class_acc_via_cm(self, preds : np.array,
                               labels : np.array) -> dict:
        mc_acc = {key : None for key in ['Negative','Neutral','Positive']}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = confusion_matrix(labels, preds)
            accuracies = matrix.diagonal()/matrix.sum(axis=1)
        
        mc_acc['Positive'] = accuracies[2]
        mc_acc['Neutral'] = accuracies[1]
        mc_acc['Negative'] = accuracies[0]
        return mc_acc
        
    def _calculate_batch_accuracies_and_total_preds_labels(self) -> None:
        for epoch in self.output:
            epoch_preds, epoch_labels = np.array([]), np.array([])
            for batch_number, batch in enumerate(self.output[epoch]):
                batch_logits = self.output[epoch][batch]['outputs']['logits']
                batch_preds = torch.argmax(batch_logits, dim = -1).detach().numpy()
                batch_labels = self.output[epoch][batch]['labels'].detach().numpy()
               
                self._update_total_preds_and_labels(batch_preds = batch_preds, batch_labels = batch_labels)

                batch_accuracies = self.multi_class_accuracy(preds = batch_preds, labels = batch_labels)

                self.results[epoch][batch] = {'mc_acc' : batch_accuracies}
              
                epoch_preds = np.append(epoch_preds, batch_preds)
                epoch_labels = np.append(epoch_labels, batch_labels)
            self.results[epoch]['total'] = self.multi_class_accuracy(preds = epoch_preds, labels = epoch_labels)
            
    def _update_total_preds_and_labels(self, 
                                       batch_preds : np.array,
                                       batch_labels : np.array) -> None:
        self.total_preds = np.append(self.total_preds , batch_preds)
        self.total_labels = np.append(self.total_labels , batch_labels)
        
    def save_results(self,
                     save_directory : str) -> None:
        with open(save_directory + '.json', "w") as outfile:
            json.dump(self.results, outfile)
