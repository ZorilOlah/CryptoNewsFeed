import torch
from typing import List
import numpy as np
import json
from evaluation.metrics import multi_class_accuracy

class Results:
    def __init__(self,
                 output : dict) -> None:
        self.output = output
        self.results = {key : {} for key in self.output.keys()}
        self.total_preds = np.array([])
        self.total_labels = np.array([])   

        self._calculate_batch_accuracies_and_total_preds_labels()

        self.total_acc = multi_class_accuracy(self.total_preds, self.total_labels)
    
    def _calculate_batch_accuracies_and_total_preds_labels(self) -> None:
        for epoch in self.output:
            epoch_preds, epoch_labels = np.array([]), np.array([])
            for batch_number, batch in enumerate(self.output[epoch]):
                batch_logits = self.output[epoch][batch]['outputs']['logits']
                batch_preds = torch.argmax(batch_logits, dim = -1).detach().numpy()
                batch_labels = self.output[epoch][batch]['labels'].detach().numpy()
               
                self._update_total_preds_and_labels(batch_preds = batch_preds, batch_labels = batch_labels)

                batch_accuracies = multi_class_accuracy(preds = batch_preds, labels = batch_labels)

                self.results[epoch][batch] = {'mc_acc' : batch_accuracies}
              
                epoch_preds = np.append(epoch_preds, batch_preds)
                epoch_labels = np.append(epoch_labels, batch_labels)
            self.results[epoch]['total'] = multi_class_accuracy(preds = epoch_preds, labels = epoch_labels)
            
    def _update_total_preds_and_labels(self, 
                                       batch_preds : np.ndarray,
                                       batch_labels : np.ndarray) -> None:
        self.total_preds = np.append(self.total_preds , batch_preds)
        self.total_labels = np.append(self.total_labels , batch_labels)
        
    def save_results(self,
                     save_directory : str) -> None:
        with open(save_directory + '.json', "w") as outfile:
            json.dump(self.results, outfile)
