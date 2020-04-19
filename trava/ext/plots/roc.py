import random
from typing import List

import matplotlib.pyplot as plt
from sklearn.metrics import auc

from trava.metric import Metric


def plot_roc_curves(metrics: List[Metric], label: str):
    def color_for(idx):
        r = lambda: random.randint(0, 255)
        base_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        if idx > len(base_colors) - 1:
            return '#%02X%02X%02X' % (r(), r(), r())

        return base_colors[idx]

    plt.figure(figsize=(10, 10))
    plt.title(label + ' ' + 'ROC')
    for idx, metric in enumerate(metrics):
        fpr, tpr, threshold = metric.value
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='{} AUC = {:0.2f}'.format(metric.model_id, roc_auc), color=color_for(idx))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
