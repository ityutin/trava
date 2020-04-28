from sklearn.metrics import auc

from trava.metric import Metric


def plot_roc_curves(metric: Metric, fig, ax, color: str, label: str):
    ax.set_title(label + ' ' + 'ROC')

    fpr, tpr, threshold = metric.value
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label='{} AUC = {:0.2f}'.format(metric.model_id, roc_auc), color=color)
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
