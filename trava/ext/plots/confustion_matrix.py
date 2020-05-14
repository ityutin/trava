import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from trava.metric import Metric


def plot_confusion_matrix(metric: Metric, fig, ax, color: str, label: str):
    ax.set_title(label + ' ' + 'Confusion matrix')
    conf_matrix = metric.value
    class_names = list(range(len(conf_matrix)))
    ConfusionMatrixDisplay(conf_matrix, class_names).plot(cmap=plt.cm.Blues, ax=ax)
