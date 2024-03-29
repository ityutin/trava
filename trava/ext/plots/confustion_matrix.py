import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from trava.ext.results_handlers.plotter import PlotItem
from trava.ext.results_handlers.scorer_plotter import ScorerPlotter
from trava.ext.sklearn.scorers import sk_binary_with_threshold
from trava.metric import Metric


class ConfMatrixPlotter(ScorerPlotter):
    def plot(self, metric: Metric, fig, ax, color: str, label: str):
        ax.set_title(f"{metric.model_id}: {label} Confusion matrix")
        conf_matrix = metric.value
        ConfusionMatrixDisplay(conf_matrix).plot(cmap=plt.cm.Blues, ax=ax)


class ConfMatrixPlotItem(PlotItem):
    def __init__(self, normalize: str, threshold=None):
        self.scorer = sk_binary_with_threshold(confusion_matrix, custom_threshold=threshold, normalize=normalize)
        self.plotter = ConfMatrixPlotter()
        self.can_overlap = False
