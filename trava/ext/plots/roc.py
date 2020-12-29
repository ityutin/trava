from sklearn.metrics import auc, roc_curve

from trava.ext.results_handlers.plotter import PlotItem
from trava.ext.results_handlers.scorer_plotter import ScorerPlotter
from trava.ext.sklearn.scorers import sk_proba
from trava.metric import Metric


class ROCCurvesPlotter(ScorerPlotter):
    def plot(self, metric: Metric, fig, ax, color: str, label: str):
        ax.set_title(f"{metric.model_id}: {label} ROC")

        fpr, tpr, threshold = metric.value
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label="{} AUC = {:0.2f}".format(metric.model_id, roc_auc), color=color)
        ax.legend(loc="lower right")
        ax.plot([0, 1], [0, 1], "r--")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel("True Positive Rate")
        ax.set_xlabel("False Positive Rate")


class ROCCurvesPlotItem(PlotItem):
    def __init__(self):
        self.scorer = sk_proba(roc_curve)
        self.plotter = ROCCurvesPlotter()
        self.can_overlap = True
