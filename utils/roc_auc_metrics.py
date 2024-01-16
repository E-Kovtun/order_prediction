from sklearn.metrics import roc_auc_score
import numpy as np


def calculate_roc_auc_metrics(all_scores, all_gt):
    tasks_with_non_trivial_targets = np.where(all_gt.sum(axis=0) != 0)[0]
    ## remove tasks where we have target only 0
    all_scores = all_scores[:, tasks_with_non_trivial_targets]
    all_gt = all_gt[:, tasks_with_non_trivial_targets]
    metrics_dict = {'roc_auc_micro': roc_auc_score(all_gt, all_scores, average='micro'),
                    'roc_auc_macro': roc_auc_score(all_gt, all_scores, average='macro')}
    return metrics_dict