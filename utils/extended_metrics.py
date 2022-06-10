import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score


# all_preds [test_size, cat_vocab_size]
# all_gt [test_size, cat_vocab_size]

# Example-Based
def precision_ex_based(all_preds, all_gt):
    return np.mean([np.sum(np.logical_and(all_preds, all_gt), axis=1)[i] / np.sum(all_preds, axis=1)[i]
                    if np.sum(all_preds, axis=1)[i] > 0 else 0 for i in range(all_gt.shape[0])])


def recall_ex_based(all_preds, all_gt):
    return np.mean(np.sum(np.logical_and(all_preds, all_gt), axis=1) / np.sum(all_gt, axis=1))


def f1_ex_based(all_preds, all_gt):
    precision_samples = np.sum(np.logical_and(all_preds, all_gt), axis=1) / np.sum(all_preds, axis=1)
    recall_samples = np.sum(np.logical_and(all_preds, all_gt), axis=1) / np.sum(all_gt, axis=1)
    return np.mean([2 * precision_samples[i] * recall_samples[i] / (precision_samples[i] + recall_samples[i])
                    if (precision_samples[i] + recall_samples[i]) > 0 else 0 for i in range(all_gt.shape[0])])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def valid_thr(gt_valid, scores_valid):
    # gt_valid [sample_size, cat_vocab_size] - one-hot
    # scores_valid [sample_size, cat_vocab_size]

    sigmoid_scores_valid = sigmoid(scores_valid)
    possible_thr = np.arange(0, 1, 0.01)
    final_thr = np.array([possible_thr[np.argmax(np.array([f1_score(gt_valid[:, j], sigmoid_scores_valid[:, j] >= thr)
                                                 for thr in possible_thr]))] for j in range(gt_valid.shape[1])])
    # final_thr [cat_vocab_size]
    k = np.median(np.sum(gt_valid, axis=1))
    return final_thr, k


def calculate_all_metrics(all_gt, all_scores, gt_valid, scores_valid, kind='thr'):
    # all_gt [sample_size, cat_vocab_size] - one-hot
    # all_scores [sample_size, cat_vocab_size] - confidence-scores
    tasks_with_non_trivial_targets = np.where(all_gt.sum(axis=0) != 0)[0]
    all_gt = all_gt[:, tasks_with_non_trivial_targets]
    all_scores = all_scores[:, tasks_with_non_trivial_targets]
    all_scores_sigmoid = sigmoid(all_scores)

    final_thr, k = valid_thr(gt_valid, scores_valid)
    final_thr = final_thr[tasks_with_non_trivial_targets]

    if kind == 'thr':
        all_preds = np.array([(all_scores_sigmoid[b, :] >= final_thr).astype(np.int64) for b in range(all_gt.shape[0])])
    if kind == 'topk':
        all_preds = np.array([np.insert(np.zeros(all_gt.shape[1]), sorted(range(len(all_scores[b, :])), key=lambda i: all_scores[b, :][i])[-k:], 1).tolist()
                    for b in range(all_gt.shape[0])])

    metrics_dict = {'precision_ex_based': precision_ex_based(all_preds, all_gt),
                    'recall_ex_based': recall_ex_based(all_preds, all_gt),
                    'f1_ex_based': f1_ex_based(all_preds, all_gt),
                    'f1_micro': f1_score(all_gt, all_preds, average='micro'),
                    'f1_macro': f1_score(all_gt, all_preds, average='macro'),
                    'roc_auc_micro': roc_auc_score(all_gt, all_scores, average='micro'),
                    'roc_auc_macro': roc_auc_score(all_gt, all_scores, average='macro')}
    return metrics_dict
