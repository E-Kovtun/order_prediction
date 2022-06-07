import numpy as np
import torch

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


# Label-based
# Macro averaged
def precision_macro(all_preds, all_gt):
    return np.mean([np.sum(np.logical_and(all_preds, all_gt), axis=0)[i] / np.sum(all_preds, axis=0)[i]
                    if np.sum(all_preds, axis=0)[i] > 0 else 0 for i in range(all_preds.shape[1])])


def recall_macro(all_preds, all_gt):
    return np.mean([np.sum(np.logical_and(all_preds, all_gt), axis=0)[i] / np.sum(all_gt, axis=0)[i]
                    if np.sum(all_gt, axis=0)[i] > 0 else 0 for i in range(all_gt.shape[1])])


def f1_macro(all_preds, all_gt):
    return np.mean([2 * np.sum(all_preds * all_gt, axis=0)[i] / (np.sum(all_preds, axis=0)[i] + np.sum(all_gt, axis=0)[i])
                    if (np.sum(all_preds, axis=0)[i] + np.sum(all_gt, axis=0)[i]) > 0 else 0 for i in range(all_gt.shape[1])])


# Micro averaged
def precision_micro(all_preds, all_gt):
    return np.sum(np.logical_and(all_preds, all_gt)) / np.sum(all_preds)


def recall_micro(all_preds, all_gt):
    return np.sum(np.logical_and(all_preds, all_gt)) / np.sum(all_gt)


def f1_micro(all_preds, all_gt):
    return np.sum(2 * all_preds * all_gt) / (np.sum(all_preds) + np.sum(all_gt))


def map(all_scores, all_gt):
    all_scores = torch.tensor(all_scores)
    all_gt = torch.tensor(all_gt)
    return np.mean([np.mean([len(np.intersect1d(torch.topk(all_scores[b, :], k=i, dim=0).indices.numpy(),
                                                      torch.where(all_gt[b, :] == 1)[0].numpy())) / i
                                    for i in range(1, torch.sum(all_gt[b, :])+1)]) for b in range(all_scores.shape[0])])


def calculate_all_metrics(all_preds, all_gt, all_scores):
    metrics_dict = {'precision_ex_based': precision_ex_based(all_preds, all_gt), 'recall_ex_based': recall_ex_based(all_preds, all_gt),
                    'f1_ex_based': f1_ex_based(all_preds, all_gt),
                    'precision_macro': precision_macro(all_preds, all_gt), 'recall_macro': recall_macro(all_preds, all_gt),
                    'f1_macro': f1_macro(all_preds, all_gt),
                    'precision_micro': precision_micro(all_preds, all_gt), 'recall_micro': recall_micro(all_preds, all_gt),
                    'f1_micro': f1_micro(all_preds, all_gt),
                    'map': map(all_scores, all_gt)}
    return metrics_dict
