import torch
from torch.nn.functional import cross_entropy


def multilabel_crossentropy_loss(x_conf_scores, x_current_cat, cat_vocab_size):
    # x_conf_scores (logits) - batch_size x cat_vocab_size
    # x_current_cat - batch_size x max_cat_len
    multi_loss = torch.sum(torch.stack([torch.sum(torch.stack([cross_entropy(x_conf_scores[b, :].reshape(1, -1),
                                                                             label.reshape(-1))
                                                  for label in x_current_cat[b, :] if label != cat_vocab_size], dim=0))
                           for b in range(x_conf_scores.shape[0])]), dim=0)
    return multi_loss
