import torch
from torch import nn


class LSTMEncoding(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim):
        super(LSTMEncoding, self).__init__()

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_dim, batch_first=True)

    def forward(self, x):
        x_out, (_, _) = self.lstm(x)
        x_last_state = x_out[:, -1, :]
        return x_last_state


class RegressionNet(nn.Module):
    def __init__(self, linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                 cat_vocab_size, id_vocab_size,
                 id_embedding_dim, linear_concat1_dim, linear_concat2_dim):
        super(RegressionNet, self).__init__()

        self.relu = nn.ReLU()

        self.lstm_encoding_cat = LSTMEncoding(cat_embedding_dim, lstm_hidden_dim)
        self.lstm_encoding_num = LSTMEncoding(linear_num_feat_dim, lstm_hidden_dim)

        self.linear_num_feat = nn.Linear(2, linear_num_feat_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_size + 1, embedding_dim=cat_embedding_dim,
                                          padding_idx=cat_vocab_size)
        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=id_embedding_dim)

        self.linear_concat1 = nn.Linear(2 * lstm_hidden_dim + id_embedding_dim, linear_concat1_dim)
        self.linear_concat2 = nn.Linear(linear_concat1_dim, linear_concat2_dim)
        self.linear_final = nn.Linear(linear_concat2_dim, cat_vocab_size + 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, cat_arr, mask_cat, current_cat, mask_current_cat, onehot_current_cat, num_arr, id_arr):
        x_cat_emb = self.cat_embedding(cat_arr)  # [batch_size, look_back, max_day_len0, cat_embedding_dim]
        x_cat_emb = x_cat_emb * mask_cat.unsqueeze(3)
        x_cat_emb_sum = torch.sum(x_cat_emb, dim=2)  # [batch_size, look_back, cat_embedding_dim]

        x_current_cat_emb = self.cat_embedding(current_cat)  # [batch_size, max_day_len1, cat_embedding_dim]
        x_current_cat_emb = x_current_cat_emb * mask_current_cat
        x_current_cat_emb_sum = torch.sum(x_current_cat_emb, dim=1).unsqueeze(1)  # [batch_size, 1, cat_embedding_dim]

        x_lstm_input_cat = torch.cat((x_cat_emb_sum, x_current_cat_emb_sum),
                                     dim=1)  # [batch_size, look_back+1, cat_embedding_dim]
        x_encoding_cat = self.lstm_encoding_cat(x_lstm_input_cat)  # [batch_size, lstm_hidden_dim]

        x_num1 = self.linear_num_feat(num_arr)  # [batch_size, look_back, linear_num_feat_dim]
        x_encoding_num = self.lstm_encoding_num(x_num1)  # [batch_size, lstm_hidden_dim]

        x_id_emb = self.id_embedding(id_arr).squeeze(1)  # [batch_size, id_embedding_dim]
        x_concat = torch.cat((x_encoding_cat, x_encoding_num, x_id_emb),
                             dim=1)  # [batch_size, 2*lstm_hidden_dim+id_embedding_dim]

        x1 = self.linear_concat1(x_concat)
        x1 = self.relu(x1)
        x2 = self.linear_concat2(x1)
        x2 = self.relu(x2)
        x_out = self.linear_final(x2)  # [batch_size, cat_vocab_size+1]
        # x_out = self.sigmoid(x_out)
        x_final = x_out * onehot_current_cat  # [batch_size, cat_vocab_size+1]

        return x_final
