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


class DoubleVariableNet(nn.Module):
    def __init__(self, linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                 cat_vocab_size, id_vocab_size,
                 id_embedding_dim, linear_concat1_dim, linear_concat2_dim):
        super(DoubleVariableNet, self).__init__()

        self.relu = nn.ReLU()

        self.lstm_encoding_cat = LSTMEncoding(cat_embedding_dim, lstm_hidden_dim)

        self.linear_num_feat = nn.Linear(2, linear_num_feat_dim)
        self.lstm_encoding_num = LSTMEncoding(linear_num_feat_dim, lstm_hidden_dim)

        self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_size + 1, embedding_dim=cat_embedding_dim,
                                          padding_idx=cat_vocab_size)
        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=id_embedding_dim)

        self.linear_concat1_mat = nn.Linear(2 * lstm_hidden_dim + id_embedding_dim, linear_concat1_dim)
        self.linear_concat2_mat = nn.Linear(linear_concat1_dim, linear_concat2_dim)
        self.linear_material = nn.Linear(linear_concat2_dim, cat_vocab_size)

        self.linear_concat1_amount = nn.Linear(2 * lstm_hidden_dim + id_embedding_dim, linear_concat1_dim)
        self.linear_concat2_amount = nn.Linear(linear_concat1_dim, linear_concat2_dim)
        self.linear_amount = nn.Linear(linear_concat2_dim, cat_vocab_size + 1)

    def forward(self, cat_arr, mask_cat, num_arr, id_arr):
        x_cat_emb = self.cat_embedding(cat_arr)  # [batch_size, look_back, max_day_len0, cat_embedding_dim]
        x_cat_emb = x_cat_emb * mask_cat.unsqueeze(3)
        x_cat_emb_sum = torch.sum(x_cat_emb, dim=2)  # [batch_size, look_back, cat_embedding_dim]

        x_encoding_cat = self.lstm_encoding_cat(x_cat_emb_sum)  # [batch_size, lstm_hidden_dim]

        x_num1 = self.linear_num_feat(num_arr)  # [batch_size, look_back, linear_num_feat_dim]
        x_encoding_num = self.lstm_encoding_num(x_num1)  # [batch_size, lstm_hidden_dim]

        x_id_emb = self.id_embedding(id_arr).squeeze(1)  # [batch_size, id_embedding_dim]
        x_concat = torch.cat((x_encoding_cat, x_encoding_num, x_id_emb), dim=1)  # [batch_size, 2*lstm_hidden_dim+id_embedding_dim]

        x1_amount = self.linear_concat1_amount(x_concat)
        x1_amount = self.relu(x1_amount)
        x2_amount = self.linear_concat2_amount(x1_amount)
        x2_amount = self.relu(x2_amount)
        x_amount = self.linear_amount(x2_amount)  # [batch_size, cat_vocab_size+1]

        x1_mat = self.linear_concat1_mat(x_concat)
        x1_mat = self.relu(x1_mat)
        x2_mat = self.linear_concat2_mat(x1_mat)
        x2_mat = self.relu(x2_mat)
        x_material = self.linear_material(x2_mat)  # [batch_size, cat_vocab_size]

        return x_amount, x_material