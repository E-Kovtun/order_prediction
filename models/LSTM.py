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


class LSTMnet(nn.Module):
    def __init__(self, look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, max_cat_len, emb_dim):
        super(LSTMnet, self).__init__()

        self.cat_vocab_size = cat_vocab_size
        self.relu = nn.ReLU()

        self.lstm_encoding_cat = LSTMEncoding(3*emb_dim, 3*emb_dim)
        self.linear_history1 = nn.Linear(3*emb_dim+emb_dim, 2*emb_dim)
        self.linear_history2 = nn.Linear(2*emb_dim, cat_vocab_size)

        self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_size+1, embedding_dim=emb_dim,
                                          padding_idx=cat_vocab_size)
        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=emb_dim)
        self.amount_embedding = nn.Embedding(num_embeddings=amount_vocab_size+1, embedding_dim=emb_dim,
                                             padding_idx=amount_vocab_size)
        self.dt_embedding = nn.Embedding(num_embeddings=dt_vocab_size, embedding_dim=emb_dim)


    def forward(self, cat_arr, dt_arr, amount_arr, id_arr):
        x_cat_emb = self.cat_embedding(cat_arr)  # [batch_size, look_back, max_cat_len, emb_dim]
        x_mask_cat = torch.tensor(~(cat_arr == self.cat_vocab_size), dtype=torch.int64).unsqueeze(3) # [batch_size, look_back, max_cat_len, 1]
        x_cat_emb = x_cat_emb * x_mask_cat
        x_cat_emb_sum = torch.sum(x_cat_emb, dim=2)  # [batch_size, look_back, emb_dim]

        x_amount_emb = self.amount_embedding(amount_arr)  # [batch_size, look_back, max_cat_len, emb_dim]
        x_amount_emb = x_amount_emb * x_mask_cat
        x_amount_emb_sum = torch.sum(x_amount_emb, dim=2)  # [batch_size, look_back, emb_dim]

        x_dt_emb = self.dt_embedding(dt_arr) # [batch_size, look_back, emb_dim]

        x_encoding = self.lstm_encoding_cat(torch.cat((x_cat_emb_sum, x_amount_emb_sum, x_dt_emb), dim=2))  # [batch_size, 3*emb_dim]

        x_id_emb = self.id_embedding(id_arr).squeeze(1)  # [batch_size, emb_dim]
        x_concat = torch.cat((x_encoding, x_id_emb), dim=1)  # [batch_size, 4*emb_dim]

        x1 = self.linear_history1(x_concat)
        x1 = self.relu(x1)
        z_history = self.linear_history2(x1)

        return z_history