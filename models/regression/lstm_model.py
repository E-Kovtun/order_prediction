import torch
from torch import nn


class LSTMEncoding(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim):
        super(LSTMEncoding, self).__init__()

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_dim, batch_first=True)

    def forward(self, x_changing):
        x_out,( _, _) = self.lstm(x_changing)
        x_last_state = x_out[:, -1, :]
        return x_last_state


class RegressionNet(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, linear_encoding1_dim, linear_encoding2_dim,
                 vocab_size0, vocab_size1, vocab_size2, const_embedding_dim,
                 linear_embedding1_dim, linear_embedding2_dim,
                 linear_concat1):
        super(RegressionNet, self).__init__()

        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.linear_encoding1_dim = linear_encoding1_dim
        self.linear_encoding2_dim = linear_encoding2_dim
        self.vocab_size0 = vocab_size0
        self.vocab_size1 = vocab_size1
        self.vocab_size2 = vocab_size2
        self.const_embedding_dim = const_embedding_dim
        self.linear_embedding1_dim = linear_embedding1_dim
        self.linear_embedding2_dim = linear_embedding2_dim
        self.linear_concat1 = linear_concat1

        self.relu = nn.ReLU()

        self.lstm_encoding = LSTMEncoding(lstm_input_dim, lstm_hidden_dim)
        self.linear_encoding1 = nn.Linear(lstm_hidden_dim, linear_encoding1_dim)
        self.linear_encoding2 = nn.Linear(linear_encoding1_dim, linear_encoding2_dim)

        self.embedding0 = nn.Embedding(num_embeddings=vocab_size0, embedding_dim=const_embedding_dim)
        self.embedding1 = nn.Embedding(num_embeddings=vocab_size1, embedding_dim=const_embedding_dim)
        self.embedding2 = nn.Embedding(num_embeddings=vocab_size2, embedding_dim=const_embedding_dim)

        self.linear_embedding1 = nn.Linear(3 * const_embedding_dim, linear_embedding1_dim)
        self.linear_embedding2 = nn.Linear(linear_embedding1_dim, linear_embedding2_dim)

        self.linear_concat1 = nn.Linear(linear_encoding2_dim+linear_embedding2_dim, linear_concat1)
        self.linear_concat2 = nn.Linear(linear_concat1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_changing, x_const):

        x_encoding = self.lstm_encoding(x_changing) # [batch_size, lstm_hidden_dim]
        x_enc1 = self.linear_encoding1(x_encoding) # [batch_size, linear_encoding1_dim]
        x_enc1 = self.relu(x_enc1)
        x_enc2 = self.linear_encoding2(x_enc1) # [batch_size, linear_encoding2_dim]

        x_embedding0 = self.embedding0(x_const[:, 0]) # [batch_size, const_embedding_dim]
        x_embedding1 = self.embedding1(x_const[:, 1])  # [batch_size, const_embedding_dim]
        x_embedding2 = self.embedding2(x_const[:, 2])  # [batch_size, const_embedding_dim]
        x_emb_concat = torch.cat((x_embedding0, x_embedding1, x_embedding2), dim=-1)
        x_emb1 = self.linear_embedding1(x_emb_concat) # [batch_size, linear_embedding1_dim]
        x_emb1 = self.relu(x_emb1)
        x_emb2 = self.linear_embedding2(x_emb1) # [batch_size, linear_embedding2_dim]

        x_concat_feat = torch.cat((x_enc2, x_emb2), dim=-1) # [batch_size, linear_encoding2_dim+linear_embedding2_dim]
        x_out1 = self.linear_concat1(x_concat_feat) # [batch_size, linear_concat1]
        x_out1 = self.relu(x_out1)
        x_out2 = self.linear_concat2(x_out1) # [batch_size, 1]

        x_final = self.sigmoid(x_out2) # [batch_size, 1]
        return x_final


# if __name__ == '__main__':
#     lstm_input_dim = 2
#     lstm_hidden_dim = 128
#     linear_encoding1_dim = 64
#     linear_encoding2_dim = 32
#     vocab_size0 = 1000
#     vocab_size1 = 700
#     vocab_size2 = 300
#     const_embedding_dim = 256
#     linear_embedding1_dim = 128
#     linear_embedding2_dim = 64
#
#     linear_concat1 = 32
#
#     net = RegressionNet(lstm_input_dim, lstm_hidden_dim, linear_encoding1_dim, linear_encoding2_dim,
#                  vocab_size0, vocab_size1, vocab_size2, const_embedding_dim,
#                  linear_embedding1_dim, linear_embedding2_dim,
#                  linear_concat1)
#     chang_matr = torch.rand((100, 4, 2))
#     const_matr = torch.randint(0, 200, (100, 3))
#     res = net(chang_matr, const_matr)
#     print(res.shape)
