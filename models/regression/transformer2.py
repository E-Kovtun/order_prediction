import torch
from torch import nn


class TransformerNet(nn.Module):
    def __init__(self, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim):
        super(TransformerNet, self).__init__()

        self.cat_vocab_size = cat_vocab_size
        self.emb_dim = emb_dim

        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=emb_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_size, embedding_dim=emb_dim)
        self.amount_embedding = nn.Embedding(num_embeddings=amount_vocab_size, embedding_dim=emb_dim)
        self.dt_embedding = nn.Embedding(num_embeddings=dt_vocab_size, embedding_dim=emb_dim)

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=2, dim_feedforward=emb_dim,
                                                              dropout=0.1, activation='relu', batch_first=True)

        self.linear1 = nn.Linear(emb_dim, cat_vocab_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(cat_vocab_size, cat_vocab_size)

    def forward(self, cat_arr, dt_arr, amount_arr, id_arr):
        x_id_emb = self.id_embedding(id_arr).unsqueeze(1)  # [batch_size, 1, emb_dim]

        x_cat_emb = torch.stack([torch.stack([torch.sum(torch.stack([self.cat_embedding(cat_arr[b, lb, j])
                                                        for j in torch.where(cat_arr[b, lb, :]!=self.cat_vocab_size)[0]], dim=0), dim=0)
                                              for lb in range(cat_arr.shape[1])], dim=0)
                                for b in range(cat_arr.shape[0])], dim=0) # [batch_size, look_back, emb_dim]

        x_amount_emb = torch.stack([torch.stack([torch.sum(torch.stack([self.amount_embedding(amount_arr[b, lb, j])
                                                           for j in torch.where(cat_arr[b, lb, :]!=self.amount_vocab_size)[0]], dim=0), dim=0)
                                                 for lb in range(amount_arr.shape[1])], dim=0)
                                    for b in range(amount_arr.shape[0])], dim=0) # [batch_size, look_back, emb_dim]

        x_dt_emb = torch.stack([torch.stack([self.dt_embedding(dt_arr[b, lb]) for lb in range(dt_arr.shape[1])], dim=0)
                                for b in range(dt_arr.shape[0])]) # [batch_size, look_back, emb_dim]

        x_encoder_input = torch.cat([x_id_emb, (x_cat_emb + x_amount_emb + x_dt_emb)], dim=1) # [batch_size, look_back+1, emb_dim]
        x_encoder_output = self.transformer_encoder(x_encoder_input)[:, 1:, :] # [batch_size, look_back, emb_dim]
        x_encoder_output = torch.mean(x_encoder_output, dim=1)  # [batch_size, emb_dim]
        # x_encoder_output = torch.mean(x_encoder_output, dim=2) # [batch_size, 3]

        x1 = self.linear1(x_encoder_output) # [batch_size, cat_vocab_size]
        x1 = self.relu(x1)
        x2 = self.linear2(x1) # [batch_size, cat_vocab_size]

        return x2

