import torch
from torch import nn
from torch.nn.functional import one_hot


class TransformerNet(nn.Module):
    def __init__(self, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim):
        super(TransformerNet, self).__init__()

        self.cat_vocab_size = cat_vocab_size
        self.amount_vocab_size = amount_vocab_size
        self.emb_dim = emb_dim

        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=emb_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_size, embedding_dim=emb_dim)
        self.amount_embedding = nn.Embedding(num_embeddings=amount_vocab_size+1, embedding_dim=emb_dim,
                                             padding_idx=amount_vocab_size)
        self.dt_embedding = nn.Embedding(num_embeddings=dt_vocab_size, embedding_dim=emb_dim)

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=2, dim_feedforward=emb_dim,
                                                              dropout=0.1, activation='relu', batch_first=True)

        self.linear1 = nn.Linear(cat_vocab_size, cat_vocab_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(cat_vocab_size , cat_vocab_size)

    def forward(self, cat_arr, dt_arr, amount_arr, id_arr):

        # x_cat_emb = self.cat_embedding(torch.arange(self.cat_vocab_size)) # [batch_size, cat_vocab_size, emb_dim]

        # x_dt_emb = torch.stack([torch.stack([torch.stack([self.dt_embedding(dt_arr[b, lb]) if i in cat_arr[b, lb, :]
        #                                                   else torch.zeros(self.emb_dim, dtype=torch.float32)
        #                                                   for i in range(self.cat_vocab_size)], dim=0)
        #                                      for lb in range(dt_arr.shape[1])], dim=0)
        #                         for b in range(cat_arr.shape[0])], dim=0) # [batch_size, look_back, cat_vocab_size, emb_dim]
        # x_dt_emb = torch.sum(x_dt_emb, dim=1) # [batch_size, cat_vocab_size, emb_dim]
        #
        # x_amount_emb = torch.stack([torch.stack([torch.stack([self.amount_embedding(amount_arr[b, lb, torch.where(cat_arr[b, lb, :]==i)[0].item()])
        #                                                       if i in cat_arr[b, lb, :]
        #                                                       else torch.zeros(self.emb_dim, dtype=torch.float32)
        #                                                       for i in range(self.cat_vocab_size)], dim=0)
        #                                         for lb in range(dt_arr.shape[1])], dim=0)
        #                             for b in range(cat_arr.shape[0])], dim=0) # [batch_size, look_back, cat_vocab_size, emb_dim]
        #
        # x_amount_emb = torch.sum(x_amount_emb, dim=1) # [batch_size, cat_vocab_size, emb_dim]

        # torch.sum(one_hot(current_cat, num_classes=self.cat_vocab_size + 1) * mask_current_cat, dim=0)

        x_id_emb = self.id_embedding(id_arr).unsqueeze(1)  # [batch_size, 1, emb_dim]
        x_cat_emb = self.cat_embedding(torch.arange(self.cat_vocab_size,
                                                    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')))
                                                    # [batch_size, cat_vocab_size, emb_dim]
        x_mask_cat = torch.tensor(~(cat_arr == self.cat_vocab_size), dtype=torch.int64).unsqueeze(3) # [batch_size, look_back, max_cat_len, 1]
        x_onehot_cat = torch.sum(one_hot(cat_arr, num_classes=self.cat_vocab_size+1) * x_mask_cat, dim=2)[:, :, :-1 ]# [batch_size, look_back, cat_vocab_size]

        x_amount_emb = torch.sum(self.amount_embedding(x_onehot_cat.index_put_(indices=torch.where(x_onehot_cat==0),
                                               values=torch.tensor(self.amount_vocab_size)).index_put_(indices=torch.where(x_onehot_cat==1),
                                                                 values=amount_arr.flatten()[amount_arr.flatten()!=self.amount_vocab_size])) *
                                 x_onehot_cat.unsqueeze(3), dim=1) # [batch_size, cat_vocab_size, emb_dim]

        x_dt_emb = torch.sum(self.dt_embedding(dt_arr).unsqueeze(2).expand(-1, -1, self.cat_vocab_size, -1) * x_onehot_cat.unsqueeze(3), dim=1) # [batch_size, cat_vocab_size, emb_dim ]

        x_encoder_input = torch.cat([x_id_emb, (x_cat_emb + x_dt_emb + x_amount_emb)], dim=1) # [batch_size, cat_vocab_size+1, emb_dim]

        x_encoder_output = self.transformer_encoder(x_encoder_input)[:, 1:, :] # [batch_size, cat_vocab_size, emb_dim]
        x_encoder_output = torch.mean(x_encoder_output, dim=2) # [batch_size, cat_vocab_size]

        x1 = self.linear1(x_encoder_output) # [batch_size, cat_vocab_size]
        x1 = self.relu(x1)
        x2 = self.linear2(x1) # [batch_size, cat_vocab_size]

        return x2