import torch
from torch import nn
from torch.nn.functional import one_hot
from models.C2AE.C2AE_class import C2AE, Fd, Fe, Fx
from models.regression.lstm_material import ClassificationNet


class C2AE_Transformer_invert(nn.Module):
    def __init__(self, look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim,
                 device="cpu"):
        super().__init__()
        self.device = torch.device(device)

        linear_num_feat_dim = 32
        cat_embedding_dim = 512
        lstm_hidden_dim = 1024
        id_embedding_dim = 512
        linear_concat1_dim = 1024
        linear_concat2_dim = 512

        # C2AE for cat
        classifier_net = ClassificationNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                                           cat_vocab_size, id_vocab_size,
                                           id_embedding_dim, linear_concat1_dim, linear_concat2_dim)
        classifier_net.linear_material = nn.Linear(linear_concat2_dim, 512)
        classifier_net.bn3 = nn.BatchNorm1d(512)
        self.num_labels_cat = cat_vocab_size + 1 # 11 for default dataset # 61 + 1

        latent_dim = 512
        fx_hidden_dim = 1024
        fe_hidden_dim = 256
        fd_hidden_dim = 256
        alpha = 10
        beta = 0.5

        fx = Fx(512, fx_hidden_dim, fx_hidden_dim, latent_dim).to(device)
        fe = Fe(self.num_labels_cat, fe_hidden_dim, latent_dim).to(device)
        fd = Fd(latent_dim, fd_hidden_dim, 512).to(device)
        self.c2ae_cat = C2AE(classifier_net.to(device), fx, fe, fd, beta=beta, alpha=alpha, emb_lambda=0.01,
                             latent_dim=latent_dim,
                             device=device).to(device)

        # C2AE for amount
        classifier_net = ClassificationNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                                           amount_vocab_size, id_vocab_size,
                                           id_embedding_dim, linear_concat1_dim, linear_concat2_dim)
        classifier_net.linear_material = nn.Linear(linear_concat2_dim, 512)
        classifier_net.bn3 = nn.BatchNorm1d(512)
        self.num_labels_amount = amount_vocab_size

        fx = Fx(512, fx_hidden_dim, fx_hidden_dim, latent_dim).to(device)
        fe = Fe(self.num_labels_amount, fe_hidden_dim, latent_dim).to(device)
        fd = Fd(latent_dim, fd_hidden_dim, 512).to(device)
        self.c2ae_amount = C2AE(classifier_net.to(device), fx, fe, fd, beta=beta, alpha=alpha, emb_lambda=0.01,
                                latent_dim=latent_dim,
                                device=device).to(device)
        self.look_back = look_back
        self.cat_vocab_size = cat_vocab_size
        self.amount_vocab_size = amount_vocab_size
        self.emb_dim = emb_dim

        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=3 * emb_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=512, embedding_dim=emb_dim)
        self.amount_embedding = nn.Embedding(num_embeddings=512 + 1, embedding_dim=emb_dim,
                                             padding_idx=512)
        self.dt_embedding = nn.Embedding(num_embeddings=dt_vocab_size, embedding_dim=emb_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=look_back, embedding_dim=emb_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=3 * emb_dim * 2, nhead=4, dim_feedforward=3 * emb_dim * 2,
                                                        dropout=0.2, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.bn0 = nn.BatchNorm1d(cat_vocab_size)
        self.linear1 = nn.Linear(cat_vocab_size, cat_vocab_size)
        self.bn1 = nn.BatchNorm1d(cat_vocab_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(cat_vocab_size, cat_vocab_size)

    def forward(self, cat_arr, cat_mask, dt_arr, amount_arr, id_arr, batch_onehot_current_cat, batch_target):
        print("cat_arr", cat_arr.shape)
        cat_fx_x, cat_fe_y, cat_fd_z = self.c2ae_cat(cat_arr, cat_mask, amount_arr, id_arr,
                                                     batch_onehot_current_cat)
        x_id_emb = self.id_embedding(id_arr).unsqueeze(1)  # [batch_size, 1, emb_dim]
        #x_onehot_cat = self.cat_embedding(torch.arange(512,
        #                                            device=self.device)).unsqueeze(
        #    0).expand(cat_fd_z.shape[0], -1, -1)
        #x_onehot_cat = torch.sum(one_hot(cat_fd_z, num_classes=512 + 1) * cat_mask, dim=2)[:, :,
        #               :-1]  # [batch_size, look_back, 512]

        amount_fx_x, amount_fe_y, amount_fd_z = self.c2ae_amount(cat_arr, cat_mask, amount_arr, id_arr,
                                                                 batch_target)
        #x_amount_emb = amount_fd_z.reshape(-1, 512, 1) # [batch_size, 512, emb_dim]

        dt_arr = torch.sum((self.dt_embedding(dt_arr) +
                              self.pos_embedding(torch.arange(dt_arr.shape[1],
                              device=self.device)).unsqueeze(0).expand(dt_arr.shape[0], -1, -1))
                           .unsqueeze(2).expand(-1, -1, self.cat_vocab_size, -1), dim=1)

        print("x_id_emb", x_id_emb.shape)
        print("cat_fd_z", cat_fd_z.shape)
        print("dt", dt_arr.shape)
        print("amount_fd_z", amount_fd_z.shape)


        # x_encoder_input = torch.cat([x_id_emb, (x_cat_emb + x_dt_emb + x_amount_emb)], dim=1) # [batch_size, cat_vocab_size+1, emb_dim]
        x_encoder_input = torch.cat([x_id_emb.squeeze(), torch.cat([cat_fd_z, torch.mean(dt_arr, dim=1), amount_fd_z], dim=1)], dim=1)

        x_encoder_output = self.transformer_encoder(x_encoder_input).reshape(x_id_emb.shape[0], 64, -1)[:, 3:, :]  # [batch_size, cat_vocab_size, emb_dim]
        print("x_encoder_output", x_encoder_output.shape)
        x_encoder_output = torch.mean(x_encoder_output, dim=2)  # [batch_size, cat_vocab_size]

        x_encoder_output = self.bn0(x_encoder_output)
        x1 = self.linear1(x_encoder_output)  # [batch_size, cat_vocab_size]
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.linear2(x1)  # [batch_size, cat_vocab_size]

        if self.training:
            return x2, cat_fx_x, cat_fe_y, cat_fd_z, amount_fx_x, amount_fe_y, amount_fd_z
        else:
            return x2