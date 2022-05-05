import torch
from torch import nn
from torch.nn.functional import one_hot


class EncoderTemplate(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_size):
        super(EncoderTemplate, self).__init__()

        self.linear_enc = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = self.linear_enc(x)
        x = self.relu(x)
        enc_mu = self.mu(x)
        enc_logvar = self.logvar(x)
        return enc_mu, enc_logvar


class DecoderTemplate(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(DecoderTemplate, self).__init__()

        self.linear_dec1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.linear_dec2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.linear_dec1(x)
        x = self.relu(x)
        x = self.linear_dec2(x)
        return x


class TransformerNet(nn.Module):
    def __init__(self, look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim):
        super(TransformerNet, self).__init__()

        self.look_back = look_back
        self.cat_vocab_size = cat_vocab_size
        self.amount_vocab_size = amount_vocab_size
        self.emb_dim = emb_dim

        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=3*emb_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_size, embedding_dim=emb_dim)
        self.amount_embedding = nn.Embedding(num_embeddings=amount_vocab_size+1, embedding_dim=emb_dim,
                                             padding_idx=amount_vocab_size)
        self.dt_embedding = nn.Embedding(num_embeddings=dt_vocab_size, embedding_dim=emb_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=look_back, embedding_dim=emb_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=3*emb_dim, nhead=4, dim_feedforward=3*emb_dim,
                                                        dropout=0.2, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, cat_arr, dt_arr, amount_arr, id_arr):
        x_id_emb = self.id_embedding(id_arr).unsqueeze(1)  # [batch_size, 1, emb_dim]
        x_cat_emb = self.cat_embedding(torch.arange(self.cat_vocab_size,
                                                    device=torch.device('cuda:0')
                                                    if torch.cuda.is_available() else torch.device('cpu'))).unsqueeze(0).expand(cat_arr.shape[0], -1, -1)
                                                    # [batch_size, cat_vocab_size, emb_dim]
        x_mask_cat = torch.tensor(~(cat_arr == self.cat_vocab_size), dtype=torch.int64).unsqueeze(3) # [batch_size, look_back, max_cat_len, 1]
        x_onehot_cat = torch.sum(one_hot(cat_arr, num_classes=self.cat_vocab_size+1) * x_mask_cat, dim=2)[:, :, :-1 ]# [batch_size, look_back, cat_vocab_size]

        x_amount_emb = torch.sum(self.amount_embedding(x_onehot_cat.index_put_(indices=torch.where(x_onehot_cat==0),
                                               values=torch.tensor(self.amount_vocab_size,
                                                                   device=torch.device('cuda:0') if torch.cuda.is_available()
                                                                   else torch.device('cpu'))).index_put_(indices=torch.where(x_onehot_cat==1),
                                                                                        values=amount_arr.flatten()[amount_arr.flatten()!=self.amount_vocab_size])) *
                                 x_onehot_cat.unsqueeze(3), dim=1) # [batch_size, cat_vocab_size, emb_dim]

        x_dt_emb = torch.sum((self.dt_embedding(dt_arr) +
                              self.pos_embedding(torch.arange(dt_arr.shape[1],
                              device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))).unsqueeze(0).expand(dt_arr.shape[0], -1, -1)).unsqueeze(2).expand(-1, -1, self.cat_vocab_size, -1) *
                              x_onehot_cat.unsqueeze(3), dim=1)

        x_encoder_input = torch.cat([x_id_emb, torch.cat([x_cat_emb, x_dt_emb, x_amount_emb], dim=2)], dim=1)  # [batch_size, cat_vocab_size+1, emb_dim]

        x_encoder_output = self.transformer_encoder(x_encoder_input)[:, 1:, :] # [batch_size, cat_vocab_size, emb_dim]
        x_encoder_output = torch.mean(x_encoder_output, dim=2) # [batch_size, cat_vocab_size]
        return x_encoder_output


class TransformerVAE(nn.Module):
    def __init__(self, look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim, latent_dim, device):
        super(TransformerVAE, self).__init__()

        self.cat_vocab_size = cat_vocab_size
        self.latent_dim = latent_dim
        self.device = device

        self.transformer_encoder = TransformerNet(look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim)

        self.encoder_history = EncoderTemplate(cat_vocab_size, latent_dim, latent_dim)
        self.encoder_curr_labels = EncoderTemplate(cat_vocab_size, latent_dim, latent_dim)

        self.decoder_history = DecoderTemplate(latent_dim, latent_dim, cat_vocab_size)
        self.decoder_curr_labels = DecoderTemplate(latent_dim, latent_dim, cat_vocab_size)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar)
        eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1).to(self.device)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps

    def forward(self, cat_arr, dt_arr, amount_arr, id_arr, current_cat=None):
        x_history = self.transformer_encoder(cat_arr, dt_arr, amount_arr, id_arr)
        mu_history, logvar_history = self.encoder_history(x_history)
        z_history = self.reparameterize(mu_history, logvar_history) # [bacth_size, latent_dim]

        history_from_history = self.decoder_history(z_history)
        labels_from_history = self.decoder_curr_labels(z_history)

        if current_cat is not None: # [batch_size, max_cat_len]
            x_current_mask = torch.tensor(~(current_cat == self.cat_vocab_size), dtype=torch.int64).unsqueeze(2)  # [batch_size, max_cat_len, 1]
            x_current_onehot = torch.sum(one_hot(current_cat, num_classes=self.cat_vocab_size + 1) * x_current_mask, dim=1)[:, :-1].float() # [batch_size, cat_vocab_size]
            mu_curr_labels, logvar_curr_labels = self.encoder_curr_labels(x_current_onehot)
            z_curr_labels = self.reparameterize(mu_curr_labels, logvar_curr_labels) # [bacth_size, latent_dim]
            labels_from_labels = self.decoder_curr_labels(z_curr_labels)
            history_from_labels = self.decoder_history(z_curr_labels)
            return (x_history, x_current_onehot), \
                   [history_from_history, labels_from_history, labels_from_labels, history_from_labels], \
                   (mu_history, logvar_history), \
                   (mu_curr_labels, logvar_curr_labels)
        else:
            return labels_from_history


class TransformerVAELoss(nn.Module):
    def __init__(self, beta, gamma, delta):
        super(TransformerVAELoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.reconstruction_criterion = nn.MSELoss(size_average=False)

    def vae_loss(self, x_history, x_current_onehot, history_from_history, labels_from_labels,
                 mu_history, logvar_history, mu_curr_labels, logvar_curr_labels):
        reconstruction_loss = self.reconstruction_criterion(history_from_history, x_history) + \
                              self.reconstruction_criterion(labels_from_labels, x_current_onehot)
        kl_divergence = (0.5 * torch.sum(1 + logvar_curr_labels - mu_curr_labels.pow(2) - logvar_curr_labels.exp())) \
                        + (0.5 * torch.sum(1 + logvar_history - mu_history.pow(2) - logvar_history.exp()))
        return reconstruction_loss - self.beta * kl_divergence

    def cross_reconstruction_loss(self, x_history, x_current_onehot, labels_from_history, history_from_labels):
        cross_rec = self.reconstruction_criterion(history_from_labels, x_history) + \
                    self.reconstruction_criterion(labels_from_history, x_current_onehot)
        return cross_rec

    def distribution_alignment_loss(self, mu_history, logvar_history, mu_curr_labels, logvar_curr_labels):
        dist = torch.sqrt(torch.sum((mu_history - mu_curr_labels) ** 2, dim=1) + \
               torch.sum((torch.sqrt(logvar_history.exp()) - torch.sqrt(logvar_curr_labels.exp())) ** 2, dim=1))
        dist = dist.sum()
        return dist

    def forward(self, x_history, x_current_onehot,
                history_from_history, labels_from_history, labels_from_labels, history_from_labels,
                mu_history, logvar_history,
                mu_curr_labels, logvar_curr_labels):
        overall_loss = self.vae_loss(x_history, x_current_onehot, history_from_history, labels_from_labels,
                                     mu_history, logvar_history, mu_curr_labels, logvar_curr_labels) + \
                       self.gamma * self.cross_reconstruction_loss(x_history, x_current_onehot,
                                                                   labels_from_history, history_from_labels) + \
                       self.delta * self.distribution_alignment_loss(mu_history, logvar_history,
                                                                     mu_curr_labels, logvar_curr_labels)
        return overall_loss






