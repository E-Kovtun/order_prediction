import torch
import torch.nn.functional as F
import pandas as pd

class Fd(torch.nn.Module):
    """
    Simple fully connected decoder network, part of C2AE.
    """
    def __init__(self, in_dim, H, out_dim, fin_act=None):
        super(Fd, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.bn = torch.nn.BatchNorm1d(H)
        self.fc2 = torch.nn.Linear(H, out_dim)
        self.fin_act = fin_act

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return self.fin_act(x) if self.fin_act else x


class Fx(torch.nn.Module):
    """
    Simple fully connected network for input x, part of C2AE.
    """
    def __init__(self, in_dim, H1, H2, out_dim):
        super(Fx, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H1)
        self.bn1 = torch.nn.BatchNorm1d(H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.bn2 = torch.nn.BatchNorm1d(H2)
        self.fc3 = torch.nn.Linear(H2, out_dim)
        self.bn3 = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        return x


class Fe(torch.nn.Module):
    """
    Simple fully connected encoder network, part of C2AE.
    """
    def __init__(self, in_dim, H, out_dim):
        super(Fe, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.bn1 = torch.nn.BatchNorm1d(H)
        self.fc2 = torch.nn.Linear(H, out_dim)
        self.bn2 = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        return x


class C2AE(torch.nn.Module):

    def __init__(self, transformer, fx, fe, fd, beta=1, alpha=.5, emb_lambda=.5, latent_dim=6,
                 device=None):
        super().__init__()
        # Define main network components.
        self.transformer = transformer
        # Encodes x into latent space. X ~ z
        self.fx = fx
        # Encodes y into latent space. Y ~ z
        self.fe = fe
        # Decodes latent space into Y. z ~ Y
        self.fd = fd

        # Hyperparam used to set tradeoff between latent loss, and corr loss.
        self.alpha = alpha
        self.beta = beta
        # Lagrange to use in embedding loss.
        self.emb_lambda = emb_lambda
        self.latent_I = torch.eye(latent_dim).to(device)

    def forward(self, batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr, batch_current_cat=None,
                batch_onehot_current_cat=None, current_minus1_cat=None):
        """
        Forward pass of C2AE model.
        Training:
            Runs feature vector x through Fx, then encodes y through Fe and
            computes latent loss (MSE between feature maps). Then z = Fe(y) is
            sent through decoder network in which it tries to satisfy
            correlation equation.
        Testing:
            Simply runs feature vector x through autoencoder. Fd(Fx(x))
            This will result in a logits vec of multilabel preds.
        """
        if self.training:
            x = self.transformer(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr) #, batch_current_cat)
            
            # Calculate feature, and label latent representations.
            fx_x = self.fx(x)
            fe_y = self.fe(batch_onehot_current_cat.float())
            #fe_y_t = self.fe(current_minus1_cat.float())#current_minus1_cat.float())
            # Calculate decoded latent representation.
            fd_z = self.fd(fe_y)
            return fx_x, fe_y, fd_z #, fe_y_t
        else:
            x = self.transformer(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)
            # If evaluating just send through encoder and decoder.
            return self.predict(x)
        # else:
        #     x = self.classification(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)
        #     # If evaluating just send through encoder and decoder.
        #     fx_x = self.fx(x)
        #     fe_y = self.fe(current_minus1_cat.float())
        #     # Calculate decoded latent representation.
        #     fd_z = self.fd(0.7*fe_y+0.3*fx_x)
        #     return fd_z
        # else:
        #     x = self.classification(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)
        #     # If evaluating just send through encoder and decoder.
        #     return self.predict(x)

    def _predict(self, y):
        """This method predicts with the y encoded latent space.
        """
        return self.fd(self.fe(y))

    def predict(self, x):
        """This method predicts with the x encoded latent space.
        """
        return self.fd(self.fx(x))

    def corr_loss(self, preds, y):
        """This method compares the predicted probabilitie class distribution
        from the decoder, with the true y labels.
        """
        # Generate masks for [0,1] elements.
        ones = (y == 1)
        zeros = (y == 0)
        # Use broadcasting to apply logical and between mask arrays.
        # This will only indicate locations where both masks are 1.
        # THis corresponds to set we are enumerating in eq (3) in Yah et al.
        ix_matrix = ones[:, :, None] & zeros[:, None, :]
        # Use same broadcasting logic to generate exponetial differences.
        # This like the above broadcast will do so between all pairs of points
        # for every datapoint.
        diff_matrix = torch.exp(-(preds[:, :, None] - preds[:, None, :]))
        # This will sum all contributes to loss for each datapoint.
        losses = torch.flatten(diff_matrix * ix_matrix, start_dim=1).sum(dim=1)
        # Normalize each loss add small epsilon incase 0.
        losses /= (ones.sum(dim=1) * zeros.sum(dim=1) + 1e-4)
        # Replace inf, and nans with 0.
        losses[losses == float('Inf')] = 0
        losses[torch.isnan(losses)] = 0
        # Combine all losses to retrieve final loss.
        return losses.sum()

    def latent_loss(self, fx_x, fe_y):
        """
        Loss between latent space generated from fx, and fe.
        ||Fx(x) - Fe(y)||^2 s.t. FxFx^2 = FyFy^2 = I
        First version contains decomposition of loss function, making use of
        lagrange multiplier to account for constraint.
        Second version just calculates the mean squared error.
        """
        # ********** Version 1 # **********
        # Initial condition.
        c1 = fx_x - fe_y
        # Here to help hold constraint of FxFx^2 = FyFy^2 = I.
        c2 = fx_x.T @ fx_x - self.latent_I
        c3 = fe_y.T @ fe_y - self.latent_I
        # Combine loss components.
        latent_loss = torch.trace(
            c1 @ c1.T) + self.emb_lambda * torch.trace(c2 @ c2.T + c3 @ c3.T)
        # ********** Version 2: Ignore constraint **********
        # latent_loss = torch.mean((fx_x - fe_y)**2)
        return latent_loss

    def losses(self, fx_x, fe_y, fd_z, y):
        """This method calculates the main loss functions required
        when composing the loss function.
        """
        l_loss = self.latent_loss(fx_x, fe_y)
        #l_loss_t = self.latent_loss(fx_x, fe_y_t)
        c_loss = self.corr_loss(fd_z, y)
        return l_loss, c_loss #, l_loss_t
