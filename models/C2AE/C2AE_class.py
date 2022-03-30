import sys
import utils
import numpy as np
import torch
from torch import nn


import torch
import torch.nn.functional as F


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


class ClassificationNet(nn.Module):
    def __init__(self, linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                 cat_vocab_size, id_vocab_size,
                 id_embedding_dim, linear_concat1_dim, linear_concat2_dim):
        super(ClassificationNet, self).__init__()

        self.relu = nn.ReLU()

        self.lstm_encoding_cat = LSTMEncoding(cat_embedding_dim, lstm_hidden_dim)

        self.linear_num_feat = nn.Linear(2, linear_num_feat_dim)
        self.lstm_encoding_num = LSTMEncoding(linear_num_feat_dim, lstm_hidden_dim)

        self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_size + 1, embedding_dim=cat_embedding_dim,
                                          padding_idx=cat_vocab_size)
        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=id_embedding_dim)

        self.linear_concat1 = nn.Linear(2 * lstm_hidden_dim + id_embedding_dim, linear_concat1_dim)
        self.linear_concat2 = nn.Linear(linear_concat1_dim, linear_concat2_dim)

        # self.linear_mat_num = nn.Linear(linear_concat2_dim, 13)
        self.linear_material = nn.Linear(linear_concat2_dim, cat_vocab_size)

    def forward(self, cat_arr, mask_cat, num_arr, id_arr):
        x_cat_emb = self.cat_embedding(cat_arr)  # [batch_size, look_back, max_day_len0, cat_embedding_dim]
        x_cat_emb = x_cat_emb * mask_cat.unsqueeze(3)
        x_cat_emb_sum = torch.sum(x_cat_emb, dim=2)  # [batch_size, look_back, cat_embedding_dim]

        x_encoding_cat = self.lstm_encoding_cat(x_cat_emb_sum)  # [batch_size, lstm_hidden_dim]

        x_num1 = self.linear_num_feat(num_arr)  # [batch_size, look_back, linear_num_feat_dim]
        x_encoding_num = self.lstm_encoding_num(x_num1)  # [batch_size, lstm_hidden_dim]

        x_id_emb = self.id_embedding(id_arr).squeeze(1)  # [batch_size, id_embedding_dim]
        x_concat = torch.cat((x_encoding_cat, x_encoding_num, x_id_emb),
                             dim=1)  # [batch_size, 2*lstm_hidden_dim+id_embedding_dim]

        x1 = self.linear_concat1(x_concat)
        x1 = self.relu(x1)
        x2 = self.linear_concat2(x1)
        x2 = self.relu(x2)
        # x_mat_num = self.linear_mat_num(x2)  # [batch_size, 13]
        x_material = self.linear_material(x2)  # [batch_size, cat_vocab_size]

        return x_material


class Fd(torch.nn.Module):
    """
    Simple fully connected decoder network, part of C2AE.
    """
    def __init__(self, in_dim, H, out_dim, fin_act=None):
        super(Fd, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.fc2 = torch.nn.Linear(H, out_dim)
        self.fin_act = fin_act

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return self.fin_act(x) if self.fin_act else x


class Fx(torch.nn.Module):
    """
    Simple fully connected network for input x, part of C2AE.
    """
    def __init__(self, in_dim, H1, H2, out_dim):
        super(Fx, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.fc3 = torch.nn.Linear(H2, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x


class Fe(torch.nn.Module):
    """
    Simple fully connected encoder network, part of C2AE.
    """
    def __init__(self, in_dim, H, out_dim):
        super(Fe, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, H)
        self.fc2 = torch.nn.Linear(H, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x


class C2AE(torch.nn.Module):

    def __init__(self, Fx, Fe, Fd, beta=1, alpha=.5, emb_lambda=.5, latent_dim=6,
                 device=None):
        super(C2AE, self).__init__()
        # Define main network components.
        # Encodes x into latent space. X ~ z
        self.Fx = Fx
        # Encodes y into latent space. Y ~ z
        self.Fe = Fe
        # Decodes latent space into Y. z ~ Y
        self.Fd = Fd

        # Hyperparam used to set tradeoff between latent loss, and corr loss.
        self.alpha = alpha
        self.beta = beta
        # Lagrange to use in embedding loss.
        self.emb_lambda = emb_lambda
        self.latent_I = torch.eye(latent_dim).to(device)

    def forward(self, x, y=None):
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
            # Calculate feature, and label latent representations.
            fx_x = self.Fx(x)
            fe_y = self.Fe(y)
            # Calculate decoded latent representation.
            fd_z = self.Fd(fe_y)
            return fx_x, fe_y, fd_z
        else:
            # If evaluating just send through encoder and decoder.
            return self.predict(x)

    def _predict(self, y):
        """This method predicts with the y encoded latent space.
        """
        return self.Fd(self.Fe(y))

    def predict(self, x):
        """This method predicts with the x encoded latent space.
        """
        return self.Fd(self.Fx(x))

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
        # ********** Version 1: Implemented as suggested in Yeh et al. # **********
        # Initial condition.
        c1 = fx_x - fe_y
        # Here to help hold constraint of FxFx^2 = FyFy^2 = I.
        c2 = fx_x.T @ fx_x - self.latent_I
        c3 = fe_y.T @ fe_y - self.latent_I
        # Combine loss components as specified in Yah et al.
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
        c_loss = self.corr_loss(fd_z, y)
        return l_loss, c_loss


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_cls, path, *args, **kwargs):
    model = model_cls(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def eval_metrics(mod, metrics, datasets, device, apply_sig=False):
    res_dict = {}
    for ix, dataset in enumerate(datasets):
        mod.eval()
        x = dataset.tensors[0].to(device).float()
        # Make predictions.
        preds = mod(x)
        # Convert them to binary multilabels.
        if apply_sig:
            y_pred = torch.round(torch.sigmoid(preds)).cpu().detach().numpy()
        else:
            y_pred = torch.round(preds).cpu().detach().numpy()
        y_true = dataset.tensors[1].cpu().detach().numpy()
        # Calculate metric.
        res_dict[f'dataset_{ix}'] = {metric.__name__: metric(y_true, y_pred) for metric in metrics}
    return res_dict

"""-----------------"""

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

class Network(object):
    def __init__(self, config, summarizer):
        tf.set_random_seed(1234)
        self.summarizer = summarizer
        self.config = config
        self.Wx1, self.Wx2, self.Wx3, self.bx1, self.bx2, self.bx3 = self.init_Fx_variables()
        self.We1, self.We2, self.be1, self.be2 = self.init_Fe_variables()
        self.Wd1, self.Wd2, self.bd1, self.bd2 = self.init_Fd_variables()

    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape,  initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def init_Fx_variables(self):
        W1 = self.weight_variable([self.config.features_dim, self.config.solver.hidden_dim], "weight_x1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.hidden_dim], "weight_x2")
        W3 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_x3")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_x1")
        b2 = self.bias_variable([self.config.solver.hidden_dim], "bias_x2")
        b3 = self.bias_variable([self.config.solver.latent_embedding_dim], "bias_x3")
        return W1, W2, W3, b1, b2, b3

    def init_Fe_variables(self):
        W1 = self.weight_variable([self.config.labels_dim, self.config.solver.hidden_dim], "weight_e1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_e2")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_e1")
        b2 = self.bias_variable([self.config.solver.latent_embedding_dim], "bias_e2")
        return W1, W2, b1, b2

    def init_Fd_variables(self):
        W1 = self.weight_variable([self.config.solver.latent_embedding_dim, self.config.solver.hidden_dim], "weight_d1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.labels_dim], "weight_d2")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_d1")
        b2 = self.bias_variable([self.config.labels_dim], "bias_d2")
        return W1, W2, b1, b2

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))

    def Fx(self, X, keep_prob):
        hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(X, self.Wx1) + self.bx1), keep_prob)
        hidden2 = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden1, self.Wx2) + self.bx2), keep_prob)
        hidden3 = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden2, self.Wx3) + self.bx3), keep_prob)
        return hidden3

    def Fe(self, Y, keep_prob):
        hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(Y, self.We1)) + self.be1, keep_prob)
        pred = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden1, self.We2) + self.be2), keep_prob)
        return pred

    def Fd(self, input, keep_prob):
        hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(input, self.Wd1) + self.bd1), keep_prob)
        y_pred = tf.matmul(hidden1, self.Wd2) + self.bd2
        return y_pred

    def prediction(self, X, keep_prob):
        Fx = self.Fx(X, keep_prob)
        return self.Fd(Fx, keep_prob)

    def embedding_loss(self, Fx, Fe):
        Ix, Ie = tf.eye(tf.shape(Fx)[0]), tf.eye(tf.shape(Fe)[0])
        C1, C2, C3 = tf.abs(Fx - Fe), tf.matmul(Fx, tf.transpose(Fx)) - Ix, tf.matmul(Fe, tf.transpose(Fe)) - Ie
        return tf.reduce_mean(tf.square(Fx - Fe)) #tf.trace(tf.matmul(C1, tf.transpose(C1))) + self.config.solver.lagrange_const * tf.trace(tf.matmul(C2, tf.transpose(C2))) + self.config.solver.lagrange_const * tf.trace(tf.matmul(C3, tf.transpose(C3)))

    # My blood, sweat and tears were also embedded into the emebedding.
    def output_loss(self, predictions, labels):
        Ei = 0.0
        i, cond = 0, 1
        while cond == 1:
            cond = tf.cond(i >= tf.shape(labels)[0] - 1, lambda: 0, lambda: 1)
            prediction_, Y_ = tf.slice(predictions, [i, 0], [1, self.config.labels_dim]), tf.slice(labels, [i, 0], [1, self.config.labels_dim])
            zero, one = tf.constant(0, dtype=tf.float32), tf.constant(1, dtype=tf.float32)
            ones, zeros = tf.gather_nd(prediction_, tf.where(tf.equal(Y_, one))), tf.gather_nd(prediction_, tf.where(tf.equal(Y_, zero)))
            y1 = tf.reduce_sum(Y_)
            y0 = Y_.get_shape().as_list()[1] - y1
            temp = (1/y1 * y0) * tf.reduce_sum(tf.exp(-(tf.reduce_sum(ones) / tf.cast(tf.shape(ones)[0], tf.float32) - tf.reduce_sum(zeros) / tf.cast(tf.shape(zeros)[0], tf.float32))))
            Ei += tf.cond(tf.logical_or(tf.is_inf(temp), tf.is_nan(temp)), lambda : tf.constant(0.0), lambda : temp)
            i += 1
        return Ei

    def cross_loss(self, features, labels, keep_prob):
        predictions = self.prediction(features, keep_prob)
        Fx = self.Fx(features, keep_prob)
        Fe = self.Fe(labels, keep_prob)
        cross_loss = tf.add(tf.log(1e-10 + tf.nn.sigmoid(predictions)) * labels, tf.log(1e-10 + (1 - tf.nn.sigmoid(predictions))) * (1 - labels))
        cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(cross_loss, 1))
        return cross_entropy_label

    def loss(self, features, labels, keep_prob):
        lamda = 0.00
        prediction = tf.nn.sigmoid(self.prediction(features, keep_prob))
        Fx = self.Fx(features, keep_prob)
        Fe = self.Fe(labels, keep_prob)
        l2_norm = tf.reduce_sum(tf.square(self.Wx1)) + tf.reduce_sum(tf.square(self.Wx2)) + tf.reduce_sum(tf.square(self.Wx3)) + tf.reduce_sum(tf.square(self.We1)) + tf.reduce_sum(tf.square(self.We2)) + tf.reduce_sum(tf.square(self.Wd1)) + tf.reduce_sum(tf.square(self.Wd2))
        return self.embedding_loss(Fx, Fe) + self.config.solver.alpha * self.output_loss(prediction, labels) + lamda * l2_norm # self.cross_loss(features, labels, keep_prob)

    def train_step(self, loss):
        optimizer = self.config.solver.optimizer
        return optimizer(self.config.solver.learning_rate).minimize(loss)