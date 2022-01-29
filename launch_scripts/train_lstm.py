import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
sys.path.append("Repository")

from models.Time_Prediction.LSTM import RegressionNet
from data_preparation.data_reader import OrderReader
from data_preparation.dataset_preparation import OrderDataset
from sklearn.metrics import r2_score
import numpy as np

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    num_epochs = 100

    data_folder = ""
    train_file = "df_beer_train.csv"
    test_file = "df_beer_test.csv"
    look_back = 1
    fix_material = True
    current_info = False
    predicted_value = "time"  # "amount"

    const_features = ['Ship.to', 'PLZ', 'Material']

    lstm_input_dim = 2
    lstm_hidden_dim = 128
    linear_encoding1_dim = 64
    linear_encoding2_dim = 32
    order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value)
    vocab_size0 = len(order_dataset.get_vocab(const_features[0]))
    vocab_size1 = len(order_dataset.get_vocab(const_features[1]))
    vocab_size2 = len(order_dataset.get_vocab(const_features[2]))
    const_embedding_dim = 256
    linear_embedding1_dim = 128
    linear_embedding2_dim = 64
    linear_concat1 = 32

    net = RegressionNet(lstm_input_dim, lstm_hidden_dim, linear_encoding1_dim, linear_encoding2_dim,
                        vocab_size0, vocab_size1, vocab_size2, const_embedding_dim,
                        linear_embedding1_dim, linear_embedding2_dim,
                        linear_concat1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    regr_loss = nn.MSELoss()

    print('Start preparing data')
    train_dataset = OrderReader(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value, 'train')
    test_dataset = OrderReader(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value, 'test')
    print('Finish preparing data')

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    for epoch in range(1, num_epochs+1):
        net.train(True)
        epoch_train_loss = 0
        print('Training...')
        for batch_ind, (batch_const_feat, batch_changing_feat, batch_target) in enumerate(train_dataloader):
            batch_const_feat, batch_changing_feat, batch_target = \
            batch_const_feat.to(device), batch_changing_feat.to(device), batch_target.to(device)
            optimizer.zero_grad()
            predicted_amounts = net(batch_changing_feat.float(), batch_const_feat.int())
            loss = regr_loss(predicted_amounts, batch_target.float())
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')
        net.train(False)
        test_predicted_amounts = []
        test_gt_amounts = []

        for batch_ind, (batch_const_feat, batch_changing_feat, batch_target) in enumerate(test_dataloader):
            batch_const_feat, batch_changing_feat, batch_target = \
            batch_const_feat.to(device), batch_changing_feat.to(device), batch_target.to(device)
            test_predicted_amounts.extend(net(batch_changing_feat.float(), batch_const_feat.int()).squeeze().detach().cpu().numpy().tolist())
            test_gt_amounts.extend(batch_target.squeeze().detach().cpu().numpy().tolist())

        r2_metric_test = r2_score(test_dataset.mms_amount.inverse_transform(np.array(test_gt_amounts, dtype=np.float64).reshape(-1, 1)),
                                  test_dataset.mms_amount.inverse_transform(np.array(test_predicted_amounts, dtype=np.float64).reshape(-1, 1)))
        print(f'Epoch {epoch}/{num_epochs} || Test r2_score {r2_metric_test}')

if __name__ == "__main__":
    train()