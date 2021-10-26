from data_preparation.dataset_preparation import OrderDataset
from sklearn.preprocessing import StandardScaler
from data_preparation.window_split import window_combination
import pandas as pd
import os

def prepare_target_regression(data_folder, train_file, test_file, look_back, init_data=True):
    """Target preparation for time regression problem

    Parameters
    ---
    init_data: bool, default=True
        Possibility to init new data or load from csv.

    """
    if init_data:
        train_data, test_data = OrderDataset(data_folder, train_file, test_file, look_back).prepare_data()
    else:
        train_data = pd.read_csv(os.path.join(data_folder, train_file))
        test_data = pd.read_csv(os.path.join(data_folder, test_file))
        for df in [train_data, test_data]:
            df.drop(['Unnamed: 0'], axis=1, inplace=True)
    ss_time = StandardScaler()
    train_time = ss_time.fit_transform(train_data['dt'].values.reshape(-1, 1))
    test_time = ss_time.transform(test_data['dt'].values.reshape(-1, 1))
    train_comb = window_combination(train_data, look_back)
    y_train = []
    y_train_scaled = []
    for i, (ref_points, pred_point) in enumerate(train_comb):
        y_train.append(train_data['dt'][pred_point])
        y_train_scaled.append(train_time[pred_point])
    test_comb = window_combination(test_data, look_back)
    y_test = []
    y_test_scaled = []
    for i, (ref_points, pred_point) in enumerate(test_comb):
        y_test.append(test_data['dt'][pred_point])
        y_test_scaled.append(test_time[pred_point])
    return (y_train, y_train_scaled, y_test, y_test_scaled, ss_time)