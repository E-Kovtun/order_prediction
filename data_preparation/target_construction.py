from data_preparation.dataset_preparation import OrderDataset
from sklearn.preprocessing import StandardScaler
from data_preparation.window_split import window_combination


def prepare_target_regression(data_folder, train_file, test_file, look_back):
    train_data, test_data = OrderDataset(data_folder, train_file, test_file, look_back).prepare_data()
    ss_amount = StandardScaler()
    train_amounts = ss_amount.fit_transform(train_data['Amount_HL'].values.reshape(-1, 1))
    test_amounts = ss_amount.transform(test_data['Amount_HL'].values.reshape(-1, 1))
    train_comb = window_combination(train_data, look_back)
    y_train = []
    y_train_scaled = []
    for i, (ref_points, pred_point) in enumerate(train_comb):
        y_train.append(train_data['Amount_HL'][pred_point])
        y_train_scaled.append(train_amounts[pred_point])
    test_comb = window_combination(test_data, look_back)
    y_test = []
    y_test_scaled = []
    for i, (ref_points, pred_point) in enumerate(test_comb):
        y_test.append(test_data['Amount_HL'][pred_point])
        y_test_scaled.append(test_amounts[pred_point])
    return (y_train, y_train_scaled, y_test, y_test_scaled, ss_amount)
