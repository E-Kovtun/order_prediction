from sklearn.metrics import r2_score
from data_preparation.dataset_preparation import OrderDataset
from sklearn.linear_model import LinearRegression
import json
from sacred import Experiment
import os


def linreg_model(data_folder, train_file, test_file, look_back, fix_material, current_info,
                   predicted_value, file_name, load_data=False):
    if load_data:
        model_name = file_name  # 'LREG_for_time_simple_lookback_4'
    else:
        model_name = file_name  # 'LREG_for_time_with_difficult_restorant_lookback_10'

    order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material, current_info,
                                 predicted_value, load_data=load_data)
    X_train, X_test = order_dataset.construct_features()
    (y_train, y_train_scaled, y_test, y_test_scaled, ss_time) = order_dataset.prepare_target_regression()

    model = LinearRegression()
    model.fit(X_train, y_train_scaled)
    y_pred_scaled = model.predict(X_test)
    r2_metric = r2_score(y_test, ss_time.inverse_transform(y_pred_scaled.reshape(-1, 1)))

    print(f'{model_name}, test_r2_score = {r2_metric}')
    os.makedirs('results/', exist_ok=True)
    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)