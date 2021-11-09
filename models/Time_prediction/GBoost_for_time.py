from sklearn.metrics import r2_score
from data_preparation.features_construction import construct_features
from data_preparation.target_time import prepare_target_regression
from xgboost import XGBRegressor
import json
from sacred import Experiment
import os


def xgboost_model(data_folder, train_file, test_file, look_back, init_data=True):
    if init_data:
        model_name = 'XGB_for_time_simple_lookback_10'
    else:
        model_name = 'XGB_for_time_with_difficult_restorant_lookback_10'

    X_train, X_test = construct_features(data_folder, train_file, test_file, look_back, init_data=init_data)
    (y_train, y_train_scaled, y_test, y_test_scaled, ss_time) = prepare_target_regression(data_folder,
                                                                                          train_file, test_file,
                                                                                          look_back,
                                                                                          init_data=init_data)

    model = XGBRegressor(random_state=10)
    model.fit(X_train, y_train_scaled)
    y_pred_scaled = model.predict(X_test)
    r2_metric = r2_score(y_test, ss_time.inverse_transform(y_pred_scaled.reshape(-1, 1)))

    print(f'{model_name}, test_r2_score = {r2_metric}')
    os.makedirs('results/', exist_ok=True)
    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)