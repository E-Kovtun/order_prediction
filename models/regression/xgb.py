from sklearn.metrics import r2_score
from data_preparation.features_construction import construct_features
from data_preparation.target_construction import prepare_target_regression
from xgboost import XGBRegressor
import json
from sacred import Experiment
import os


ex = Experiment('XGBRegressor')
ex.add_config('configs/basic.json')
@ex.automain
def xgboost_model(data_folder, train_file, test_file, look_back):
    model_name = 'XGBRegressor'
    X_train, X_test = construct_features(data_folder, train_file, test_file, look_back)
    (y_train, y_train_scaled, y_test, y_test_scaled, ss_amount) = prepare_target_regression(data_folder, train_file, test_file, look_back)

    model = XGBRegressor(random_state=10)
    model.fit(X_train, y_train_scaled)
    y_pred_scaled = model.predict(X_test)
    r2_metric = r2_score(y_test, ss_amount.inverse_transform(y_pred_scaled.reshape(-1, 1)))

    print(f'{model_name}, test_r2_score = {r2_metric}')
    os.makedirs('results/', exist_ok=True)
    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)
