from sklearn.metrics import r2_score
from data_preparation.dataset_preparation import OrderDataset
from sklearn.linear_model import LinearRegression
import json
import os
from sacred import Experiment


ex = Experiment('LinearRegression')
# ex.add_config('configs/basic.json')
@ex.automain
def linreg_model(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value):
    model_name = 'LinearRegression'
    order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value)
    X_train, X_test = order_dataset.construct_features()
    (y_train, y_train_scaled, y_test, y_test_scaled, mms_target) = order_dataset.prepare_target_regression()

    model = LinearRegression()
    model.fit(X_train, y_train_scaled)
    y_pred_scaled = model.predict(X_test)
    r2_metric = r2_score(y_test, mms_target.inverse_transform(y_pred_scaled.reshape(-1, 1)))

    print(f'{model_name}, test_r2_score = {r2_metric}')
    os.makedirs(f'results/{model_name}/', exist_ok=True)
    with open(f'results/{model_name}/look_back_{look_back}_curr_{current_info}_fix_{fix_material}.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)