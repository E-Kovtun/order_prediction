from sklearn.metrics import r2_score, mean_squared_error
from data_preparation.dataset_preparation import OrderDataset
from sklearn.ensemble import GradientBoostingClassifier
import json
import os

def gboost_model_class(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value,
                  file_name, classif=None):
    model_name = file_name
    order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material, current_info,
                                 predicted_value, classif=classif)
    X_train, X_test = order_dataset.construct_features()
    (y_train, y_train_scaled, y_test, y_test_scaled, mms_target) = order_dataset.prepare_target_regression()

    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.01, max_depth=5, random_state=10)
    print("X : ", X_train)
    print("y : ", y_train_scaled)
    print("X_train[0] : ", X_train[0])
    model.fit(X_train, y_train_scaled)
    y_pred_scaled = model.predict(X_test)
    y_pred_scaled_1 = model.predict(X_train)
    r2_metric = r2_score(y_test, mms_target.inverse_transform(y_pred_scaled.reshape(-1, 1)))
    r2_metric_1 = r2_score(y_train, mms_target.inverse_transform(y_pred_scaled_1.reshape(-1, 1)))
    mse1 = mean_squared_error(y_test, mms_target.inverse_transform(y_pred_scaled.reshape(-1, 1)))
    mse2 = mean_squared_error(y_train, mms_target.inverse_transform(y_pred_scaled_1.reshape(-1, 1)))

    print(f'{model_name}, test_r2_score = {r2_metric}')
    print(f'{model_name}, train_r2_score = {r2_metric_1}')
    print(f'{model_name}, test_mse_score = {mse1}')
    print(f'{model_name}, train_mse_score = {mse2}')
    os.makedirs('results/', exist_ok=True)
    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)
