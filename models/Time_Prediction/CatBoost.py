from sklearn.metrics import r2_score
from data_preparation.dataset_preparation import OrderDataset
from catboost import CatBoostRegressor, Pool
import json
import os
from sacred import Experiment
import pandas as pd
import numpy as np


def catboost_model(data_folder, train_file, test_file, look_back, fix_material, current_info,
                   predicted_value, file_name):
    """CatBoost Regression

    data  from OrderDataset

    delete function with one-hot preprocessing
    made window combination,
        further in the loop we use it to combine the rows,
            making mini tables, which we put in pd.Series

    create class Poll, where we provide data and categorical columns

    """

    model_name = file_name  # 'CatBoost_for_time_simple_lookback_10'
    order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material, current_info,
                                 predicted_value)
    X_train, X_test = order_dataset.preprocess_dataframe()
    print('1')
    train_comb = order_dataset.window_combinations(X_train)
    test_comb = order_dataset.window_combinations(X_test)
    print('1')
    x_train = pd.DataFrame(columns=X_train.keys())
    for i, (ref_points) in enumerate(train_comb):
        df = X_train.loc[ref_points[0]]
        x_train = x_train.append(pd.Series(df.iat[0, 0], index=df.columns), ignore_index=True)
    print('1')
    x_test = pd.DataFrame(columns=X_test.keys())
    for i, (ref_points) in enumerate(test_comb):
        df = X_test.loc[ref_points[0]]
        x_test = x_test.append(pd.Series(df.iat[0, 0], index=df.columns), ignore_index=True)

    (y_train, y_train_scaled, y_test, y_test_scaled, ss_time) = order_dataset.prepare_target_regression()
    train_data = Pool(data=x_train,
                      label=y_train_scaled,
                      cat_features=['Ship.to', 'PLZ', 'Status', 'Material', 'MaterialGroup.1', 'MaterialGroup.2',
                                    'MaterialGroup.4'])

    cat_model = CatBoostRegressor(iterations=2000)
    print('1')
    cat_model.fit(train_data, verbose=False, plot=True)
    y_pred_scaled = cat_model.predict(x_test)
    r2_metric = r2_score(y_test, ss_time.inverse_transform(y_pred_scaled.reshape(-1, 1)))

    print(f'{model_name}, test_r2_score = {r2_metric}')
    os.makedirs('results/', exist_ok=True)
    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)

    y_pred_scaled = cat_model.predict(x_train)
    r2_metric = r2_score(y_train, ss_time.inverse_transform(y_pred_scaled.reshape(-1, 1)))

    print(f'{model_name}, train_r2_score = {r2_metric}')
