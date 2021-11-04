from sklearn.metrics import r2_score
from data_preparation.features_construction import construct_features
from data_preparation.target_time import prepare_target_regression
from catboost import CatBoostRegressor, Pool
from data_preparation.window_split import window_combination
from data_preparation.dataset_preparation import OrderDataset
import json
import pandas as pd
import numpy as np

from sacred import Experiment
import os


def catboost_model(data_folder, train_file, test_file, look_back, init_data=True):
    """CatBoost Regression

    data  from OrderDataset

    delete function with one-hot preprocessing
    made window combination,
        further in the loop we use it to combine the rows,
            making mini tables, which we put in pd.Series

    create class Poll, where we provide data and categorical columns

    """
    if init_data:
        model_name = 'CatBoost_for_time_simple_lookback_1'
        X_train, X_test = OrderDataset(data_folder, train_file, test_file, look_back).prepare_data()
    else:
        model_name = 'CatBoost_for_time_with_difficult_restorant_lookback_1'
        X_train = pd.read_csv(os.path.join(data_folder, train_file))
        X_test = pd.read_csv(os.path.join(data_folder, test_file))
        for df in [X_train, X_test]:
            df.drop(['Unnamed: 0'], axis=1, inplace=True)

    train_comb = window_combination(X_train, look_back)
    test_comb = window_combination(X_test, look_back)

    x_train = pd.DataFrame(columns=X_train.keys())
    for i, (ref_points) in enumerate(train_comb):
        df = X_train.loc[ref_points[0]]
        x_train = x_train.append(pd.Series(df.iat[0, 0], index=df.columns), ignore_index=True)

    x_test = pd.DataFrame(columns=X_test.keys())
    for i, (ref_points) in enumerate(test_comb):
        df = X_test.loc[ref_points[0]]
        x_test = x_test.append(pd.Series(df.iat[0, 0], index=df.columns), ignore_index=True)

    (y_train, y_train_scaled, y_test, y_test_scaled, ss_time) = prepare_target_regression(data_folder,
                                                                                          train_file, test_file,
                                                                                          look_back,
                                                                                          init_data=init_data)
    train_data = Pool(data=x_train,
                      label=y_train_scaled,
                      cat_features=['Ship.to', 'PLZ', 'Status', 'Material', 'MaterialGroup.1', 'MaterialGroup.2', 'MaterialGroup.4'])

    cat_model = CatBoostRegressor(iterations=2000)
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