from data_preparation.dataset_preparation import OrderDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import numpy as np
import pandas as pd
import os
from data_preparation.window_split import window_combination
from data_preparation.features_description import constant_features_one_hot, changing_features_one_hot, \
                                                 features_for_vectorizer, numerical_features


def construct_features(data_folder, train_file, test_file, look_back, init_data=True):
    """Construct features for time regression problem.

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


    # OneHotEncoding for categorical features
    ohe_constant = OneHotEncoder()
    train_ohe_constant = ohe_constant.fit_transform(train_data[constant_features_one_hot]).toarray()
    test_ohe_constant = ohe_constant.transform(test_data[constant_features_one_hot]).toarray()

    ohe_changing = OneHotEncoder()
    train_ohe_changing = ohe_changing.fit_transform(train_data[changing_features_one_hot]).toarray()
    test_ohe_changing = ohe_changing.transform(test_data[changing_features_one_hot]).toarray()

    # Bag of Words processing
    train_sparse_matrices = []
    test_sparse_matrices = []
    for feature_name in features_for_vectorizer:
        vocab = OrderDataset(data_folder, train_file, test_file, look_back).get_vocab(feature_name, init_data=init_data)
        vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False)
        train_matr = vectorizer.fit_transform(train_data[feature_name].values)
        test_matr = vectorizer.transform(test_data[feature_name].values)
        train_sparse_matrices.append(train_matr)
        test_sparse_matrices.append(test_matr)
    train_vectorized_features = sparse.hstack(train_sparse_matrices, format='csr').toarray()
    test_vectorized_features = sparse.hstack(test_sparse_matrices, format='csr').toarray()

    # Scaling
    ss = StandardScaler()
    train_scaled_features = ss.fit_transform(np.concatenate([train_vectorized_features, train_data[numerical_features]], axis=1))
    test_scaled_features = ss.transform(np.concatenate([test_vectorized_features, test_data[numerical_features]], axis=1))

    # Construct features for train/test data
    train_comb = window_combination(train_data, look_back)
    test_comb = window_combination(test_data, look_back)
    train_resulting_shape = (len(train_comb),
                             train_ohe_constant.shape[1] +
                             (train_ohe_changing.shape[1] + train_scaled_features.shape[1]) * look_back)
    test_resulting_shape = (len(test_comb),
                            test_ohe_constant.shape[1] +
                            (test_ohe_changing.shape[1] + test_scaled_features.shape[1]) * look_back)
    X_train = np.empty(train_resulting_shape)
    for i, (ref_points, pred_point) in enumerate(train_comb):
        X_train[i, :] = np.concatenate([train_ohe_constant[ref_points[0], :],
                                        train_ohe_changing[ref_points, :].flatten(),
                                        train_scaled_features[ref_points, :].flatten()], axis=0)
    X_test = np.empty(test_resulting_shape)
    for i, (ref_points, pred_point) in enumerate(test_comb):
        X_test[i, :] = np.concatenate([test_ohe_constant[ref_points[0], :],
                                       test_ohe_changing[ref_points, :].flatten(),
                                       test_scaled_features[ref_points, :].flatten()], axis=0)

    return X_train, X_test
