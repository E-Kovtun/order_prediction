import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from utils.utils import get_vocab, get_fitted_scaler


class OrderDataset:
    def __init__(self, data_folder, train_file, test_file, valid_file, look_back):
        self.data_folder = data_folder
        self.train_file = train_file
        self.test_file = test_file
        self.valid_file = valid_file
        self.look_back = look_back

        # final columns: id, categorical, amount, date, dt
        self.id = 'Ship.to'
        self.categorical = 'Material'
        self.amount = 'Amount_HL'
        self.date = 'Delivery_Date_week' # should be in the format YEAR-MONTH-DAY
        self.dt = 'dt'

        self.target = 'Amount_HL'

    def prepare_features(self, df, categorical_vocab, id_vocab):
        """
        Encode categorical feature with OrdinalEncoder.
        Process date column.
        Remove unnecessary columns.
        """
        df_upd = df[[self.id, self.categorical, self.amount, self.date]]
        df_upd[self.date] = pd.to_datetime(df_upd[self.date])

        cat_encoder = OrdinalEncoder(categories=categorical_vocab, dtype=np.int64)
        df_upd[self.categorical] = cat_encoder.fit_transform(df_upd[self.categorical].values.reshape(-1, 1))
        id_encoder = OrdinalEncoder(categories=id_vocab, dtype=np.int64)
        df_upd[self.id] = id_encoder.fit_transform(df_upd[self.id].values.reshape(-1, 1))
        return df_upd

    def group_rows(self, df):
        """
        Rows, which relate to the same day, are combined into one row.
        """
        df_copy = df.copy(deep=True)
        df_grouped = df_copy.sort_values([self.categorical]).groupby([self.id, self.date]).agg({self.categorical: lambda x: list(x),
                                                                                                self.amount: lambda x: list(x)}).reset_index()
        return df_grouped

    def add_time_difference(self, df):
        """
        Add column 'dt' with time difference between orders.
        """
        all_differences = []
        interm_df = df.groupby(self.id)[self.date].apply(lambda x: x.diff())
        for ind in interm_df.index:
            corr_diff = np.nan_to_num(interm_df.iloc[ind].days).tolist()
            all_differences.extend(corr_diff if type(corr_diff) == list else [corr_diff])
        df.insert(2, self.dt, all_differences)

    def preprocess_dataframe(self):
        """
        Combine several steps of dataset preprocessing: encoding, scaling, grouping rows, adding 'dt' feature.
        """

        train = pd.read_csv(os.path.join(self.data_folder, self.train_file))
        test = pd.read_csv(os.path.join(self.data_folder, self.test_file))
        valid = pd.read_csv(os.path.join(self.data_folder, self.valid_file))

        categorical_vocab = get_vocab(train, test, valid, self.categorical)
        cat_vocab_size = categorical_vocab.shape[1]
        id_vocab = get_vocab(train, test, valid, self.id)
        id_vocab_size = id_vocab.shape[1]
        datasets = [train, test, valid]
        processed_datasets = []
        mms_amount = get_fitted_scaler(train, test, valid, self.amount)
        for df in datasets:
            prepared_df = self.prepare_features(df, categorical_vocab, id_vocab)
            prepared_df[self.amount] = mms_amount.transform(prepared_df[self.amount].values.reshape(-1, 1))
            grouped_df = self.group_rows(prepared_df)
            self.add_time_difference(grouped_df)
            processed_datasets.append(grouped_df)

        mms_dt = get_fitted_scaler(*processed_datasets, self.dt)
        for df in processed_datasets:
            df[self.dt] = mms_dt.transform(df[self.dt].values.reshape(-1, 1))

        if self.target == self.amount:
            return processed_datasets, mms_amount, cat_vocab_size, id_vocab_size
        if self.target == self.dt:
            return processed_datasets, mms_dt, cat_vocab_size, id_vocab_size

    def window_combinations(self, df):
        """
        Specify indices of rows which we use as inputs and index of the row for which we make the prediction.
        """
        chunks_relevant_indices = []
        for num_id in df[self.id].unique():
            chunks_relevant_indices.append(df.loc[df[self.id] == num_id].index)

        all_combinations = []
        for relevant_indices in chunks_relevant_indices:
            for i in range(len(relevant_indices) - self.look_back):
                current_group = relevant_indices[i:i + self.look_back].tolist()
                index_to_pred = relevant_indices[i + self.look_back]
                all_combinations.append((current_group, index_to_pred))
        return all_combinations

# if __name__ == '__main__':
#     [train_f, test_f, valid_f], mms_amount = OrderDataset('../initial_data/', 'df_beer_train_nn.csv', 'df_beer_test.csv', 'df_beer_valid_nn.csv', 3).preprocess_dataframe()
#     print(train_f[['dt', 'Amount_HL']])

    # def construct_features(self):
    #     """ Construct features for Machine Learning algorithms.
    #     """
    #     train_data, test_data = self.preprocess_dataframe()
    #
    #     ohe_constant = OneHotEncoder()
    #     train_ohe_constant = ohe_constant.fit_transform(train_data[self.constant_features_one_hot]).toarray()
    #     test_ohe_constant = ohe_constant.transform(test_data[self.constant_features_one_hot]).toarray()
    #
    #     ohe_changing = OneHotEncoder()
    #     train_ohe_changing = ohe_changing.fit_transform(train_data[self.changing_features_one_hot]).toarray()
    #     test_ohe_changing = ohe_changing.transform(test_data[self.changing_features_one_hot]).toarray()
    #
    #     mms = MinMaxScaler()
    #     train_numerical_features = mms.fit_transform(train_data[self.numerical_features])
    #     test_numerical_features = mms.transform(test_data[self.numerical_features])
    #     train_vectorized_features = np.empty((train_data.shape[0], 0))
    #     test_vectorized_features = np.empty((test_data.shape[0], 0))
    #     if not self.fix_material:
    #         train_sparse_matrices = []
    #         test_sparse_matrices = []
    #         for feature_name in self.features_for_vectorizer:
    #             vocab = self.get_vocab(feature_name)
    #             # Bag of Words processing
    #             vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern=r"\b\w+\b")
    #             train_matr = vectorizer.fit_transform(train_data[feature_name].values)
    #             test_matr = vectorizer.transform(test_data[feature_name].values)
    #             train_sparse_matrices.append(train_matr)
    #             test_sparse_matrices.append(test_matr)
    #         train_vectorized_features = sparse.hstack(train_sparse_matrices, format='csr').toarray()
    #         test_vectorized_features = sparse.hstack(test_sparse_matrices, format='csr').toarray()
    #         mms = MinMaxScaler()
    #         train_vectorized_features = mms.fit_transform(train_vectorized_features)
    #         test_vectorized_features = mms.transform(test_vectorized_features)
    #
    #     train_comb = self.window_combinations(train_data)
    #     test_comb = self.window_combinations(test_data)
    #     train_resulting_shape = (len(train_comb),
    #                              train_ohe_constant.shape[1] +
    #                              (train_ohe_changing.shape[1] +
    #                               train_numerical_features.shape[1] +
    #                               train_vectorized_features.shape[1]) * self.look_back +
    #                              self.current_info *
    #                              ((self.target_feature == 'Amount_HL') * (1 + train_ohe_changing.shape[1] + train_vectorized_features.shape[1]) +
    #                               (self.target_feature == 'dt') * (1 + train_vectorized_features.shape[1])))
    #     test_resulting_shape = (len(test_comb),
    #                             test_ohe_constant.shape[1] +
    #                             (test_ohe_changing.shape[1] +
    #                             test_numerical_features.shape[1] +
    #                             test_vectorized_features.shape[1]) * self.look_back +
    #                             self.current_info *
    #                             ((self.target_feature == 'Amount_HL') * (1 + test_ohe_changing.shape[1] + test_vectorized_features.shape[1]) +
    #                              (self.target_feature == 'dt') * (1 + test_vectorized_features.shape[1])))
    #     X_train = np.empty(train_resulting_shape)
    #     for i, (ref_points, pred_point) in enumerate(train_comb):
    #         X_train[i, :] = np.concatenate([train_ohe_constant[ref_points[0], :].flatten(),
    #                                         train_ohe_changing[ref_points, :].flatten(),
    #                                         train_numerical_features[ref_points, :].flatten(),
    #                                         train_vectorized_features[ref_points, :].flatten(),
    #                                         train_numerical_features[pred_point, 0].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
    #                                         train_ohe_changing[pred_point, :].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
    #                                         train_vectorized_features[pred_point, :].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
    #                                         train_numerical_features[pred_point, 1].flatten() if (self.current_info and self.target_feature == 'dt') else np.empty((0)),
    #                                         train_vectorized_features[pred_point, :].flatten() if (self.current_info and self.target_feature == 'dt') else np.empty((0)),
    #                                         ], axis=0)
    #     X_test = np.empty(test_resulting_shape)
    #     for i, (ref_points, pred_point) in enumerate(test_comb):
    #         X_test[i, :] = np.concatenate([test_ohe_constant[ref_points[0], :].flatten(),
    #                                        test_ohe_changing[ref_points, :].flatten(),
    #                                        test_numerical_features[ref_points, :].flatten(),
    #                                        test_vectorized_features[ref_points, :].flatten(),
    #                                        test_numerical_features[pred_point, 0].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
    #                                        test_ohe_changing[pred_point, :].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
    #                                        test_vectorized_features[pred_point, :].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
    #                                        test_numerical_features[pred_point, 1].flatten() if (self.current_info and self.target_feature == 'dt') else np.empty((0)),
    #                                        test_vectorized_features[pred_point, :].flatten() if (self.current_info and self.target_feature == 'dt') else np.empty((0)),
    #                                         ], axis=0)
    #
    #     return X_train, X_test
    #
    # def prepare_target_regression(self):
    #     """ Prepare target variable for a regression problem.
    #     """
    #     train_data, test_data = self.preprocess_dataframe()
    #     mms_target = MinMaxScaler()
    #     train_target = mms_target.fit_transform(train_data[self.target_feature].values.reshape(-1, 1))
    #     test_target = mms_target.transform(test_data[self.target_feature].values.reshape(-1, 1))
    #     train_comb = self.window_combinations(train_data)
    #     y_train = []
    #     y_train_scaled = []
    #     for i, (ref_points, pred_point) in enumerate(train_comb):
    #         y_train.append(train_data[self.target_feature][pred_point])
    #         y_train_scaled.append(train_target[pred_point])
    #     test_comb = self.window_combinations(test_data)
    #     y_test = []
    #     y_test_scaled = []
    #     for i, (ref_points, pred_point) in enumerate(test_comb):
    #         y_test.append(test_data[self.target_feature][pred_point])
    #         y_test_scaled.append(test_target[pred_point])
    #     return (y_train, y_train_scaled, y_test, y_test_scaled, mms_target)
