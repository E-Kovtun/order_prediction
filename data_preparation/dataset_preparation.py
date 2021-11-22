import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse


class OrderDataset:
    def __init__(self, data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value,
                 classif=None):
        self.data_folder = data_folder
        self.train_file = train_file
        self.test_file = test_file
        self.look_back = look_back
        self.fix_material = fix_material  # whether we consider the history of orders related to particular material
        self.current_info = current_info  # whether to use information related to timastamp for which we make predictions
        self.predicted_value = predicted_value  # 'amount' or 'time'
        self.classif = classif
        if self.predicted_value == 'amount':
            self.target_feature = 'Amount_HL'
        if self.predicted_value == 'time':
            self.target_feature = 'dt'
        if self.predicted_value == 'classifier':
            self.target_feature = 'Time_7_or_not'

        self.categorical_features = ['Ship.to', 'PLZ', 'Day', 'Month', 'Material', 'Status',
                                     'MaterialGroup.1', 'MaterialGroup.2', 'MaterialGroup.4']
        if self.fix_material:
            # Features that are constant within the time for a particular restaurant if a material is fixed
            self.constant_features_one_hot = ['Ship.to', 'PLZ', 'Material', 'MaterialGroup.1', 'MaterialGroup.2', 'MaterialGroup.4']
        else:
            # Features that are processed with Bag of Words method
            self.features_for_vectorizer = ['Material', 'MaterialGroup.1', 'MaterialGroup.2', 'MaterialGroup.4']
            # Features that are constant within the time for a particular restaurant if a material is not fixed
            self.constant_features_one_hot = ['Ship.to', 'PLZ']
        # Features that are changing withing the time for a particular restaurant
        self.changing_features_one_hot = ['Day', 'Month', 'Status']
        self.numerical_features = ['dt', 'Amount_HL', 'Time_7_or_not']
        self.all_features = self.categorical_features + self.numerical_features

    def group_rows(self, df):
        """ Reorganize rows in an initial dataframe.
        If we consider the history of orders for a particular restaurant and fixed material (self.fix_material is True),
        then the rows are grouped by restaurant id and material id and sorted according to 'Delivery_Date_week'.
        In another case (self.fix_material is False), we combine rows in which different materials are bought
        at the same day for a particular restaurant, summing up 'Amount_HL'.
        """
        df_copy = df.copy(deep=True)
        if self.fix_material:
            df_grouped = df_copy.sort_values(['Ship.to', 'Material', 'Delivery_Date_week']).reset_index(drop=True)
        else:
            for feature_name in self.features_for_vectorizer:
                df_copy[feature_name] = df_copy[feature_name].apply(lambda x: str(x))
            df_grouped = df_copy.groupby(['Ship.to', 'Delivery_Date_week']).agg({'PLZ': pd.Series.mode,
                                                                                 'Material': lambda x: ' '.join(x),
                                                                                 'Day': pd.Series.mode,
                                                                                 'Month': pd.Series.mode,
                                                                                 'Status': lambda x: x.mode().loc[0],
                                                                                 'MaterialGroup.1': lambda x: ' '.join(x),
                                                                                 'MaterialGroup.2': lambda x: ' '.join(x),
                                                                                 'MaterialGroup.4': lambda x: ' '.join(x),
                                                                                 'Amount_HL': np.sum}).reset_index()
        return df_grouped

    def encode_features(self):
        """ Encode categorical features with OrdinalEncoder, delete unnecessary features and create day and month columns.
        """
        train = pd.read_csv(os.path.join(self.data_folder, self.train_file))
        test = pd.read_csv(os.path.join(self.data_folder, self.test_file))

        for df in [train, test]:
            df.drop(['Unnamed: 0', 'HKUNNR', 'group', 'Delivery_year'], axis=1, inplace=True)
            df['Delivery_Date_week'] = pd.to_datetime(df['Delivery_Date_week'])
            df.insert(4, 'Day', df.loc[:, 'Delivery_Date_week'].apply(lambda x: x.day))
            df.insert(5, 'Month', df.loc[:, 'Delivery_Date_week'].apply(lambda x: x.month))

        cat_encoder = OrdinalEncoder(dtype=np.int64)
        train[self.categorical_features] = cat_encoder.fit_transform(train[self.categorical_features])
        test[self.categorical_features] = cat_encoder.transform(test[self.categorical_features])
        return train, test

    def get_vocab(self, feature_name):
        """ Prepare vocabulary of a specified feature for vectorizer input.
        """

        encoded_train, _ = self.encode_features()

        unique_values = np.sort(encoded_train[feature_name].unique())
        vocab = dict(zip(map(str, unique_values), unique_values))
        return vocab

    def add_time_difference(self, df):
        """ Add column with time difference between orders.
        """
        all_differencies = []
        if self.fix_material:
            interm_df = df.groupby(['Ship.to', 'Material']).agg({'Delivery_Date_week': lambda x: x.diff()})
        else:
            interm_df = df.groupby('Ship.to').agg({'Delivery_Date_week': lambda x: x.diff()})
        for ind in interm_df.index:
            corr_diff = np.nan_to_num(interm_df.loc[ind]['Delivery_Date_week'].days).tolist()
            all_differencies.extend(corr_diff if type(corr_diff) == list else [corr_diff])
        df.insert(2, 'dt', all_differencies)
        df.drop('Delivery_Date_week', axis=1, inplace=True)
        if self.classif:
            df['Time_7_or_not'] = df['dt']
            count = 0
            for i in range(len(df['dt'])):
                if df['dt'][i] == 7:
                    df['Time_7_or_not'][i] = 1
                    count += 1
                else:
                    df['Time_7_or_not'][i] = 0


    def preprocess_dataframe(self):
        """ Combine several steps of dataframe preprocessing: features encoding, grouping of rows, adding of new features.
        Return preprocessed train and test dataframes.
        """
        encoded_train, encoded_test = self.encode_features()
        train_grouped, test_grouped = self.group_rows(encoded_train), self.group_rows(encoded_test)
        self.add_time_difference(train_grouped)
        self.add_time_difference(test_grouped)
        return train_grouped, test_grouped

    def window_combinations(self, df):
        """ Specify indices of rows which we join together and index of the row for which we make the prediction.
        """
        chunks_relevant_indices = []
        if self.fix_material:
            proper_cases = []
            history_df = df.groupby(['Ship.to', 'Material']).size()
            for ind in history_df.index:
                if history_df.loc[ind] >= self.look_back + 1:
                    proper_cases.append(ind)
            for (ship_id, material_id) in proper_cases:
                chunks_relevant_indices.append(df.loc[(df['Ship.to'] == ship_id) & (df['Material'] == material_id)].index)
        else:
            for ship_id in df['Ship.to'].unique():
                chunks_relevant_indices.append(df.loc[df['Ship.to'] == ship_id].index)

        all_combinations = []
        for relevant_indices in chunks_relevant_indices:
            for i in range(len(relevant_indices) - self.look_back):
                current_group = relevant_indices[i:i + self.look_back].tolist()
                index_to_pred = relevant_indices[i + self.look_back]
                all_combinations.append((current_group, index_to_pred))
        return all_combinations

    def construct_features(self):
        """ Construct features for Machine Learning algorithms.
        """

        train_data, test_data = self.preprocess_dataframe()

        ohe_constant = OneHotEncoder()
        train_ohe_constant = ohe_constant.fit_transform(train_data[self.constant_features_one_hot]).toarray()
        test_ohe_constant = ohe_constant.transform(test_data[self.constant_features_one_hot]).toarray()

        ohe_changing = OneHotEncoder()
        train_ohe_changing = ohe_changing.fit_transform(train_data[self.changing_features_one_hot]).toarray()
        test_ohe_changing = ohe_changing.transform(test_data[self.changing_features_one_hot]).toarray()

        mms = MinMaxScaler()
        train_numerical_features = mms.fit_transform(train_data[self.numerical_features])
        test_numerical_features = mms.transform(test_data[self.numerical_features])
        train_vectorized_features = np.empty((train_data.shape[0], 0))
        test_vectorized_features = np.empty((test_data.shape[0], 0))
        if not self.fix_material:
            train_sparse_matrices = []
            test_sparse_matrices = []
            for feature_name in self.features_for_vectorizer:
                vocab = self.get_vocab(feature_name)
                # Bag of Words processing
                vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern=r"\b\w+\b")
                train_matr = vectorizer.fit_transform(train_data[feature_name].values)
                test_matr = vectorizer.transform(test_data[feature_name].values)
                train_sparse_matrices.append(train_matr)
                test_sparse_matrices.append(test_matr)
            train_vectorized_features = sparse.hstack(train_sparse_matrices, format='csr').toarray()
            test_vectorized_features = sparse.hstack(test_sparse_matrices, format='csr').toarray()
            mms = MinMaxScaler()
            train_vectorized_features = mms.fit_transform(train_vectorized_features)
            test_vectorized_features = mms.transform(test_vectorized_features)

        train_comb = self.window_combinations(train_data)
        test_comb = self.window_combinations(test_data)
        train_resulting_shape = (len(train_comb),
                                 train_ohe_constant.shape[1] +
                                 (train_ohe_changing.shape[1] +
                                  train_numerical_features.shape[1] +
                                  train_vectorized_features.shape[1]) * self.look_back +
                                 self.current_info *
                                 ((self.target_feature == 'Amount_HL') * (1 + train_ohe_changing.shape[1] + train_vectorized_features.shape[1]) +
                                  (self.target_feature == 'dt') * (1 + train_vectorized_features.shape[1])))
        test_resulting_shape = (len(test_comb),
                                test_ohe_constant.shape[1] +
                                (test_ohe_changing.shape[1] +
                                test_numerical_features.shape[1] +
                                test_vectorized_features.shape[1]) * self.look_back +
                                self.current_info *
                                ((self.target_feature == 'Amount_HL') * (1 + test_ohe_changing.shape[1] + test_vectorized_features.shape[1]) +
                                 (self.target_feature == 'dt') * (1 + test_vectorized_features.shape[1])))
        X_train = np.empty(train_resulting_shape)
        for i, (ref_points, pred_point) in enumerate(train_comb):
            X_train[i, :] = np.concatenate([train_ohe_constant[ref_points[0], :].flatten(),
                                            train_ohe_changing[ref_points, :].flatten(),
                                            train_numerical_features[ref_points, :].flatten(),
                                            train_vectorized_features[ref_points, :].flatten(),
                                            train_numerical_features[pred_point, 0].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
                                            train_ohe_changing[pred_point, :].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
                                            train_vectorized_features[pred_point, :].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
                                            train_numerical_features[pred_point, 1].flatten() if (self.current_info and self.target_feature == 'dt') else np.empty((0)),
                                            train_vectorized_features[pred_point, :].flatten() if (self.current_info and self.target_feature == 'dt') else np.empty((0)),
                                            ], axis=0)
        X_test = np.empty(test_resulting_shape)
        for i, (ref_points, pred_point) in enumerate(test_comb):
            X_test[i, :] = np.concatenate([test_ohe_constant[ref_points[0], :].flatten(),
                                           test_ohe_changing[ref_points, :].flatten(),
                                           test_numerical_features[ref_points, :].flatten(),
                                           test_vectorized_features[ref_points, :].flatten(),
                                           test_numerical_features[pred_point, 0].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
                                           test_ohe_changing[pred_point, :].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
                                           test_vectorized_features[pred_point, :].flatten() if (self.current_info and self.target_feature == 'Amount_HL') else np.empty((0)),
                                           test_numerical_features[pred_point, 1].flatten() if (self.current_info and self.target_feature == 'dt') else np.empty((0)),
                                           test_vectorized_features[pred_point, :].flatten() if (self.current_info and self.target_feature == 'dt') else np.empty((0)),
                                            ], axis=0)

        return X_train, X_test

    def prepare_target_regression(self):
        """ Prepare target variable for a regression problem.
        """

        train_data, test_data = self.preprocess_dataframe()
        mms_target = MinMaxScaler()
        train_target = mms_target.fit_transform(train_data[self.target_feature].values.reshape(-1, 1))
        test_target = mms_target.transform(test_data[self.target_feature].values.reshape(-1, 1))
        train_comb = self.window_combinations(train_data)
        y_train = []
        y_train_scaled = []
        for i, (ref_points, pred_point) in enumerate(train_comb):
            y_train.append(train_data[self.target_feature][pred_point])
            y_train_scaled.append(train_target[pred_point])
        test_comb = self.window_combinations(test_data)
        y_test = []
        y_test_scaled = []
        for i, (ref_points, pred_point) in enumerate(test_comb):
            y_test.append(test_data[self.target_feature][pred_point])
            y_test_scaled.append(test_target[pred_point])
        return (y_train, y_train_scaled, y_test, y_test_scaled, mms_target)
