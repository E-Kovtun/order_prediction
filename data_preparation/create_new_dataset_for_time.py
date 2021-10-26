import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
from data_preparation.features_description import categorical_features, features_for_vectorizer


class OrderDataset():
    def __init__(self, data_folder, train_file, test_file, look_back):
        self.data_folder = data_folder
        self.train_file = train_file
        self.test_file = test_file
        self.look_back = look_back

    def group_rows(self, df):
        df_copy = df.copy(deep=True)
        for feature_name in features_for_vectorizer:
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
        train = pd.read_csv(os.path.join(self.data_folder, self.train_file))
        test = pd.read_csv(os.path.join(self.data_folder, self.test_file))

        # delete unnecessary features, add day and month columns
        for df in [train, test]:
            df.drop(['Unnamed: 0', 'HKUNNR', 'group', 'Delivery_year'], axis=1, inplace=True)
            df['Delivery_Date_week'] = pd.to_datetime(df['Delivery_Date_week'])
            df.insert(4, 'Day', df.loc[:, 'Delivery_Date_week'].apply(lambda x: x.day))
            df.insert(5, 'Month', df.loc[:, 'Delivery_Date_week'].apply(lambda x: x.month))

        # encode categorical features
        cat_encoder = OrdinalEncoder(dtype=np.int64)
        train[categorical_features] = cat_encoder.fit_transform(train[categorical_features])
        test[categorical_features] = cat_encoder.transform(test[categorical_features])

        return train, test

    def get_vocab(self, feature_name):
        encoded_train, _ = self.encode_features()
        unique_values = np.sort(encoded_train[feature_name].unique())
        vocab = dict(zip(map(str, unique_values), unique_values))
        return vocab

    def prepare_data(self):
        encoded_train, encoded_test = self.encode_features()

        # group rows in which different materials are bought at the same day
        train_grouped, test_grouped = self.group_rows(encoded_train), self.group_rows(encoded_test)

        # add column with time difference between observations
        threshold = 0.6

        ships_below_threshold = {"test": [], "train": []}
        df_name = "train"
        for df in [train_grouped, test_grouped]:
            all_differencies = []
            interm_df = df.groupby('Ship.to').agg({'Delivery_Date_week': lambda x: x.diff()})
            for ind in df['Ship.to'].unique():
                all_differencies.extend(np.nan_to_num(interm_df.loc[ind, 'Delivery_Date_week'].days).tolist())
            df.insert(2, 'dt', all_differencies)
            table_of_freq_const_delta = dict.fromkeys(df['Ship.to'].unique())
            df.drop('Delivery_Date_week', axis=1, inplace=True)

            values = dict()
            for rest in interm_df['Delivery_Date_week'].keys():
                table_of_freq_const_delta[rest] = []
                values[rest] = []
                for uniq_delta in np.nan_to_num(np.unique(np.array(interm_df['Delivery_Date_week'][rest][:].days))):
                    if uniq_delta == 0:
                        continue
                    table_of_freq_const_delta[rest].append(
                        np.array([x == uniq_delta for x in interm_df['Delivery_Date_week'][rest][:].days]).sum() / len(
                            interm_df['Delivery_Date_week'][rest]))
                    values[rest].append(uniq_delta)
                if np.array(table_of_freq_const_delta[rest]).max() <= threshold:
                    ships_below_threshold[df_name].append(rest)
            df_name = "test"

        ships_below_threshold = np.intersect1d(np.array(ships_below_threshold["train"]),
                                               np.array(ships_below_threshold["test"]))
        train_grouped.loc[train_grouped['Ship.to'].isin(ships_below_threshold)].to_csv(
            'df_beer_train_time_0.6_threshold.csv',
            sep='\t')
        test_grouped.loc[test_grouped['Ship.to'].isin(ships_below_threshold)].to_csv(
            'df_beer_test_time_0.6_threshold.csv',
            sep='\t')

        return train_grouped, test_grouped
