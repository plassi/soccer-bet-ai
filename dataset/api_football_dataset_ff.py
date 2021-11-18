import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
from sklearn import preprocessing, impute, compose
from sklearn.pipeline import make_pipeline
import warnings
from pandas.core.common import SettingWithCopyWarning

from .features.feature_selector import SelectFeatures

import gc


class ApiFootballDataset(Dataset):
    """
    Dataset for API Football dataset.
    """

    def transform_scores_data(self, X):
        X_copy = X.copy()
        X_copy.resize(X_copy.shape[0], X.shape[1] * 2)

        # loop through rows in X
        print("Transforming scores data")
        for i in tqdm(range(X.shape[0])):
            # loop through columns in row
            for j in range(X.shape[1]):
                # if value is string
                if isinstance(X[i, j], str):
                    values = X[i, j].split('-')
                    if values[0] is not None and values[0] != 'None' and values[0] != 'nan':
                        X_copy[i, j*2] = float(values[0])
                        X_copy[i, j*2+1] = float(values[1])
                    else:
                        X_copy[i, j*2] = 0
                        X_copy[i, j*2+1] = 0
                else:
                    X_copy[i, j*2] = 0
                    X_copy[i, j*2+1] = 0
        return X_copy

    def form_data_transformer(self, X):
        """
        Form data transformer.
        reverse string, find longest string in column. Separate letters to own columns. One-hot-encode.
        :param X:
        :return:
        """
        X_copy = X.copy()
        X_copy.resize(X_copy.shape[0], int(
            self.longest_form_string_length) * 2)
        # loop through rows in X
        print("Transforming forms data")
        for i in tqdm(range(X.shape[0])):
            # loop through columns in row
            for j in range(X.shape[1]):
                if j == 0:
                    start_index = 0
                elif j == 1:
                    start_index = int(self.longest_form_string_length)
                # if value is string
                if isinstance(X[i, j], str):
                    # reverse string
                    x_str = X[i, j][::-1]

                    # loop through letters in x_str
                    for k in range(int(self.longest_form_string_length)):
                        if k < len(x_str):
                            # add letter to X[i, j]
                            X_copy[i, k + start_index] = x_str[k]
                        else:
                            X_copy[i, k + start_index] = np.nan

        # change X_copy all datatypes to str
        X_copy = X_copy.astype(str)
        return X_copy

    def transform_percentage_data(self, X):
        """
        Transform percentage data to float.
        X is a 2-d numpy array with strings in form "10.0%"
        :param X:
        :return:
        """

        # loop through rows in X and replace strings with floats
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if isinstance(X[i, j], str):
                    try:
                        X[i, j] = float(X[i, j][:-1]) / 100
                    except:
                        X[i, j] = 0
                else:
                    X[i, j] = X[i, j]
        return X

    def __init__(self, df, feature_set=1):
        """
        Args:
            csv_path (str): path to csv file
            set (str): 'lstm' or 'ff'
        """

        # ignore SettingWithCopyWarning
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

        self.features = SelectFeatures().get_feature_set(feature_set)

        df = df.reset_index(drop=True)

        df[self.features.one_hot_encode_features] = df[self.features.one_hot_encode_features].astype(
            str)
        df[self.features.normalize_features] = df[self.features.normalize_features].astype(
            float)
        df[self.features.percentage_features] = df[self.features.percentage_features].astype(
            str)
        df[self.features.form_features] = df[self.features.form_features].astype(
            str)
        df[self.features.scores_features] = df[self.features.scores_features].astype(
            str)

        # For form datas, get the longest strings lengths
        home_form_longest_string = df['predictions_0_teams_home_league_form'].str.len(
        ).max()
        away_form_longest_string = df['predictions_0_teams_away_league_form'].str.len(
        ).max()
        self.longest_form_string_length = max(
            home_form_longest_string, away_form_longest_string)

        print("Longest form string: ", self.longest_form_string_length)

        self.df_columns = df.columns

        ##########################################################################
        #
        #   Create y_data
        #
        ##########################################################################

        # print(Xy_data['teams_home_winner'])
        self.y_data = np.empty(shape=[len(df), 3], dtype=np.int8)

        for i, data in enumerate(df['teams_home_winner']):
            if data == None:
                self.y_data[i, 0] = 0
                self.y_data[i, 1] = 1
                self.y_data[i, 2] = 0

            elif data == True:
                self.y_data[i, 0] = 1
                self.y_data[i, 1] = 0
                self.y_data[i, 2] = 0

            elif data == False:
                self.y_data[i, 0] = 0
                self.y_data[i, 1] = 0
                self.y_data[i, 2] = 1

        # for i, data in enumerate(df['teams_home_winner']):
        #     if data != data:
        #         newrow = np.array([0, 1, 0], dtype=np.int8)
        #         self.y_data[i] = newrow
        #     if data == 'True':
        #         newrow = np.array([1, 0, 0], dtype=np.int8)
        #         self.y_data[i] = newrow
        #     if data == 'False':
        #         newrow = np.array([0, 0, 1], dtype=np.int8)
        #         self.y_data[i] = newrow

        # self.y_data = self.y_data.reshape(
        #     int(self.y_data.shape[0] / self.seq_len), self.seq_len, self.y_data.shape[1])

        print("self.y_data.shape:", self.y_data.shape)
        # print("self.y_data:", self.y_data)

        #########################################################################
        #
        #   Create X_data transformers
        #
        #########################################################################

        print('Creating data pipelines...')

        scores_data_pipeline = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                 fill_value='0-0', add_indicator=True),
            preprocessing.FunctionTransformer(
                self.transform_scores_data,
            ),
            preprocessing.MinMaxScaler(),
            verbose=1,
        )

        # Create percentage_data_pipeline. Input data is in form "10.0%"
        percentage_data_pipeline = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                 fill_value=0, add_indicator=True),
            preprocessing.FunctionTransformer(
                self.transform_percentage_data,
            ),
            preprocessing.MinMaxScaler(),
            verbose=1,
        )

        form_data_pipeline = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                 fill_value=np.nan, add_indicator=False),
            preprocessing.FunctionTransformer(
                self.form_data_transformer),
            preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=True),
            verbose=1
        )

        # Create Numerical data pipeline
        normalize_pipeline = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                 fill_value=0, add_indicator=True),
            preprocessing.MinMaxScaler(),
            verbose=1,
        )

        # Create onehotencode data pipeline
        onehotencode_pipeline = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                 fill_value='None', add_indicator=True),
            preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=True),

            verbose=1,
        )

        # Compose master X_data_pipeline
        self.X_data_pipeline = compose.make_column_transformer(
            (scores_data_pipeline, self.features.scores_features),
            (percentage_data_pipeline, self.features.percentage_features),
            (normalize_pipeline, self.features.normalize_features),
            (onehotencode_pipeline, self.features.one_hot_encode_features),
            (form_data_pipeline, self.features.form_features),
            sparse_threshold=1
        )

        ##########################################################################

        # Fit X_data_pipeline
        self.X_data_pipeline = self.X_data_pipeline.fit(
            df)

        self.X_data = self.X_data_pipeline.transform(df)

        del df

        gc.collect()

        # X_data_slicer = gen_batches(self.X_data.shape[0], self.seq_len)
        # self.X_slices = [slice for slice in X_data_slicer]

        # print("X_data", self.X_data)
        # transform X_data shape to self.seq_length

        # print("X_data.shape", self.X_data.shape)
        # print("X_data", self.X_data)

        print("X_data_pipeline created")

    #########################################################################

    def __len__(self):

        return self.X_data.shape[0]

    def __getitem__(self, idx):

        X_numpy = self.X_data[idx].toarray()

        y_numpy = self.y_data[idx]

        X_torch = torch.from_numpy(
            X_numpy).type(torch.float32)
        # self.X_data[idx].toarray()).type(torch.float32)
        y_torch = torch.from_numpy(y_numpy).type(
            torch.float32)

        # X_torch=torch.from_numpy(
        #     self.X_data[idx]).type(torch.float32)

        return X_torch, y_torch
