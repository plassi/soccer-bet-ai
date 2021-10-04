import pandas as pd
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from sklearn import preprocessing, impute, compose
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline as pipe
import warnings
from pandas.core.common import SettingWithCopyWarning

from .helpers.features3 import Features


class ApiFootballDataset(Dataset):
    """
    Dataset for API Football dataset.
    """

    def __init__(self, df):
        """
        Args:
            csv_path (str): path to csv file
            set (str): 'lstm' or 'ff'
        """

        # ignore SettingWithCopyWarning
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        self.features = Features()

        self.df = df


        # Arrange frame by fixture_date
        self.df = self.df.sort_values(by=['fixture_date'])

        ##########################################################################
        #
        #   Create y_data
        #
        ##########################################################################

        # print(Xy_data['teams_home_winner'])
        self.y_data = np.empty(shape=[0, 3], dtype=np.int8)

        for data in self.df['teams_home_winner']:
            if data != data:
                newrow = np.array([[0, 1, 0]], dtype=np.int8)
                self.y_data = np.append(self.y_data, newrow, axis=0)
            if data == 'True':
                newrow = np.array([[1, 0, 0]], dtype=np.int8)
                self.y_data = np.append(self.y_data, newrow, axis=0)
            if data == 'False':
                newrow = np.array([[0, 0, 1]], dtype=np.int8)
                self.y_data = np.append(self.y_data, newrow, axis=0)

        #########################################################################
        #
        #   Create X_data transformers
        #
        #########################################################################

        print('Creating data pipelines...')

        # Create Numerical data pipeline
        numerical_pipeline = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                 fill_value=0, add_indicator=True),
            preprocessing.MinMaxScaler(),
            verbose=1,
        )

        # Create onehotencode data pipeline
        onehotencode_pipeline = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                 fill_value=np.nan, add_indicator=True),
            preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=True),
            verbose=1,
        )

        

        ##########################################################################

        #   Create WoE encoder

        ##########################################################################

        # print('Creating WoE encoder...')

        # woe_pipeline = pipe([
        #     ('label_encoder', RareLabelEncoder(
        #         tol=0.001, n_categories=100, ignore_format=True)),
        #     ('missing_indicators', AddMissingIndicator()),
        #     ('categorical_imputer', CategoricalImputer()),
        #     ('woe_encoder', WoEEncoder(ignore_format=True)),
        #     ('min_max_scaler', preprocessing.MinMaxScaler(),)
        # ], verbose=True,)

        # Compose master X_data_pipeline
        self.X_data_pipeline = compose.make_column_transformer(
            (numerical_pipeline, self.features.numerical),
            (onehotencode_pipeline, self.features.onehotencode + self.features.WoEencode),
            # (woe_pipeline, self.features.WoEencode),
        )


        ##########################################################################

        # Fit X_data_pipeline
        self.X_data = self.X_data_pipeline.fit_transform(
            self.df)

        print("X_data_pipeline created")

    #########################################################################

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        X_torch = torch.from_numpy(self.X_data[idx].toarray()).type(torch.float32)
        y_torch = torch.from_numpy(self.y_data[idx]).type(torch.float32).view(1, 3)

        return X_torch, y_torch
