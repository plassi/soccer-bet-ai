import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset
from sklearn import preprocessing, impute, compose
from sklearn.pipeline import make_pipeline
import torch
import warnings
from pandas.core.common import SettingWithCopyWarning

from .helpers.features2 import Features


class ApiFootballDataset(Dataset):
    """
    Dataset for API Football dataset.
    """

    def __init__(self, dataframe):
        """
        Args:
            csv_path (str): path to csv file
            set (str): 'lstm' or 'ff'
        """

        # ignore SettingWithCopyWarning
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


        # Set features
        features = Features(columns=dataframe.columns)

        self.players_df = dataframe[features.players_features]
        self.predictions_df = dataframe[features.ff_input_features]
        # self.X_data = frame[features.players_features + features.ff_input_features]


        #########################################################################
        #
        #   Create X_data transformers
        #
        #########################################################################
            
        print('Creating data pipelines...')

        # Create X_ordinal_data_pipeline

        
        self.X_data_pipeline_players = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                fill_value=np.nan, add_indicator=False),
            preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=True),
            verbose=1,
        )


        self.X_data_pipeline_predictions = make_pipeline(
            impute.SimpleImputer(strategy='constant',
                                fill_value=0, add_indicator=True),
            preprocessing.MinMaxScaler(),
            verbose=1,
        )

        # # Compose master X_data_pipeline
        # self.X_data_pipeline = compose.make_column_transformer(
        #     (X_data_pipeline_01, features.ff_input_features),
        #     (X_data_pipeline_02, features.players_features),
        # )
        # # Fit X_data_pipeline
        # self.X_data_pipeline = self.X_data_pipeline.fit(self.X_data)

        # print("X_data_pipeline created")

        self.X_data_pipeline_players = self.X_data_pipeline_players.fit(self.players_df)
        self.X_data_pipeline_predictions = self.X_data_pipeline_predictions.fit(self.predictions_df)



        ##########################################################################
        #
        #   Create y_data transformers
        #
        ##########################################################################

        # print(Xy_data['teams_home_winner'])
        self.y_data = []


        for data in dataframe['teams_home_winner']:
            if data != data:
                self.y_data.append([0, 1, 0])
            if data == 'True':
                self.y_data.append([1, 0, 0])
            if data == 'False':
                self.y_data.append([0, 0, 1])

        # print(self.y_data)



    #########################################################################

    def __len__(self):

        return len(self.players_df)


    def __getitem__(self, idx):

        players_X_numpy = self.X_data_pipeline_players.transform(
            pd.DataFrame(self.players_df.iloc[idx]).transpose()
        )
        players_X_numpy = players_X_numpy.astype('float32')

        predictions_X_numpy = self.X_data_pipeline_predictions.transform(
            pd.DataFrame(self.predictions_df.iloc[idx]).transpose()
        )
        predictions_X_numpy = predictions_X_numpy.astype('float32')

        players_X_torch = torch.from_numpy(players_X_numpy.toarray())
        predictions_X_torch = torch.from_numpy(predictions_X_numpy)
        y_numpy = torch.from_numpy(np.array([self.y_data[idx]]))

        return players_X_torch, predictions_X_torch, y_numpy
       