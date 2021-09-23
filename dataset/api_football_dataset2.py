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

    def __init__(self, dataframe, dataset):
        """
        Args:
            csv_path (str): path to csv file
            set (str): 'lstm' or 'ff'
        """

        self.dataset = dataset

        # ignore SettingWithCopyWarning
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        frame = dataframe

        # Set features
        features = Features(columns=self.frame.columns)

        self.X_data = frame[features.ff_input_features]



       

        #########################################################################
        #
        #   Create X_data transformers
        #
        #########################################################################
            
        print('Creating data pipelines...')

        if self.dataset == 'lstm':

            # Create X_ordinal_data_pipeline
            X_data_pipeline_01 = make_pipeline(
                impute.SimpleImputer(strategy='constant',
                                    fill_value=np.nan, add_indicator=True),
                preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan),
                # preprocessing.MaxAbsScaler(),
                verbose=1,
            )

            X_data_pipeline_02 = make_pipeline(
                impute.SimpleImputer(strategy='constant',
                                    fill_value=0, add_indicator=True),
                preprocessing.MinMaxScaler(),
                verbose=1,
            )


            # Compose master X_data_pipeline
            self.X_data_pipeline = compose.make_column_transformer(
                (X_data_pipeline_01, features.lstm_string_features),
                (X_data_pipeline_02, [feature for feature in features.lstm_input_features if feature not in features.lstm_string_features]),
            )
            # Fit X_data_pipeline
            self.X_data_pipeline = self.X_data_pipeline.fit(self.X_data)

            print("LSTM_data_pipeline created")


        if self.dataset == 'ff':

            self.X_data_pipeline = make_pipeline(
                impute.SimpleImputer(strategy='constant',
                                    fill_value=0, add_indicator=True),
                preprocessing.MinMaxScaler(),
                verbose=1,
            )

            self.X_data_pipeline = self.X_data_pipeline.fit(self.X_data)


        ##########################################################################
        #
        #   Create y_data transformers
        #
        ##########################################################################

        if self.dataset == 'lstm':

            self.y_data = Xy_data['goals_home'].astype('int') - Xy_data['goals_away'].astype('int')

        if self.dataset == 'ff':

            # print(Xy_data['teams_home_winner'])
            self.y_data = []


            for data in Xy_data['teams_home_winner']:
                if data != data:
                    self.y_data.append([0, 1, 0])
                if data == 'True':
                    self.y_data.append([1, 0, 0])
                if data == 'False':
                    self.y_data.append([0, 0, 1])

            # print(self.y_data)



    #########################################################################

    def __len__(self):

        return len(self.X_data)


    def __getitem__(self, idx):

        if self.dataset == 'lstm':

            X_numpy = self.X_data_pipeline.transform(
                pd.DataFrame(self.X_data.iloc[idx]).transpose()
            )
            X_numpy = X_numpy.astype('float32')

            X_torch = torch.from_numpy(X_numpy)
            y_numpy = torch.from_numpy(np.array([[self.y_data.iloc[idx]]], dtype=np.float32))

            return X_torch, y_numpy

        if self.dataset == 'ff':

            X_numpy = self.X_data_pipeline.transform(
                pd.DataFrame(self.X_data.iloc[idx]).transpose()
            )
            X_numpy = X_numpy.astype('float32')

            X_torch = torch.from_numpy(X_numpy)
            y_numpy = torch.from_numpy(np.array([self.y_data[idx]]))

            return X_torch, y_numpy
       