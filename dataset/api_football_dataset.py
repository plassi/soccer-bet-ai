import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset
from sklearn import preprocessing, impute, compose
from sklearn.pipeline import make_pipeline
import torch
import warnings
from pandas.core.common import SettingWithCopyWarning

from .helpers.features import Features

from pickle import dump, load


class ApiFootballDataset(Dataset):
    """
    Dataset for API Football dataset.
    """

    def __init__(self, dataframe, Xy_pair):
        """
        Args:
            csv_path (str): path to csv file
            Xy_pair (str): '1' or '2'
        """

        # ignore SettingWithCopyWarning
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

        self.Xy_pair = Xy_pair

        self.frame = dataframe

        # Set features
        self.features = Features(columns=self.frame.columns)


        #########################################################################
        #
        #   Create X_data transformers
        #
        #########################################################################

        if self.Xy_pair == '1':
            
            print("Getting datacolumns...")
            # Get different types of datacolumns
            self.X_nominal_features = self.features.X_nominal_features
            self.X_ordinal_features = self.features.X_ordinal_features
            self.X_numerical_features = self.features.X_numerical_features
            self.X_percentage_features = self.features.X_percentage_features

            # Create X_nominal_data_pipeline
            X_nominal_data_pipeline = make_pipeline(
                impute.SimpleImputer(strategy='constant',
                                    fill_value='none', add_indicator=True),
                preprocessing.OneHotEncoder(),
                verbose=1
            )

            # Create X_ordinal_data_pipeline
            X_ordinal_data_pipeline = make_pipeline(
                impute.SimpleImputer(strategy='constant',
                                    fill_value=-1, add_indicator=True),
                preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                            unknown_value=-1),
                verbose=1,
            )

            # Create X_numerical_data_pipeline
            X_numerical_data_pipeline = make_pipeline(
                impute.SimpleImputer(
                    strategy='constant', fill_value=0, add_indicator=True),
                preprocessing.MinMaxScaler(),
                verbose=1
            )

            # Compose master X_data_pipeline
            self.X_data_pipeline = compose.make_column_transformer(
                (X_nominal_data_pipeline, self.X_nominal_features),
                (X_ordinal_data_pipeline, self.X_ordinal_features),
                (X_numerical_data_pipeline, self.X_numerical_features + self.X_percentage_features),
            )
            # Fit X_data_pipeline
            self.X_data_pipeline = self.X_data_pipeline.fit(self.frame)


            print("X_data_pipeline created")

        #########################################################################
        #
        #   Create y_1_data transformers
        #
        #########################################################################

        if self.Xy_pair == '1' or self.Xy_pair == '2':

            # get y_1 features
            self.y_1_nominal_features = self.features.y_1_nominal_features
            self.y_1_numerical_features = self.features.y_1_numerical_features
            self.y_1_percentage_features = self.features.y_1_percentage_features

            print("Create y_1_nominal_data_pipeline")
            # Create y_1_nominal_data_pipeline
            self.y_1_nominal_data_pipeline = make_pipeline(
                impute.SimpleImputer(
                    strategy='constant', fill_value='0', add_indicator=True),
                preprocessing.OneHotEncoder(),
                verbose=1
            )

            print("Create y_1_numerical_data_pipeline")
            # Create y_1_numerical_data_pipeline
            self.y_1_numerical_data_pipeline = make_pipeline(
                impute.SimpleImputer(
                    strategy='constant', fill_value=0, add_indicator=True),
                preprocessing.MinMaxScaler(),
                verbose=1
            )
            
            print("Create y_1_data_pipeline")
            # Compose master y_1_data_pipeline
            self.y_1_data_pipeline = compose.make_column_transformer(
                (self.y_1_nominal_data_pipeline, self.y_1_nominal_features),
                (self.y_1_numerical_data_pipeline, self.y_1_numerical_features),
            )
            # Fit y_1_data_pipeline
            self.y_1_data_pipeline = self.y_1_data_pipeline.fit(self.frame)

            print("y_1_data_pipeline created")



        #########################################################################
        #
        #   Create y_2_data transformers
        #
        #########################################################################

        if self.Xy_pair == '2':
            
            self.y_2_nominal_features = self.features.y_2_nominal_features
            self.y_2_numerical_features = self.features.y_2_numerical_features

            # create y_2_nominal_data_pipeline
            self.y_2_nominal_data_pipeline = make_pipeline(
                impute.SimpleImputer(
                    strategy='constant', fill_value='0', add_indicator=True),
                preprocessing.OneHotEncoder(),
                verbose=1
            )

            # create y_2_numerical_data_pipeline
            self.y_2_numerical_data_pipeline = make_pipeline(
                impute.SimpleImputer(
                    strategy='constant', fill_value=0, add_indicator=True),
                preprocessing.MinMaxScaler(),
                verbose=1
            )
            
            # Compose master y_2_data_pipeline
            self.y_2_data_pipeline = compose.make_column_transformer(
                (self.y_2_nominal_data_pipeline, self.y_2_nominal_features),
                (self.y_2_numerical_data_pipeline, self.y_2_numerical_features),
            )
            # Fit y_2_data_pipeline
            self.y_2_data_pipeline = self.y_2_data_pipeline.fit(self.frame)

            print("y_2_data_pipeline created")




        #########################################################################

    def __len__(self):

        return len(self.frame)


    def __getitem__(self, idx):

        if self.Xy_pair == '1':
            # Create X_numpy
            try:
                X_numpy = self.X_data_pipeline.transform(
                    pd.DataFrame(self.frame.iloc[idx]).transpose(),
                )
            except Exception as e:
                print(f'e: {e}')
                print(f'e.args: {e.args}')
                print(f'self.frame.iloc[idx].shape: {self.frame.iloc[idx].shape}')

        if self.Xy_pair == '1' or self.Xy_pair == '2':
            # Create y_1_numpy
            y_1_numpy = self.y_1_data_pipeline.transform(
                pd.DataFrame(self.frame.iloc[idx]).transpose()
            )

        if self.Xy_pair == '2':
            # Create y_2_numpy
            y_2_numpy = self.y_2_data_pipeline.transform(
                pd.DataFrame(self.frame.iloc[idx]).transpose()
            )

        #########################################################################
        #
        # return tensors
        #
        if self.Xy_pair == '1':
            returnable = [0, 1]
            # check if X_numpy is np.ndarray
            if isinstance(X_numpy, np.ndarray):
                returnable[0] = torch.from_numpy(X_numpy)
            else:
                returnable[0] = torch.from_numpy(X_numpy.toarray())

            if isinstance(y_1_numpy, np.ndarray):
                returnable[1] = torch.from_numpy(y_1_numpy)
            else:
                returnable[1] = torch.from_numpy(y_1_numpy.toarray())

            return returnable[0], returnable[1]

        if self.Xy_pair == '2':
            return torch.from_numpy(y_1_numpy.toarray()), torch.from_numpy(y_2_numpy)
