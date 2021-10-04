from datetime import datetime
import glob
from pickle import dump, load
from tqdm import tqdm
from .features3 import Features

import pandas as pd

class Load_data:

    def __init__(self, csv_data_path):
        self.csv_data_path = csv_data_path
        self.data = self.load_data()

    def load_data(self):
        #########################################################################
        #
        # Load data
        #
        #########################################################################
        
        # Check for cached data
        if not glob.glob("cache/dataframe.pkl"):

            features = Features()

            all_files = glob.glob(self.csv_data_path + "/*.csv")

            li = []

            print("Loading data ")
            for filename in tqdm(all_files):
                df = pd.read_csv(filename, index_col=None, header=0, dtype='object')

                # if 'lineups_0_startXI_0_player_id' is empty, drop line
                df = df.dropna(subset=['lineups_0_startXI_0_player_id'])

                df = df.dropna(subset=features.WoEencode)

                # Drop all rows that have df['fixture_status_short'] other than 'FT'
                df = df[df['fixture_status_short'] == 'FT']

                li.append(df)

            frame = pd.concat(li, axis=0, ignore_index=True)

            # dump(frame, open('cache/dataframe.pkl', 'wb'))

            return frame
        
        else:
            frame = load(open('cache/dataframe.pkl', 'rb'))
            print("Dataframe loaded from cache")
            
            return frame

    def get_data(self):
        return self.data