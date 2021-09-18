import glob
from tqdm import tqdm
from pickle import dump, load

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

            all_files = glob.glob(self.csv_data_path + "/*.csv")

            li = []

            print("Loading data ")
            for filename in tqdm(all_files):
                df = pd.read_csv(filename, index_col=None, header=0, dtype='object')
                li.append(df)

            frame = pd.concat(li, axis=0, ignore_index=True)

            dump(frame, open('cache/dataframe.pkl', 'wb'))

            return frame
        
        else:
            frame = load(open('cache/dataframe.pkl', 'rb'))
            print("Dataframe loaded from cache")
            
            return frame

    def get_data(self):
        return self.data