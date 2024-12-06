# Import functions

import functions as func
import pandas as pd

if __name__ == "__main__":
    dataset = 'bbc_data_pre_processed.csv'
    df = pd.read_csv(dataset)
    count = func.splitDataset(df)
    print(len(count))






