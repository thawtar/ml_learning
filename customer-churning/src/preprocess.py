import numpy as np
import pandas as pd


# File to load data
from logging import Logger
# Initialize logger
logger = Logger(__name__)

class Preprocessor:
    def __init__(self,data:pd.DataFrame):
        self.data = data

    def preprocess(self):
        print("Preprocessing data...")
        # Example preprocessing: fill missing values and encode categorical variables
        self.data.fillna(method='ffill', inplace=True)
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.data[col].astype('category').cat.codes
        return self.data