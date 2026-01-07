import numpy as np
import pandas as pd

# Data preprocessing, splitting and standardization
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

# File to load data
from logging import Logger
# Initialize logger
logger = Logger(__name__)

class Preprocessor:
    def __init__(self,data:pd.DataFrame):
        self.data = data
        self.standardizer = StandardScaler()

    def set_MinMaxScaler(self):
        
        self.standardizer = MinMaxScaler()

    def set_StandardScaler(self):
        self.standardizer = StandardScaler()

    def preprocess(self):
        print("Preprocessing data...")
        # Example preprocessing: fill missing values and encode categorical variables
        self.data.fillna(method='ffill', inplace=True)
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.data[col].astype('category').cat.codes
        return self.data
    
    def standardize(self, X_train, X_test):
        print("Standardizing data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled