"""Data preprocessing pipeline for customer churn prediction."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

DATA_PATH = "../data/customer_churn_dataset-training-master.csv"
OUTPUT_DIR = "../data_processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
gap = 5


