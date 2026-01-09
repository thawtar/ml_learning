"""FastAPI model serving application."""
from contextlib import asynccontextmanager
from typing import Literal
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "../models/model.pkl"
ENCODERS_PATH = "../data_processed/encoders.pkl"
SCALER_PATH = "../data_processed/scaler.pkl"

# Global model and preprocessors
model = None
encoders = None
scaler = None


