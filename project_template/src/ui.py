"""Streamlit UI for Customer Churn Prediction."""
import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

