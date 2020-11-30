# Project Imports
import pandas as pd
# import plotly.offline as py  # visualization
# py.init_notebook_mode(connected=True)  # visualization
from sklearn.model_selection import train_test_split

# Module Imports
from utils.data_pre_process import data_processing
from utils.model_building import telecom_churn_prediction
from config import Models


def train_model(telecom):
    # defining the studied or used independent features (columns) as well the target
    target_col = ['Churn']
    cols = [i for i in telecom.columns if i not in target_col]

    # splitting the principal training dataset to subtrain and subtest datasets
    x_train, x_test, y_train, y_test = train_test_split(telecom[cols], telecom[target_col],
                                                        test_size=.25, random_state=111)

    for model_name, model in Models.items():
        telecom_churn_prediction(model, x_train, x_test, y_train, y_test)
