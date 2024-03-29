import os
import sys

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np


class CustomDataset:
    def __init__(self):
        pass

    def get_data_as_dataframe(self,
                 LIMIT_BAL: float,
                 SEX: int,
                 EDUCATION: int,
                 MARRIAGE: int,
                 AGE: int,
                 PAY_1: int,
                 PAY_2: int,
                 PAY_3: int,
                 PAY_4: int,
                 PAY_5: int,
                 PAY_6: int,
                 BILL_AMT1: float,
                 BILL_AMT2: float,
                 BILL_AMT3: float,
                 BILL_AMT4: float,
                 BILL_AMT5: float,
                 BILL_AMT6: float,
                 PAY_AMT1: float,
                 PAY_AMT2: float,
                 PAY_AMT3: float,
                 PAY_AMT4: float,
                 PAY_AMT5: float,
                 PAY_AMT6: float):
        try:
            logging.info("Making dictionary")
            custom_data_input_dict = {
                'LIMIT_BAL': LIMIT_BAL,
                'SEX': SEX,
                'EDUCATION': EDUCATION,
                'MARRIAGE': MARRIAGE,
                'AGE': AGE,
                'PAY_1': PAY_1,
                'PAY_2': PAY_2,
                'PAY_3': PAY_3,
                'PAY_4': PAY_4,
                'PAY_5': PAY_5,
                'PAY_6': PAY_6,
                'BILL_AMT1': BILL_AMT1,
                'BILL_AMT2': BILL_AMT2,
                'BILL_AMT3': BILL_AMT3,
                'BILL_AMT4': BILL_AMT4,
                'BILL_AMT5': BILL_AMT5,
                'BILL_AMT6': BILL_AMT6,
                'PAY_AMT1': PAY_AMT1,
                'PAY_AMT2': PAY_AMT2,
                'PAY_AMT3': PAY_AMT3,
                'PAY_AMT4': PAY_AMT4,
                'PAY_AMT5': PAY_AMT5,
                'PAY_AMT6': PAY_AMT6
            }
            logging.info(f"{custom_data_input_dict}")
            logging.info("Making Dataframe")
            df = pd.DataFrame(custom_data_input_dict, index=[0])
            logging.info("DataFrame gathered")
            print(df)
            return df

        except Exception as e:
            logging.info("An error has occurred in get_data_as_dataframe method")
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:

            # Getting paths for preprocessor anr model pickled file
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # loading data from pickled file
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # arr = np.array(features)
            # reshaped_arr = arr.reshape(1, -1)
            #
            # print(reshaped_arr)

            data_scaled = preprocessor.transform(features)
            logging.info("Data scaled using scaling")
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            logging.info("Exception has occurred in prediction")
            raise CustomException(e, sys)