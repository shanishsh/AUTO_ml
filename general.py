import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
# from utils import NumericalPlot,checkNull,CatPlot
from utils import checkNull,Plots,is_regression,train_test_set,target_col_encoding
from exception import CustomException
import sys,os
from logger import logging
from data_preprocessing import DataPreprocessor
from model_trainer import ModelTrainer


class MlPipeline:

    def __init__(self,file_path, target_column) -> None:
        self.file_path=file_path
        self.target_column=target_column

        self.df=None


    def numerical_plots(self,num_cols,df):

        try:
            logging.info("created a NumericalPlot() object")
            self.numerical_plot=Plots()

            self.numerical_plot.HistPolot(df=df,numerical_cols=num_cols)

            self.numerical_plot.BoxPlot(df=df,numerical_cols=num_cols)

            self.numerical_plot.heat_map(df=df)

        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

    def categorical_plots(self,cat_cols,df):

        try:
            self.cat_plot=Plots()

            self.cat_plot.count_plot(cat_cols=cat_cols,df=df)

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_pipeline(self):

        try:

            logging.info("pipeline initiated")

            df=pd.read_csv(self.file_path)

            print("****************** STARTED ************************\n")
            print(f"First five rows\n{df.head()}\n")

            print(f"************************INFO**********************\n{df.info()}\n")
            print(f"************************SHAPE*********************\n{df.shape}\n")
            print(f"************************COLUMNS*******************\n{df.columns}\n")

            greater_25_null_percentage, less_25_null_percentage,sum_null=checkNull(df)

            print(f"************************NULL- greater than 25 percentage*****************************\n {greater_25_null_percentage}\n")
            print(f"************************NULL- less than 25 percentage********************************\n{less_25_null_percentage}\n")

            print(f"************************DUPLICATE- rows count******************************\n{df.duplicated().sum()}\n")

            num_cols=df.select_dtypes(exclude="O").columns
            cat_cols=df.select_dtypes(include="O").columns

            print(f"\n*************************Numerical Columns***************************\n{num_cols}\n*************************Categorical Columns*********************\n{cat_cols}\n")

            print(f"**************************Check the column with constant value******************\n")

            print("**********************Numerical Plots\nNumerical distribution")
            
            logging.info("plotting started")
            # self.numerical_plots(num_cols=num_cols,df=df)
            # self.categorical_plots(cat_cols=cat_cols,df=df)

            print(f"***********************Target Column********************\n{self.target_column}")

            is_reg=is_regression(df=df,target=self.target_column)
            print(f"*******************IS regression******************\n{is_reg}")

            if not is_reg:
                print(f"**********************Target Categorical")
                df[self.target_column].value_counts(normalize=True)
            else:
                print("***********************Target Regression")

            print(f"**********************Creating Train_set Test_set**********************\n*************")

            # df=target_col_encoding(df=df,target=self.target_column)
            train_data_path,test_data_path=train_test_set(df=df,target=self.target_column)

            return(
                train_data_path,test_data_path, is_reg,sum_null,self.target_column
            )

        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

if __name__=="__main__":
    pl=MlPipeline("train.csv","satisfaction")

    train_data_path,test_data_path,is_reg,sum_null,target_col=pl.initiate_pipeline()

    logging.info("I AM INTO DATA PREPROCESSOR")

    data_preprocessor=DataPreprocessor()

    logging.info("CREATED DATAPREPROCESSOR OBJECT")

    train_arr,test_arr,_=data_preprocessor.initiate_data_preprocessor(train_path=train_data_path,test_path=test_data_path,target=target_col,is_reg=is_reg,sum_null=sum_null)

    model_trainer=ModelTrainer()

    model_trainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)



        






        
    