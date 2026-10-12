import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from utils import NumericalPlot,checkNull,CatPlot
from exception import CustomException
import sys,os
from logger import logging


class MlPipeline:

    def __init__(self,file_path, target_column) -> None:
        self.file_path=file_path
        self.target_column=target_column

        self.df=None


    def numerical_plots(self,num_cols,df):

        try:
            logging.info("created a NumericalPlot() object")
            self.numerical_plot=NumericalPlot()

            self.numerical_plot.HistPolot(df=df,numerical_cols=num_cols)

            self.numerical_plot.BoxPlot(df=df,numerical_cols=num_cols,target=target)

            self.numerical_plot.heat_map(df=df)

        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

    def categorical_plots(self,cat_cols,df):

        try:
            self.cat_plot=CatPlot()

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

            greater_25_null_percentage, less_25_null_percentage=checkNull(df)

            print(f"************************NULL- greater than 25 percentage*****************************\n {greater_25_null_percentage}\n")
            print(f"************************NULL- less than 25 percentage********************************\n{less_25_null_percentage}\n")

            print(f"************************DUPLICATE- rows count******************************\n{df.duplicated().sum()}\n")

            num_cols=df.select_dtypes(exclude="O").columns
            cat_cols=df.select_dtypes(include="O").columns

            print(f"\n*************************Numerical Columns***************************\n{num_cols}\n*************************Categorical Columns*********************\n{cat_cols}\n")

            print(f"**************************")

            print("**********************Numerical Plots\nNumerical distribution")
            
            logging.info("plotting started")
            self.numerical_plots(num_cols=num_cols,df=df)
            self.categorical_plots(cat_cols=cat_cols,df=df)




        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)



        






        
    