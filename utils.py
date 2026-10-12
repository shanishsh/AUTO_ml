import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exception import CustomException
import sys,os
from pathlib import Path
from logger import logging

# df=pd.read_csv("train.csv")

def checkNull(df):

    x=df.isna().sum()
    cols=df.columns
    total_rows=df.shape[0]

    greater_than_25_percent=[]
    less_than_25_percent=[]
    for i,col in zip(x,cols):
        y=i/total_rows*100

        if y>25:
            greater_than_25_percent.append(col)
        else:
            less_than_25_percent.append(col)

    return (greater_than_25_percent, less_than_25_percent)




class NumericalPlot:
    def __init__(self) -> None:
        pass

    def HistPolot(self,df,numerical_cols):

        try:
            logging.info("inside the hist plot Method")
            length=len(numerical_cols)
            row=length//2

            fig_width=12
            fig_height=row*3.5

            fig,ax=plt.subplots(nrows=row+1,ncols=2,figsize=(fig_width,fig_height))

            ax=ax.flatten()

            logging.info("flatten the ax")

            for i,col in enumerate(numerical_cols):
                sns.histplot(data=df,x=col,ax=ax[i],kde=True)


            logging.info("before tight layout")
            plt.tight_layout()
            # plt.show()

            logging.info("before save")

            root_path=Path(__file__).resolve().parent
            png_path_dir=root_path/"analysis_png"
            png_path=png_path_dir/"histplot"
            png_path_dir.mkdir(parents=True,exist_ok=True)
            plt.savefig(png_path)

            logging.info("ended")


        except Exception as e:
            raise CustomException(e,sys)

    def BoxPlot(self,df,numerical_cols):

        try:
            logging.info("inside the boxplot Method")

            length=len(numerical_cols)
            row=length//2+1

            fig_width=12
            fig_height=row*3.5


            fig,ax=plt.subplots(nrows=row,ncols=2,figsize=(fig_width,fig_height))

            ax=ax.flatten()

            for i,col in enumerate(numerical_cols):

                sns.boxplot(data=df,x=col,ax=ax[i])

            plt.tight_layout()
            # plt.show()

            root_path=Path(__file__).resolve().parent
            png_path_dir=root_path/"analysis_png"
            png_path=png_path_dir/"boxplot"
            png_path_dir.mkdir(parents=True,exist_ok=True)
            plt.savefig(png_path)

            logging.info("ended")

        except Exception as e:
            raise CustomException(e,sys)


    def heat_map(self,df:pd.DataFrame):

        corr=df.corr(numeric_only=True)

        sns.heatmap(corr,annot=True)
        # plt.show()

        root_path=Path(__file__).resolve().parent
        png_path_dir=root_path/"analysis_png"
        png_path=png_path_dir/"heatmap"
        png_path_dir.mkdir(parents=True,exist_ok=True)
        plt.savefig(png_path)



class CatPlot:
    def __init__(self) -> None:
        pass


    def count_plot(self,cat_cols,df:pd.DataFrame):

        try:

            length=len(cat_cols)
            row=length//2
            fig,ax=plt.subplots(nrows=row,ncols=2,figsize=(12,8))

            for i,col in enumerate(cat_cols):
                sns.countplot(data=df,x=col)
            plt.tight_layout()
            plt.xticks(rotation=45)

            root_path=Path(__file__).resolve().parent
            png_path_dir=root_path/"analysis_png"
            png_path=png_path_dir/"countplot"
            png_path_dir.mkdir(parents=True,exist_ok=True)
            plt.savefig(png_path)

        except Exception as e:
            raise CustomException(e,sys)

# class Plots(NumericalPlot,CatPlot):
#     def __init__(self) -> None:
#         super().__init__()
    

    