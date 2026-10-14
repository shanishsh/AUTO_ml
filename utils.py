import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from exception import CustomException
import sys,os
from pathlib import Path
from logger import logging
from plots import NumericalPlot,CatPlot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
# df=pd.read_csv("train.csv")

def checkNull(df):

    x=df.isna().sum()
    cols=df.columns
    total_rows=df.shape[0]

    greater_than_25_percent=[]
    less_than_25_percent=[]
    sum_null=0
    for i,col in zip(x,cols):
        y=i/total_rows*100
        sum_null=sum_null+i
        if y>25:
            greater_than_25_percent.append(col)
        else:
            less_than_25_percent.append(col)

    return (greater_than_25_percent, less_than_25_percent,sum_null)


def is_regression(df:pd.DataFrame,target)->bool:

    unique_count=df[target].nunique()

    if unique_count<30:
        return False
    else:
        return True

def train_test_set(df:pd.DataFrame,target):

    # encoder=LabelEncoder()
    # encoded_target=encoder.fit_transform(df[target])

    # df[target]=encoded_target

    train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

    root_path=Path(__file__).resolve().parent
    data_path=root_path/"artifacts"

    data_path.mkdir(parents=True,exist_ok=True)

    train_data_path=data_path/"train.csv"
    test_data_path=data_path/"test.csv"
    raw_data_path=data_path/"data.csv"

    df.to_csv(raw_data_path,header=True,index=False)
    df.to_csv(train_data_path,header=True,index=False)
    df.to_csv(test_data_path,header=True,index=False)

    return(
        train_data_path,
        test_data_path
    )

def target_col_encoding(df:pd.DataFrame,target):

    class_dict={}

    classes=df[target].nunique()

    for i,clas in enumerate(classes):
        class_dict[clas]=i

    df[target]=df[target].replace(class_dict)

    return df



class Plots(NumericalPlot,CatPlot):
    def __init__(self) -> None:
        super().__init__()
    

def save_object(file_path,obj):

    try:

        dir_path=Path(file_path)

        dir_path.parent.mkdir(parents=True,exist_ok=True)


        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

    