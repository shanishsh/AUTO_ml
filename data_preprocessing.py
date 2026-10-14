import sys,os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from logger import logging
from exception import CustomException
from utils import save_object


@dataclass
class DataPreprocessorConfig:

    root_path=Path(__file__).resolve().parent

    artifact=root_path/"artifacts"

    train_data_path=artifact/"train.csv"
    # test_data_path=artifact/"test.csv"

    preprocessor_obj_file_path=artifact/"preprocessor.pkl"


class DataPreprocessor:
    def __init__(self) -> None:
        self.data_preprocessor_config=DataPreprocessorConfig()

    def get_preprocessor_obj(self,sum_null,target):

        try:
            df=pd.read_csv(self.data_preprocessor_config.train_data_path)
            dropped_target_df=df.drop(target,axis=1)

            num_cols=dropped_target_df.select_dtypes(exclude="O").columns
            cat_cols=dropped_target_df.select_dtypes(include="O").columns

            if sum_null>25:
                num_pipeline=Pipeline([
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ])
                cat_pipeline=Pipeline([
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("oh",OneHotEncoder(handle_unknown="ignore"))
                ])

                preprocessor=ColumnTransformer([
                    ("num",num_pipeline,num_cols),
                    ("cat",cat_pipeline,cat_cols)
                ])
                
                return preprocessor

            else:
                df.dropna(inplace=True)

                num_pipeline=Pipeline([
                    ("scaler",StandardScaler())
                ])
                cat_pipeline=Pipeline([
                    ("oh",OneHotEncoder(handle_unknown="ignore"))
                ])

                preprocessor=ColumnTransformer([
                    ("num",num_pipeline,num_cols),
                    ("cat",cat_pipeline,cat_cols)
                ],remainder="passthrough")

                return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_preprocessor(self,train_path,test_path,target,is_reg,sum_null):

        try:
            logging.info('train df and test df')
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            input_feature_train_df=train_df.drop(target,axis=1)
            target_feature_train_df=train_df[target]

            input_feature_test_df=test_df.drop(target,axis=1)
            target_feature_test_df=test_df[target]

            preprocessor=self.get_preprocessor_obj(sum_null=sum_null,target=target)

            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,
                target_feature_train_df
            ]
            test_arr=np.c_[
                input_feature_test_arr,
                target_feature_test_df
            ]

            save_object(self.data_preprocessor_config.preprocessor_obj_file_path,preprocessor)

            return(
                train_arr,
                test_arr,
                self.data_preprocessor_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)



        

        
        
