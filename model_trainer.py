import sys,os
from pathlib import Path
from dataclasses import dataclass
from exception import CustomException
from logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# 1. Linear & Regularized Baselines
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor

# 2. Non-Linear Boundary Estimators
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# 3. Instance & Core Base Trees
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# 4. Ensemble Bagging Layouts (Variance Reducers)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor

# 5. High-Performance Boosting Engines (Bias Reducers)
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from typing import Tuple,Dict
from utils import save_object

@dataclass
class ModelTrainerConfig:

    root_path=Path(__file__).resolve().parent
    artifact_path=root_path/"artifacts"

    model_file_obj_path=artifact_path/"model.pkl"


def evaluate_model(models:dict,grid_params:dict,X_train,y_train,X_test,y_test)->Tuple[Dict,Dict]:


    try:
        report={}
        best_models={}
        for name,model in models.items():

            params=grid_params.get(name,{})

            gs=GridSearchCV(estimator=model,param_grid=params,cv=5,n_jobs=-1)

            gs.fit(X_train,y_train)

            best_model=gs.best_estimator_

            y_pred=best_model.predict(X_test)

            report[name]=r2_score(y_test,y_pred)
            best_models[name]=best_model

        return(
            report,
            best_model
        )
    except Exception as e:
        raise CustomException(e,sys)




        


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr:np.ndarray,test_arr:np.ndarray):

        try:

            X_train,y_train, X_test,y_test=train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1]

            if np.nan in y_train or np.nan in y_test:
                 logging.info("nane value in")
            models = {

            # 1. Linear & Regularized Baselines
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0),
            "ElasticNet Regression": ElasticNet(alpha=1.0, l1_ratio=0.5),
            "Huber Regressor": HuberRegressor(max_iter=200),  # Robust against severe outlier anomalies
            
            # 2. Non-Linear Boundary Estimators
            "Support Vector Regressor": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "Gaussian Process Regressor": GaussianProcessRegressor(kernel=RBF(1.0), random_state=42),
            
            # 3. Instance & Core Base Trees
            "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5, weights='uniform'),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            
            # 4. Ensemble Bagging Layouts (Variance Reducers)
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
            "Bagging Regressor": BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, random_state=42, n_jobs=-1),
            
            # 5. High-Performance Boosting Engines (Bias Reducers)
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "AdaBoost Regressor": AdaBoostRegressor(n_estimators=50, random_state=42),
            "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42),
            "LightGBM Regressor": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            "CatBoost Regressor": CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)  # Native categorical wizard
            }

            params = {
            "Linear Regression": {},
            "Ridge Regression": {
                "alpha": [0.1, 1.0, 10.0]
            },
            "Lasso Regression": {
                "alpha": [0.01, 0.1, 1.0]
            },
            "ElasticNet Regression": {
                "alpha": [0.01, 0.1, 1.0],
                "l1_ratio": [0.2, 0.5, 0.8]
            },
            "Huber Regressor": {
                "epsilon": [1.35, 1.5, 2.0]
            },
            "Support Vector Regressor": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"]
            },
            "Gaussian Process Regressor": {
                "alpha": [1e-10, 1e-5, 0.01]
            },
            "K-Neighbors Regressor": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            },
            "Decision Tree Regressor": {
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            },
            "Random Forest Regressor": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10]
            },
            "Extra Trees Regressor": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10]
            },
            "Bagging Regressor": {
                "n_estimators": [10, 50],
                "max_samples": [0.7, 1.0]
            },
            "Gradient Boosting Regressor": {
                "learning_rate": [0.05, 0.1],
                "n_estimators": [50, 100],
                "max_depth": [3, 5]
            },
            "AdaBoost Regressor": {
                "n_estimators": [50, 100],
                "learning_rate": [0.1, 1.0]
            },
            "XGBoost Regressor": {
                "learning_rate": [0.05, 0.1],
                "n_estimators": [50, 100],
                "max_depth": [3, 5]
            },
            "LightGBM Regressor": {
                "learning_rate": [0.05, 0.1],
                "n_estimators": [50, 100],
                "num_leaves": [15, 31]
            },
            "CatBoost Regressor": {
                "iterations": [50, 100],
                "learning_rate": [0.05, 0.1],
                "depth": [4, 6]
            }
        }

            report, best_models=evaluate_model(models=models,grid_params=params,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)

            report_df=pd.DataFrame(report.items(),columns=["model","r2"])

            logging.info(report_df)
            logging.info(best_models)

            best_model_name=report_df["name"].iloc[0]

            logging.info(best_model_name)

            best_model_object=best_models.get(best_model_name)

            # logging.info(best_model_object)

            save_object(self.model_trainer_config.model_file_obj_path,best_model_object)

        except Exception as e:
            raise CustomException(e,sys)
