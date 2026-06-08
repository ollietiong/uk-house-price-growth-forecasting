from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data.preprocessor import build_preprocessor
from src.features.feature_engineering import FeatureEngineer


def build_pipeline(X_train):
    '''Builds a pipeline for a random forest model'''
    # build components
    preprocessor = build_preprocessor(X_train)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    
    # pipeline for model
    pipe = Pipeline(steps=[("features",FeatureEngineer()),("preprocessor",preprocessor),("model",model)])

    return pipe

def build_base_pipeline(X_train):
    '''Builds a pipeline for a linear regression model'''

    # build components
    preprocessor = build_preprocessor(X_train,drop_first=True)
    model = LinearRegression()
    # pipeline for model
    pipe = Pipeline(steps=[("features",FeatureEngineer()),("preprocessor",preprocessor),("model",model)])

    return pipe
