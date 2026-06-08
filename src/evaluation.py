from src.pipelines.pipeline import build_pipeline
from src.pipelines.pipeline import build_base_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import numpy as np

def train_and_evaluate(X_train,y_train,X_test,y_test,model="rf"):

    # Build pipeline
    if model == "lr":
        pipe =  build_base_pipeline(X_train)
    else: 
        pipe = build_pipeline(X_train)

    # Train model
    pipe.fit(X_train,y_train)

    # Predict
    preds = pipe.predict(X_test)

    # Evaluate on metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    # Proportion of predictions with correct direction (i.e positive / negative)
    direction = sum(y_test.multiply(preds) >= 0)/len(y_test)

    # Print results

    print(f"Direction: {direction}")
    print(f"Mean absolute error: {mae}")
    print(f"Root mean squared error: {rmse}\n")

def build_baseline(X_train,y_train,X_test,y_test):

    # Establish baselines
    base_mean = y_train.mean()

    # Evaluate baseline, based on mean
    base_preds = np.full(len(y_test),base_mean)
    mae_base = mean_absolute_error(y_test,base_preds)
    rmse_base = root_mean_squared_error(y_test,base_preds)
    direction_base = sum(y_test.multiply(base_preds) >= 0) / len(y_test)

    # Print results
    print(f"Direction, baseline: {direction_base}")
    print(f"Mean absolute error, baseline: {mae_base}")
    print(f"Root mean squared error,baseline: {rmse_base}\n")
