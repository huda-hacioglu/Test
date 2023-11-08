import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import os


df = pd.read_csv("Lung Cancer.csv")
df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES': 1, 'NO': 0})
df['GENDER'] = df['GENDER'].replace({'M': 0, 'F': 1})
X=df[['GENDER','AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY','PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING','ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY', 'CHEST PAIN']]
y=df['LUNG_CANCER']

real_testssize = 0.4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=real_testssize, random_state=101)

#XGBoost
def XGBOOST(X_train, y_train, X_test, y_test):

    learning_rate = 0.1
    n_estimators = 100
    max_depth = 2
    random_state = 35

    model = xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = metrics.mean_squared_error(y_test, y_pred)*100
    r2 = metrics.r2_score(y_test, y_pred)*100
    accuracy = accuracy_score(y_test, y_pred)*100

    results =[accuracy,mse,r2]

    return results

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

"""mlflow.set_tracking_uri('http://192.168.1.45:5000/')

experiment_name = "TEST - MLFlow Project"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created experiment '{experiment_name}' with ID {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    print(f"Experiment '{experiment_name}' already exists with ID {experiment_id}. You can display the run from that page")

def generate_next_run_name(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string='')
    num_runs = len(runs)
    return f"test{num_runs + 1}"""

with mlflow.start_run() as run:

    results= XGBOOST(X_train, y_train, X_test, y_test)
    
    mlflow.log_param("learning rate", 0.1)
    mlflow.log_param("n estimators", 100)
    mlflow.log_param(" max depth", 2)
    mlflow.log_param("random state", 35)

    mlflow.log_param("test size", real_testssize)
    mlflow.log_metric("accuracy", results[0])
    mlflow.log_metric("mean squared error", results[1])
    mlflow.log_metric("r squared", results[2])

    colors = ['red', 'green', 'blue', 'purple']

    plt.bar(["Accuracy","Mean squared error","r squared"],[results[0],results[1],results[2]],color=colors)
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Results Comparison')

    plt.grid(True)

    # Save the bar chart as an artifact
    plt.savefig("Results_Comparison.png")
    mlflow.log_artifact("Results_Comparison.png")

    model_name="xgboost - test"
    model_uri = f"runs:/{run.info.run_id}/xgboost-model"
    mv = mlflow.register_model(model_uri, model_name)
    
    input_example = X_train.iloc[[0]]

    mlflow.sklearn.log_model(results, model_name,input_example=input_example)
mlflow.end_run()