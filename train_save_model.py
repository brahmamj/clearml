from patient_survival_predection_model import PatientPredictionModel
import pandas as pd
from sklearn.metrics import r2_score,accuracy_score
#import mlflow
#from mlflow.models import infer_signature
from clearml import Task
from clearml import Dataset
# Initialize ClearML task
task = Task.init(project_name="Patient Survival Prediction", task_name="Patient_Survival_prediction_train_test", task_type=Task.TaskTypes.optimizer)

parameters = {"n_estimators":200, 
              "max_depth":4, 
              "max_leaves":5, 
              "random_state":42
            }


dataset = Dataset.get(dataset_project="Patient survival prediction", dataset_name="patient_dataset")
local_path = dataset.get_local_copy()


ppm = PatientPredictionModel(local_path+"/heart_failure_clinical_records_dataset.csv", "trained_model/xgboost_model.pkl",parameters)

X, y = ppm.load_data()
ppm.split_data(X, y)
ppm.train_model()
ppm.evaluate_model()
ppm.save_model()

data = pd.read_csv("dataset/heart_failure_clinical_records_dataset.csv")
data = data.sample(100)
data_feat = data.drop('DEATH_EVENT', axis=1)
data_target = data['DEATH_EVENT'].values
r2 = r2_score(data_target, ppm.predict(data_feat))
accuacy = accuracy_score(data_target, ppm.predict(data_feat))

print(f"R2 Score: {r2:.2f}")
task.connect(parameters)
task.get_logger().report_scalar("R2 Score", "score", value=r2,iteration=0)
task.get_logger().report_scalar("Accuracy", "score", value=accuacy,iteration=0)
#with open("trained_model/xgboost_model.pkl", "rb") as f:
task.upload_artifact(name="model", artifact_object="trained_model/xgboost_model.pkl")
        
#params = ppm.model.get_xgb_params()
#print(f"Parameters: {params}")
#mlflow.set_tracking_uri(uri="http://43.205.120.45:5000/")
#mlflow.set_experiment("MLflow Quickstart1")
#with mlflow.start_run():
    #add experiment Id

    #mlflow.log_param("n_estimators", 200)
    #mlflow.log_param("max_depth", params['max_depth'])
    #mlflow.log_param("max_leaves", params['max_leaves'])
    #mlflow.log_metric("r2_score", r2)
    #signature = infer_signature(data_feat, ppm.model.predict(data_feat))
    #mlflow.xgboost.log_model(ppm.model, "model", signature=signature)
