import mlflow
import mlflow.sklearn
from student_model_tuner import StudentModelTuner

class StudentModelTunerMLflow(StudentModelTuner):
    def __init__(self, data):
        super().__init__(data)

    def train_and_log(self):
        mlflow.start_run()
        model = self.train_and_tune()
        mlflow.log_params(self.best_params)
        mlflow.sklearn.log_model(model, "model")
        mlflow.end_run()
