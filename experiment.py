from student_data_generator import StudentDataGenerator
from student_model_tuner_mlflow import StudentModelTunerMLflow

if __name__ == "__main__":
    # Generate data
    generator = StudentDataGenerator()
    data = generator.generate()

    # Train and log model
    tuner = StudentModelTunerMLflow(data)
    tuner.train_and_log()
