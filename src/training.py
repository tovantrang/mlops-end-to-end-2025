from config import settings
import mlflow
from mlflow import MlflowClient
from ultralytics import YOLO


class Trainer:
    def __init__(self, model : str) -> None:
        """
        This class is used to train the model and save it

        Args:
            model : name of the YOLO pre-train file to use
        """
        self.model = YOLO(model)

    def train(self, conf_file : str, run_name : str,
              mlflow_local_path : str ="requirements.txt", mlflow_artifact_path : str ="environment"):
        """
        This function is used to train the model

        Args:
            conf : path of the configuration file for the training data
            run_name : 
            epochs : number of epochs
            device : identifier for the device use to train the model
            save_dir : directory in which save the model
            mlflow_local_path :
            mlflow_artifact_path : 
        """
        with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
            self.model.train(data=conf_file)
            mlflow.log_artifact(
                local_path=mlflow_local_path,
                artifact_path=mlflow_artifact_path,
                run_id=run.info.run_id,
            )

    def register(self, model_name : str) -> None:
        """
        This function is used to save the model

        Args:
            model_name : name to use to save the model
        """
        run_id = mlflow.last_active_run().info.run_id

        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/weights/best.pt",
            name=model_name,
        )

        client = MlflowClient()
        alias = "test"
        '''
        champion_runs = client.search_runs(tag="Champion")
        if champion_runs == []:
            alias = "Champion"
        else:
            if champion_runs[0].loss < mlflow.last_active_run().loss:
                alias = "Challenger"
            else:
                alias = "failure?"
        '''
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=model_version.version,
        )


if __name__ == "__main__":
    trainer = Trainer(settings.model)

    trainer.train(conf_file=settings.conf_file, 
                  run_name=settings.run_name, 
                  mlflow_local_path=settings.mlflow_local_path,
                  mlflow_artifact_path=settings.mlflow_artifact_path)
    trainer.register(model_name=settings.model_name)