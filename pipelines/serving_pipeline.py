import subprocess

import bentoml
from dotenv import load_dotenv


BENTO_NAME = "YOLOservice"
DEPLOYMENT_NAME = "my-yolo-deployment"


def run_command(command: str):
    """Execute a shell command and return the return code.

    Args:
        command (str): The command to execute
    """
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    process.wait()
    return process.returncode


if __name__ == "__main__":
    load_dotenv("src/config/.env")
    run_command("python src/download_champion.py")
    run_command("bentoml cloud login")

    dep = bentoml.deployment.create(bento=".", name="yolo-deployment")
    print(
        f"Le déploiement a été créé avec succès : {dep} et est en cours de déploiement."
    )

    dep.wait_until_ready(timeout=3600)

    deployment_info = bentoml.deployment.get("yolo-deployment")
    print("endpoint url:", deployment_info.get_endpoint_urls())
