import subprocess

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv("src/config/.local.env")
    scripts = ["pipelines/data_pipeline.py", "src/training.py"]

    print("Démarrage du pipeline training...")
    for script in scripts:
        print(f"Exécution de {script}...")
        result = subprocess.run(["python", script], text=True)

        if result.returncode != 0:
            print(f"Erreur lors de l'exécution de {script}")
            break

    print("Pipeline terminée.")
