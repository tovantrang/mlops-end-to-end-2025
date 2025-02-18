# 🏷️ SMART - Smart Merchandise Automated Recognition Technology

Ce projet vise à développer une solution basée sur la **Computer Vision** pour reconnaître automatiquement un ensemble de 10 produits. Il suit les bonnes pratiques de **MLOps** et exploite des outils comme **Picsellia, YOLO, MLFlow et BentoML**.

---

## 🚀 Fonctionnalités

- **Détection automatique** de 10 types de produits à partir d'images ou de vidéos.
- **Pipeline de training** utilisant **YOLO** et un dataset annoté depuis **Picsellia**. Les paramètres du modèle sont optimisés via le finetune génétique d'**Ultralytics**.
- **Tracking des expériences et versioning** des modèles via **MLFlow**. Avec le stockage sur **MinIO** et **MySQL** le tout sur docker.
- **Déploiement des modèles** avec **BentoML** pour réaliser des inférences.
- **Modes d'inférence** : traitement d'images, vidéos ou webcam en temps réel avec visualisation.

---

## 🛠️ Stack technique

| Technologie  | Usage |
|-|-|
| **Python 3.11** | Langage principal |
| **Picsellia** | Gestion des datasets et annotations |
| **Ultralytics YOLO** | Modèle de détection d'objets |
| **MLFlow** | Tracking d'expériences et Model Registry |
| **MinIO** | Stockage des modèles et artefacts |
| **MySQL** | Stockage des métriques et logs |
| **BentoML** | Déploiement et serving des modèles |
| **Docker** | Conteneurisation des pipelines |

---

## 📂 Structure du projet

```
📂 smart_project
│── 📁 docker/
|── 📁 pipelines/
│   ├── data_pipeline.py
│   ├── serving_pipeline.py
│   └── training_pipeline.py
│── 📁 src/
│   ├── 📁 config/
│   ├── config.py
│   ├── data_load.py
│   ├── data_preparation.py
│   ├── data_validation.py
│   ├── download_champion.py
│   ├── inference.py
│── 📁 test/
│   ├── test_api.py
│   └── test.jpg
│── compose.yml
│── requirements.txt
│── service.py
│── bentomlfile.yaml
│── README.md
```

---

## 🚀 Installation et utilisation

### 1️⃣ Installation des dépendances et environnement et docker-compose
```bash
pip install -r requirements.txt
```
Il y a 3 fichiers de configuration à créer :

- Basés sur les fichiers `src/config/.env.Template`
    - `src/config/.env` contenant les variables d'environnement pour docker-compose.
    - `src/config/.local.env` contenant les variables d'environnement pour le code en local

- Basé sur le fichier `src/config/.secrets.toml.Template`
    - `src/config/.secrets.toml` contenant les secrets pour picsellia

D'autres éléments (tels que les chemins pour les différents dossiers servant à stocker les données) peuvent être modifié dans le fichier `src/config/settings.toml`

```bash
docker compose build
docker compose up -d
```

Vérifier que le S3 bucket mlflow est créer (ou le nom défini dans .env)

### 2️⃣ Exécuter la pipeline d'entraînement du modèle
```bash
python pipelines/training_pipeline.py
```

### 3️⃣ Lancer une inférence sur une image
```bash
python src/inference.py  --mode IMAGE --input path/to/image.jpg
```

### 4️⃣ Lancer une inférence sur une vidéo
```bash
python src/inference.py  --mode VIDEO --input path/to/video.mp4
```
Utiliser n'importe quelle touche pour quitter la vidéo.

### 5️⃣ Lancer une inférence via webcam
```bash
python src/inference.py  --mode WEBCAM
```

Utiliser la touche `q` pour quitter la fenêtre de la webcam.

### 6️⃣ Déployer le modèle avec BentoML
```bash
python pipelines/serving_pipeline.py
```

Le login à **BentoML** est automatique mais il faut bien mettre les bonnes informations dans le fichier `.local.env`

### 7️⃣ Tester l'API déployée
```bash
python test/test_api.py
```

---
## ![problems](https://img.icons8.com/emoji/48/000000/exclamation-mark-emoji.png) Problèmes rencontrés
Nous n'avons pas réussi à déployer le modèle avec BentoML sur le cloud car le déploiement ne se terminait pas (même après plus d'une heure). Par contre le déploiement en local fonctionne parfaitement.
```bash
bentoml serve .
```
Nous avons fait des tests sur CLI, avec build et deploy, et le scripts rendu ou il ne faut plus faire les build, push,... mais cela ne fonctionne pas et se bloque à l'étape de déploiement sur une attente d'une réplication de docker.

---
## 👥 Auteurs
- **Thomas Perreuil** - Télécom physique Strasbourg
- **Nicolas To Van Trang** - Télécom physique Strasbourg
