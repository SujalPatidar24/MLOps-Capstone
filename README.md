<h1 align="center">🚀 End-to-End ML Capstone Project Deployment on AWS</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Actively%20Looking-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Skills-MLOps|CI%2FCD|AWS|EKS|Docker-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Built%20With-Python%203.10-yellow?style=for-the-badge"/>
</p>

---

## 📌 About the Project

This project demonstrates an **end-to-end Machine Learning workflow** for a real-world use case, integrated with **MLOps, Cloud Infrastructure (AWS), Docker, and CI/CD pipelines**.

It not only includes model building but also:
- Model versioning via **MLflow** + **DAGsHub**
- Data versioning via **DVC**
- CI/CD using **GitHub Actions**
- Deployment via **Docker** and **AWS EKS (Kubernetes)**
- Monitoring using **Prometheus + Grafana**

> 💡 **Goal:** Showcase production-grade ML project deployment & monitoring — not just model accuracy!

---

## 🧰 Tech Stack

| Category         | Tools Used                                                                 |
|------------------|----------------------------------------------------------------------------|
| Programming      | Python 3.10                                                                |
| ML Libraries     | Scikit-Learn, XGBoost, Pandas                                              |
| MLOps Tools      | MLflow, DVC, GitHub Actions, Cookiecutter                                  |
| Deployment       | Flask, Docker, AWS ECR, AWS EKS                                            |
| Monitoring       | Prometheus, Grafana                                                        |
| Cloud            | AWS (EC2, S3, EKS, IAM, ECR)                                               |
| Version Control  | Git + GitHub + DAGsHub                                                     |

---

## 📂 Project Structure

--------
```

├── src/
│ ├── data_ingestion.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_building.py
│ ├── model_evaluation.py
│ ├── register_model.py
│ └── logger/
├── dvc.yaml
├── params.yaml
├── flask_app/
│ ├── app.py
│ └── Dockerfile
├── .github/
│ └── workflows/
│ └── ci.yaml
├── tests/
├── scripts/
├── requirements.txt

```
--------

## 🧪 ML Pipeline & Experiment Tracking

✅ Tracked via `MLflow` (integrated with DAGsHub)  
✅ Reproducible pipeline built using `DVC`  
✅ Auto CI/CD triggered with GitHub Actions  
✅ Data pushed to `AWS S3` via DVC remote  
✅ Model metrics tracked & visualized on MLflow UI  

---

## 🌐 App Deployment

1. **Dockerized Flask App** — Serving the model.
2. **AWS ECR** — Stores the Docker image.
3. **AWS EKS (Kubernetes)** — Scalable deployment.
4. **LoadBalancer Service** — Public access via External IP.


## 📊 Monitoring Dashboard
Prometheus + Grafana is used to monitor:

- Inference time

- API latency

- Request count

- CPU/RAM usage

> Dashboards hosted on EC2, ports 9090 & 3000.