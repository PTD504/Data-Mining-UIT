import subprocess
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import ray

ray.init(ignore_reinit_error=True)

def setup_mlflow():
    print("Đang thiết lập MLflow...")
    os.makedirs("mlruns", exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")

    experiment_name = "geese-migration"
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    print(f"Theo dõi MLflow đã được thiết lập với thí nghiệm '{experiment_name}', ID: {experiment_id}")
    return mlflow.get_tracking_uri(), experiment_id


@ray.remote
def run_data_preprocessing(tracking_uri, experiment_id):
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"

    print("Đang chạy tiền xử lý dữ liệu...")
    result = subprocess.run(
        [sys.executable, "src/data/data-preprocessing.py"],
        env=env_vars,
        check=True,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    print(result.stdout)
    print(result.stderr)
    return "Tiền xử lý dữ liệu hoàn tất"


@ray.remote
def run_gan_model(tracking_uri, experiment_id):
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"

    print("Đang chạy huấn luyện mô hình GAN...")
    result = subprocess.run(
        [sys.executable, "src/models/GAN.py"],
        env=env_vars,
        check=True,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    print(result.stdout)
    print(result.stderr)
    return "Huấn luyện mô hình GAN hoàn tất"


@ray.remote
def run_vae_model(tracking_uri, experiment_id):
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"

    print("Đang chạy huấn luyện mô hình VAE...")
    result = subprocess.run(
        [sys.executable, "src/models/VAE.py"],
        env=env_vars,
        check=True,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    print(result.stdout)
    print(result.stderr)
    return "Huấn luyện mô hình VAE hoàn tất"


def main():
    tracking_uri, experiment_id_str = setup_mlflow()

    data_preprocessing_ref = run_data_preprocessing.remote(tracking_uri, experiment_id_str)
    result_preprocessing = ray.get(data_preprocessing_ref)
    print(result_preprocessing)

    if "thất bại" in result_preprocessing:
         print("Dừng pipeline do tiền xử lý dữ liệu thất bại.")
         return

    gan_model_ref = run_gan_model.remote(tracking_uri, experiment_id_str)
    vae_model_ref = run_vae_model.remote(tracking_uri, experiment_id_str)

    results = ray.get([gan_model_ref, vae_model_ref])
    print(results[0])
    print(results[1])

    print("Pipeline hoàn tất.")
    ray.shutdown()

if __name__ == "__main__":
    main()