import subprocess
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import ray  # Thêm Ray cho tính toán phân tán

# Khởi tạo Ray
ray.init(ignore_reinit_error=True)

def setup_mlflow():
    """Thiết lập máy chủ theo dõi MLflow"""
    print("Đang thiết lập MLflow...")
    os.makedirs("mlruns", exist_ok=True)
    
    # Đặt URI theo dõi thành thư mục cục bộ
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Tạo thí nghiệm nếu chưa tồn tại
    experiment_name = "geese-migration"
    client = MlflowClient()
    
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    print(f"Theo dõi MLflow đã được thiết lập với thí nghiệm '{experiment_name}', ID: {experiment_id}")

@ray.remote
def run_data_preprocessing(tracking_uri, experiment_id):
    # Thiết lập biến môi trường
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"  # Quan trọng: đặt mã hóa thành UTF-8
    
    print("Đang chạy tiền xử lý dữ liệu...")
    try:
        # Sử dụng env để truyền các biến môi trường
        result = subprocess.run(
            [sys.executable, "src/data/data-preprocessing.py"],
            env=env_vars,
            check=True  # Hiển thị lỗi nếu có xảy ra
        )
        return "Tiền xử lý dữ liệu hoàn tất"
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy tiền xử lý: {e}")
        return "Tiền xử lý dữ liệu thất bại"

@ray.remote
def run_gan_model(tracking_uri, experiment_id):
    # Thiết lập biến môi trường
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"  # Đặt mã hóa thành UTF-8
    
    print("Đang huấn luyện mô hình GAN...")
    try:
        result = subprocess.run(
            [sys.executable, "src/models/GAN.py"],
            env=env_vars,
            check=True
        )
        return "Huấn luyện mô hình GAN hoàn tất"
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy mô hình GAN: {e}")
        return "Huấn luyện mô hình GAN thất bại"

@ray.remote
def run_vae_model(tracking_uri, experiment_id):
    # Thiết lập biến môi trường
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"
    
    print("Đang huấn luyện mô hình VAE...")
    try:
        result = subprocess.run(
            [sys.executable, "src/models/VAE.py"],
            env=env_vars,
            check=True
        )
        return "Huấn luyện mô hình VAE hoàn tất"
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy mô hình VAE: {e}")
        return "Huấn luyện mô hình VAE thất bại"

def main():
    # Thiết lập theo dõi MLflow
    setup_mlflow()
    
    # Lấy thông tin để truyền cho các tác vụ
    tracking_uri = mlflow.get_tracking_uri()
    experiment_id = mlflow.get_experiment_by_name("geese-migration").experiment_id
    
    print(f"URI theo dõi MLflow: {tracking_uri}")
    print(f"ID thí nghiệm MLflow: {experiment_id}")
    
    # Chạy các thành phần của pipeline
    print("Đang khởi động pipeline ML...")
    
    try:
        # Truyền thông tin MLflow cho các tác vụ
        preprocessing_result = ray.get(run_data_preprocessing.remote(tracking_uri, experiment_id))
        print(preprocessing_result)
        
        # Tương tự cho các tác vụ khác
        gan_future = run_gan_model.remote(tracking_uri, experiment_id)
        vae_future = run_vae_model.remote(tracking_uri, experiment_id)
        
        # Chờ các tác vụ hoàn tất
        gan_result, vae_result = ray.get([gan_future, vae_future])
        
        print(gan_result)
        print(vae_result)
        
        print("Pipeline hoàn tất thành công!")
        
    except Exception as e:
        print(f"Lỗi trong pipeline: {e}")

if __name__ == '__main__':
    main()