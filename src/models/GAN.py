import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import optuna
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------ Dữ liệu ------------------

df = pd.read_csv('data/data_preprocessed_geese.csv')

features = ['location-long', 'location-lat', 'ground-speed', 'heading', 'height-above-msl', 'year', 'month', 'hour', 'day_of_year']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

data = df[features].values.astype(np.float32)

# Dataset cho GAN
class GeeseDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

train_data, _ = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = GeeseDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ------------------ Mô hình GAN ------------------

# Mô hình Generator 
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x

# Mô hình Discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# ------------------ Huấn luyện GAN ------------------

z_dim = 100  # Kích thước không gian ẩn (latent space)
input_dim = len(features)  # Số đặc trưng đầu vào
output_dim = input_dim  # Số đặc trưng đầu ra của Generator

# Khởi tạo mô hình Generator và Discriminator
generator = Generator(z_dim, output_dim)
discriminator = Discriminator(input_dim)

# Hàm mất mát
criterion = nn.BCELoss()

# Tối ưu hóa
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ------------------ Training loop với MLFlow ------------------

# Theo dõi các tham số và mô hình trong MLFlow
with mlflow.start_run():
    mlflow.log_param("z_dim", z_dim)
    mlflow.log_param("batch_size", 64)
    
    # Huấn luyện mô hình GAN
    num_epochs = 100
    for epoch in range(num_epochs):
        for i, real_data in enumerate(train_loader):
            batch_size = real_data.size(0)
            real_data = real_data.view(batch_size, -1)
            
            # Tạo nhãn thật và giả
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_data), valid)
            z = torch.randn(batch_size, z_dim)
            fake_data = generator(z)
            fake_loss = criterion(discriminator(fake_data.detach()), fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_data), valid)
            g_loss.backward()
            optimizer_G.step()

            # Log metrics vào MLFlow
            if i % 100 == 0:
                mlflow.log_metric("d_loss", d_loss.item(), step=epoch * len(train_loader) + i)
                mlflow.log_metric("g_loss", g_loss.item(), step=epoch * len(train_loader) + i)

        print(f"Epoch [{epoch+1}/{num_epochs}] | d_loss: {d_loss.item()} | g_loss: {g_loss.item()}")
    
    mlflow.pytorch.log_model(generator, "generator")
    mlflow.pytorch.log_model(discriminator, "discriminator")

    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }, os.path.join('models', 'gan_models.pth'))


# ------------------ Tuning siêu tham số với Optuna (Tùy chọn) ------------------

# Hàm tối ưu hóa siêu tham số
def objective(trial):
    # Khám phá các siêu tham số
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    
    # Khởi tạo lại data loader với batch size mới
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Khởi tạo lại mô hình và optimizer
    generator = Generator(z_dim, output_dim)
    discriminator = Discriminator(input_dim)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()

    # Huấn luyện nhanh trong ít epoch để đánh giá
    num_epochs = 20  # Số epoch ngắn gọn để tuning (không cần 100 epochs)
    for epoch in range(num_epochs):
        for i, real_data in enumerate(train_loader):
            batch_size = real_data.size(0)
            real_data = real_data.view(batch_size, -1)
            
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_data), valid)
            z = torch.randn(batch_size, z_dim)
            fake_data = generator(z)
            fake_loss = criterion(discriminator(fake_data.detach()), fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_data), valid)
            g_loss.backward()
            optimizer_G.step()

    return g_loss.item()

# Nếu muốn, có thể sử dụng Optuna để tìm kiếm các siêu tham số tối ưu
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
