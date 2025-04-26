import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ------------------ Dữ liệu ------------------

df = pd.read_csv('data/data_preprocessed_geese.csv')

features = ['location-long', 'location-lat', 'ground-speed', 'heading', 'height-above-msl', 'year', 'month', 'hour', 'day_of_year']

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

data = df[features].values.astype(np.float32)

# Dataset cho VAE
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

# ------------------ Mô hình VAE ------------------

# Mô hình Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2(h1), self.fc3(h1)  

# Mô hình Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h1 = self.relu(self.fc1(z))
        return self.sigmoid(self.fc2(h1))

# Mô hình VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KL = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KL

# ------------------ Huấn luyện VAE ------------------

input_dim = len(features)
hidden_dim = 256
latent_dim = 2  # Tùy chỉnh kích thước không gian ẩn

# Khởi tạo mô hình VAE
vae = VAE(input_dim, hidden_dim, latent_dim)

# Tối ưu hóa
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# ------------------ Training loop với MLFlow ------------------

# Theo dõi các tham số và mô hình trong MLFlow
with mlflow.start_run():
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("latent_dim", latent_dim)
    mlflow.log_param("batch_size", 64)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            data = data.view(data.size(0), -1)  

            # Forward pass
            optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data)

            # Tính loss
            loss = loss_function(recon_batch, data, mu, log_var)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log metrics vào MLFlow
            if i % 100 == 0:
                mlflow.log_metric("loss", loss.item(), step=epoch * len(train_loader) + i)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item()}")

    model_path = 'models/vae_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(vae.state_dict(), model_path)
    mlflow.log_artifact(model_path)

    mlflow.pytorch.log_model(vae, "vae_model")
