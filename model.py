import torch
import torch.nn as nn

class FloodModel(nn.Module):
    def __init__(self, input_dim=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Для загрузки на Streamlit Cloud
if __name__ == "__main__":
    model = FloodModel()
    torch.save(model, "models/flood_model.pt")
    print("Модель сохранена")
