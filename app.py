import streamlit as st
import torch
import torch.nn as nn
import numpy as np

class FloodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = FloodModel()
model.load_state_dict(torch.load("flood_model.pt", map_location="cpu"))
model.eval()

st.title("🌊 Flood AI Predictor")

temp = st.slider("Temperature", -30.0, 40.0, 5.0)
rain = st.slider("Rain", 0.0, 50.0, 5.0)
snow = st.slider("Snow", 0.0, 30.0, 0.0)
soil = st.slider("Soil Moisture", 0.0, 1.0, 0.3)
river = st.slider("River Level", 40.0, 70.0, 50.0)

snow_melt = max(0, temp) * snow * 0.1
precip_3d = rain * 3
precip_7d = rain * 7

x = np.array([[temp, rain, snow, snow_melt, soil, river,
               precip_3d, precip_7d, 0, 0]])

x = torch.tensor(x, dtype=torch.float32)

if st.button("Predict"):
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    st.write(f"Flood Risk: {prob*100:.2f}%")

    if prob < 0.3:
        st.success("Low Risk")
    elif prob < 0.7:
        st.warning("Medium Risk")
    else:
        st.error("HIGH RISK 🚨")
