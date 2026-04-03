import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

CITIES = ["Кокшетау", "Степногорск", "Щучинск", "Атбасар", "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы"]

def generate_synthetic_dataset(n_samples=5000):
    np.random.seed(42)
    data = {
        "temperature": np.random.normal(3.5, 12, n_samples),
        "rain": np.random.exponential(8, n_samples).clip(0, 55),
        "snow": np.random.exponential(15, n_samples).clip(0, 45),
        "soil_moisture": np.random.beta(3, 2, n_samples),
        "river_level": np.random.normal(52, 9, n_samples).clip(35, 75),
        "city": np.random.choice(CITIES, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    df["snow_melt"] = np.maximum(0, df["temperature"]) * df["snow"] * 0.12
    df["precip_3d"] = df["rain"] * 3
    df["precip_7d"] = df["rain"] * 7
    df["temp_rain_inter"] = df["temperature"] * df["rain"]
    df["soil_river"] = df["soil_moisture"] * df["river_level"]
    
    city_dummies = pd.get_dummies(df["city"], prefix="city")
    df = pd.concat([df.drop("city", axis=1), city_dummies], axis=1)
    
    logit = (
        0.08 * df["temperature"] +
        0.45 * df["rain"] +
        0.55 * df["snow_melt"] +
        1.8 * df["soil_moisture"] +
        0.32 * (df["river_level"] - 48) +
        0.12 * df["precip_3d"] +
        0.05 * df["temp_rain_inter"] +
        np.random.normal(0, 1.2, n_samples)
    )
    df["flood_probability"] = 1 / (1 + np.exp(-logit))
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_flood_data.csv", index=False)
    return df

if __name__ == "__main__":
    generate_synthetic_dataset()
    print("Датасет создан")
    
    df = pd.read_csv("data/synthetic_flood_data.csv")
    feature_cols = [col for col in df.columns if col not in ["flood_probability"]]
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler сохранён")
