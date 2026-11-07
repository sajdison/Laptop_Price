from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "laptop_price_model.joblib"
DATA_PATH = BASE_DIR / "laptop_dataset.csv"
TARGET = "price_usd"

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


def train_model():
    """Train the Gradient Boosting pipeline from the CSV (same setup as train_laptop_model.py)."""
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    model = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=600,
                        learning_rate=0.08,
                        max_leaf_nodes=48,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model.fit(X, y)
    return model


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the saved model. If it was saved with a different scikit-learn version
    and can't be loaded, retrain it from the CSV so the app still works."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return train_model()


model = load_model()

st.title("💻 Laptop Price Estimation Tool")
st.markdown("Configure the laptop specifications below to predict the estimated price.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["⚙️ Core Hardware", "🖥️ Display & Build", "🔌 Features & Extras"]
)

# All option lists below match the values the model was trained on.
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        brand = st.selectbox(
            "Brand",
            ["Acer", "Apple", "Asus", "Dell", "Gigabyte", "HP", "Huawei", "LG",
             "Lenovo", "MSI", "Microsoft", "Razer", "Samsung", "Xiaomi"],
            index=3,
        )
        ram_gb = st.select_slider("RAM (GB)", options=[4, 8, 16, 32, 64], value=16)
    with col2:
        cpu = st.selectbox(
            "CPU",
            ["AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 9",
             "Apple M1", "Apple M2", "Apple M3", "Apple M3 Pro", "Apple M4",
             "Intel Celeron", "Intel Core Ultra 5", "Intel Core Ultra 7",
             "Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Core i9",
             "Intel Pentium"],
            index=13,
        )
        storage_gb = st.select_slider(
            "Storage Capacity (GB)", options=[128, 256, 512, 1024, 2048, 4096], value=512
        )
    with col3:
        gpu = st.selectbox(
            "GPU",
            ["Integrated", "NVIDIA GTX 1650", "NVIDIA RTX 3050", "NVIDIA RTX 3060",
             "NVIDIA RTX 4050", "NVIDIA RTX 4060", "NVIDIA RTX 4070",
             "NVIDIA RTX 4080", "NVIDIA RTX 4090", "NVIDIA RTX 5070",
             "AMD Radeon RX 6600M"],
        )
        storage_type = st.selectbox("Storage Type", ["SSD", "HDD+SSD", "eMMC"])

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        screen_inch = st.select_slider(
            "Screen Size (Inches)", options=[11.6, 13.3, 14.0, 15.6, 16.0, 17.3], value=15.6
        )
        resolution = st.selectbox(
            "Resolution",
            ["1366x768", "1920x1080", "1920x1200", "2560x1440", "2560x1600",
             "2880x1800", "3840x2160"],
            index=1,
        )
    with col2:
        panel_type = st.selectbox("Panel Type", ["IPS", "OLED", "TN", "VA", "Mini-LED"])
        refresh_hz = st.selectbox("Refresh Rate (Hz)", [60, 90, 120, 144, 165, 240])
    with col3:
        weight_kg = st.number_input(
            "Weight (kg)", min_value=0.8, max_value=3.5, value=1.8, step=0.05, format="%.2f"
        )
        build_material = st.selectbox(
            "Build Material", ["Aluminum", "Carbon fiber", "Magnesium", "Plastic"]
        )

with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        os_name = st.selectbox(
            "Operating System",
            ["Windows 11 Home", "Windows 11 Pro", "macOS", "Linux", "ChromeOS", "No OS"],
        )
        battery_wh = st.number_input("Battery (Wh)", min_value=40, max_value=100, value=56)
        release_year = st.number_input(
            "Release Year", min_value=2020, max_value=2026, value=2024, step=1
        )
    with col2:
        condition = st.selectbox("Condition", ["New", "Refurbished", "Used"])
        warranty_years = st.number_input("Warranty (Years)", min_value=1, max_value=3, value=1)
        usb_ports = st.number_input("USB Ports Count", min_value=2, max_value=7, value=3)
    with col3:
        touchscreen = st.checkbox("Touchscreen", value=False)
        backlit_keyboard = st.checkbox("Backlit Keyboard", value=True)
        fingerprint_reader = st.checkbox("Fingerprint Reader", value=False)
        wifi = st.selectbox("Wi-Fi Standard", ["WiFi 5", "WiFi 6", "WiFi 6E", "WiFi 7"], index=1)

st.markdown("---")

if st.button("🚀 Predict Laptop Price", use_container_width=True, type="primary"):
    input_data = pd.DataFrame(
        [
            {
                "brand": brand,
                "cpu": cpu,
                "gpu": gpu,
                "ram_gb": ram_gb,
                "storage_gb": storage_gb,
                "storage_type": storage_type,
                "screen_inch": screen_inch,
                "resolution": resolution,
                "panel_type": panel_type,
                "refresh_hz": refresh_hz,
                "touchscreen": int(touchscreen),
                "os": os_name,
                "weight_kg": weight_kg,
                "battery_wh": battery_wh,
                "release_year": release_year,
                "condition": condition,
                "backlit_keyboard": int(backlit_keyboard),
                "fingerprint_reader": int(fingerprint_reader),
                "wifi": wifi,
                "usb_ports": usb_ports,
                "warranty_years": warranty_years,
                "build_material": build_material,
            }
        ]
    )
    try:
        # The saved model already converts log-price back to USD, so no extra expm1 is needed.
        prediction = float(model.predict(input_data)[0])
        st.balloons()
        st.success(f"### Estimated Price: **${prediction:,.2f} USD**")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
