"""
IT351 – IoT Sensor Data Prediction Dashboard
Run with: streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import streamlit as st

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Sensor Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Data generation (cached)
# ─────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    n_hours = 60 * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(n_hours)]

    hours = np.array([t.hour for t in timestamps])
    days  = np.array([i // 24 for i in range(n_hours)])

    temp_base   = 25 + 7 * np.sin(2 * np.pi * (hours - 6) / 24)
    temperature = temp_base + 0.03 * days + np.random.normal(0, 1.0, n_hours)

    humidity = np.clip(70 - 0.8 * (temperature - 25) + np.random.normal(0, 3, n_hours), 20, 100)

    aqi_base = 50 + 20 * np.exp(-((hours - 8)**2) / 4) + 15 * np.exp(-((hours - 17)**2) / 4)
    aqi = np.clip(aqi_base + np.random.normal(0, 5, n_hours), 0, 200)

    co2_base = 400 + 50 * np.exp(-((hours - 9)**2) / 6) + 40 * np.exp(-((hours - 18)**2) / 6)
    co2 = np.clip(co2_base + np.random.normal(0, 10, n_hours), 380, 700)

    df = pd.DataFrame({
        'timestamp':   timestamps,
        'temperature': np.round(temperature, 2),
        'humidity':    np.round(humidity, 2),
        'aqi':         np.round(aqi, 2),
        'co2_ppm':     np.round(co2, 2)
    })

    # Missing values
    for col in ['temperature', 'humidity', 'aqi', 'co2_ppm']:
        idx = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
        df.loc[idx, col] = np.nan

    # Clean & engineer
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month']       = df['timestamp'].dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['temp_lag1']   = df['temperature'].shift(1)
    df['temp_lag2']   = df['temperature'].shift(2)
    df['temp_lag24']  = df['temperature'].shift(24)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


@st.cache_resource
def train_models(df):
    feature_cols = ['hour', 'day_of_week', 'day_of_year', 'month', 'is_weekend',
                    'humidity', 'aqi', 'co2_ppm', 'temp_lag1', 'temp_lag2', 'temp_lag24']
    X = df[feature_cols]
    y = df['temperature']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    rf = RandomForestRegressor(n_estimators=150, max_depth=12,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    scaler = StandardScaler()
    lr = LinearRegression()
    lr.fit(scaler.fit_transform(X_train), y_train)
    y_pred_lr = lr.predict(scaler.transform(X_test))

    metrics = {
        'rf':  {'MAE':  mean_absolute_error(y_test, y_pred_rf),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                'R2':   r2_score(y_test, y_pred_rf)},
        'lr':  {'MAE':  mean_absolute_error(y_test, y_pred_lr),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                'R2':   r2_score(y_test, y_pred_lr)},
    }

    importances = pd.Series(rf.feature_importances_, index=feature_cols)

    return rf, split_idx, y_test, y_pred_rf, y_pred_lr, metrics, importances, feature_cols


# ─────────────────────────────────────────────
# Load data & models
# ─────────────────────────────────────────────
df = generate_data()
rf, split_idx, y_test, y_pred_rf, y_pred_lr, metrics, importances, feature_cols = train_models(df)

sensor_cols = ['temperature', 'humidity', 'aqi', 'co2_ppm']
colors_map  = {'temperature': 'tomato', 'humidity': 'steelblue', 'aqi': 'darkorange', 'co2_ppm': 'mediumseagreen'}

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/ios-filled/100/4a90d9/sensor.png", width=60)
st.sidebar.title("📡 IoT Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["📊 Overview", "📈 Trends", "🤖 ML Predictions", "📉 Model Metrics"])

day_range = st.sidebar.slider("Days to display", min_value=1, max_value=60, value=7)
sensor_choice = st.sidebar.selectbox("Primary sensor", sensor_cols)

st.sidebar.markdown("---")
st.sidebar.markdown("**IT351 – Data Science for IoT**")

# ─────────────────────────────────────────────
# Filter data by day range
# ─────────────────────────────────────────────
cutoff = df['timestamp'].min() + timedelta(days=day_range)
df_view = df[df['timestamp'] <= cutoff]

# ─────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📡 IoT Sensor Data — Overview")
    st.markdown(f"Showing **{day_range} days** of sensor readings from **{df['timestamp'].min().date()}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡️ Avg Temp", f"{df_view['temperature'].mean():.1f} °C",
              f"{df_view['temperature'].diff().mean():+.2f}")
    c2.metric("💧 Avg Humidity", f"{df_view['humidity'].mean():.1f} %",
              f"{df_view['humidity'].diff().mean():+.2f}")
    c3.metric("🌫️ Avg AQI", f"{df_view['aqi'].mean():.1f}",
              f"{df_view['aqi'].diff().mean():+.2f}")
    c4.metric("🌬️ Avg CO₂", f"{df_view['co2_ppm'].mean():.0f} ppm",
              f"{df_view['co2_ppm'].diff().mean():+.2f}")

    st.markdown("---")
    st.subheader("Sensor Reading Distribution")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, col in zip(axes.flat, sensor_cols):
        ax.hist(df_view[col], bins=35, color=colors_map[col], edgecolor='white', alpha=0.85)
        ax.axvline(df_view[col].mean(), color='black', linestyle='--', linewidth=1.5)
        ax.set_title(col.replace('_', ' ').title(), fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df_view[sensor_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask,
                ax=ax, linewidths=0.5, square=True)
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────
# PAGE: Trends
# ─────────────────────────────────────────────
elif page == "📈 Trends":
    st.title("📈 Sensor Trends")
    st.markdown(f"**{sensor_choice}** over the last **{day_range} days**")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df_view['timestamp'], df_view[sensor_choice],
            color=colors_map[sensor_choice], linewidth=1.5, alpha=0.9)
    ax.fill_between(df_view['timestamp'], df_view[sensor_choice],
                    alpha=0.1, color=colors_map[sensor_choice])
    ax.set_ylabel(sensor_choice.replace('_', ' ').title())
    ax.set_xlabel("Timestamp")
    ax.set_title(f"{sensor_choice.replace('_', ' ').title()} Over Time")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Hourly Averages")
    hourly = df_view.groupby('hour')[sensor_cols].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, col in zip(axes.flat, sensor_cols):
        ax.plot(hourly.index, hourly[col], color=colors_map[col],
                linewidth=2.5, marker='o', markersize=4)
        ax.fill_between(hourly.index, hourly[col], alpha=0.12, color=colors_map[col])
        ax.set_title(col.replace('_', ' ').title())
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────
# PAGE: ML Predictions
# ─────────────────────────────────────────────
elif page == "🤖 ML Predictions":
    st.title("🤖 Temperature Prediction — Random Forest")

    n_points = st.slider("Number of test points to display", 50, 500, 200)

    test_ts = df['timestamp'].iloc[split_idx:].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test_ts[:n_points], y_test.values[:n_points],
            label='Actual', color='steelblue', linewidth=2)
    ax.plot(test_ts[:n_points], y_pred_rf[:n_points],
            label='RF Predicted', color='tomato', linestyle='--', linewidth=1.8)
    ax.set_title("Actual vs Predicted Temperature", fontsize=13, fontweight='bold')
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Importance")
    imp = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    imp.plot.barh(ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Residual Analysis")
    residuals = y_test.values - y_pred_rf
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred_rf, residuals, alpha=0.3, color='slategray', s=8)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residual')
    axes[1].hist(residuals, bins=40, color='slateblue', edgecolor='white', alpha=0.85)
    axes[1].set_title('Residual Distribution')
    axes[1].set_xlabel('Residual')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────
# PAGE: Model Metrics
# ─────────────────────────────────────────────
elif page == "📉 Model Metrics":
    st.title("📉 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest MAE",  f"{metrics['rf']['MAE']:.4f} °C")
    col2.metric("Random Forest RMSE", f"{metrics['rf']['RMSE']:.4f} °C")
    col3.metric("Random Forest R²",   f"{metrics['rf']['R2']:.4f}")

    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    col4.metric("Linear Reg MAE",  f"{metrics['lr']['MAE']:.4f} °C",
                f"{metrics['lr']['MAE'] - metrics['rf']['MAE']:+.4f} vs RF")
    col5.metric("Linear Reg RMSE", f"{metrics['lr']['RMSE']:.4f} °C",
                f"{metrics['lr']['RMSE'] - metrics['rf']['RMSE']:+.4f} vs RF")
    col6.metric("Linear Reg R²",   f"{metrics['lr']['R2']:.4f}",
                f"{metrics['lr']['R2'] - metrics['rf']['R2']:+.4f} vs RF")

    st.subheader("Side-by-Side Comparison")
    metrics_df = pd.DataFrame({
        'Model': ['Random Forest', 'Linear Regression'],
        'MAE':   [metrics['rf']['MAE'],  metrics['lr']['MAE']],
        'RMSE':  [metrics['rf']['RMSE'], metrics['lr']['RMSE']],
        'R²':    [metrics['rf']['R2'],   metrics['lr']['R2']],
    }).set_index('Model')

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, metric, color in zip(axes, ['MAE', 'RMSE', 'R²'], ['steelblue', 'tomato', 'mediumseagreen']):
        bars = ax.bar(metrics_df.index, metrics_df[metric],
                      color=[color, 'lightgray'], edgecolor='white')
        ax.set_title(metric, fontsize=13)
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Raw Metrics Table")
    st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen').format("{:.4f}"),
                 use_container_width=True)
