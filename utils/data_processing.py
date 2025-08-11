from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import os
import streamlit as st

def load_sensor_data(file_path: str | None = None):
    """Load sensor data from the specified CSV file (defaults to ./data/PSF_sensors_all.csv)"""
    try:
        if file_path is None:
            base = Path(__file__).resolve().parents[1]  # repo root
            file_path = base / "data" / "PSF_sensors_all.csv"
        else:
            file_path = Path(file_path)

        if not file_path.exists():
            st.error(f"Data file not found: {file_path}")
            st.info("Please ensure the file `PSF_sensors_all.csv` exists in the `./data/` directory.")
            return None

        df = pd.read_csv(file_path)

        is_valid, message = validate_sensor_data(df)
        if not is_valid:
            st.error(f"Data validation failed: {message}")
            return None

        df_clean = clean_sensor_data(df)
        return df_clean

    except Exception as e:
        st.error(f"Error loading sensor data: {str(e)}")
        return None

def validate_sensor_data(df: pd.DataFrame):
    """Validate uploaded sensor data format for PSF sensors"""
    required_columns = ['Time', 'Temperature', 'Humidity', 'Pressure']
    expected_columns = ['Time', 'TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Temperature', 'Humidity', 'Pressure', 'Resistance']

    # Check required cols
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"

    # Time format: accept HH:MM:SS, be tolerant
    try:
        pd.to_datetime(df['Time'].astype(str).str.strip().head(10), format='%H:%M:%S', errors='raise')
    except Exception:
        try:
            pd.to_datetime(df['Time'].astype(str).str.strip().head(10), errors='raise')
        except Exception:
            return False, "Invalid Time format. Expected format: HH:MM:SS (e.g., 14:30:15)"

    # Numeric checks (tolerate NaNs)
    numeric_columns = ['TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Temperature', 'Humidity', 'Pressure', 'Resistance']
    for col in numeric_columns:
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors='coerce')
            if coerced.isna().sum() > 0 and df[col].notna().sum() > coerced.notna().sum():
                return False, f"Column '{col}' contains non-numeric values"

    # Reasonable ranges
    if 'Temperature' in df.columns:
        temp_min, temp_max = pd.to_numeric(df['Temperature'], errors='coerce').min(), pd.to_numeric(df['Temperature'], errors='coerce').max()
        if temp_min < -50 or temp_max > 70:
            return False, "Temperature values outside reasonable range (-50°C to 70°C)"

    if 'Humidity' in df.columns:
        hum_min, hum_max = pd.to_numeric(df['Humidity'], errors='coerce').min(), pd.to_numeric(df['Humidity'], errors='coerce').max()
        if hum_min < 0 or hum_max > 100:
            return False, "Humidity values outside valid range (0% to 100%)"

    return True, "Data validation successful"

def clean_sensor_data(df: pd.DataFrame):
    """Clean and preprocess sensor data for PSF format"""
    df_clean = df.copy()

    # Normalize Time text
    df_clean['Time'] = df_clean['Time'].astype(str).str.strip()

    # Build datetime using today's date; fall back gracefully
    base_date = datetime.now().date()
    df_clean['datetime'] = pd.to_datetime(
        f"{base_date} " + df_clean['Time'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )
    if df_clean['datetime'].isna().any():
        # Try a flexible parse
        df_clean['datetime'] = pd.to_datetime(f"{base_date} " + df_clean['Time'], errors='coerce')

    # If still NaT, fill a sequential timeline (1-minute steps as a safe default)
    if df_clean['datetime'].isna().any():
        df_clean['datetime'] = df_clean['datetime'].fillna(
            pd.date_range(start=f"{base_date} 00:00:00", periods=len(df_clean), freq='1min')
        )

    # Sort and drop exact duplicates
    df_clean = df_clean.sort_values('datetime')
    df_clean = df_clean.drop_duplicates()

    # Convert numeric cols, interpolate, fill means
    numeric_columns = ['TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Temperature', 'Humidity', 'Pressure', 'Resistance']
    for col in [c for c in numeric_columns if c in df_clean.columns]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].interpolate(method='linear')
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # Remove extreme outliers (±4σ)
    for col in [c for c in numeric_columns if c in df_clean.columns]:
        std = df_clean[col].std()
        if pd.notna(std) and std > 0:
            mean = df_clean[col].mean()
            lower, upper = mean - 4 * std, mean + 4 * std
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def get_data_summary(df: pd.DataFrame):
    summary = {
        'total_records': len(df),
        'time_span_hours': 0,
        'missing_data_pct': 0,
        'parameter_ranges': {}
    }

    if 'datetime' in df.columns:
        summary['time_span_hours'] = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600

    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    summary['missing_data_pct'] = (missing_cells / total_cells) * 100

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['parameter_ranges'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }

    return summary

def detect_anomalies(df: pd.DataFrame, column, method='zscore', threshold=3):
    if column not in df.columns:
        return pd.Series(False, index=df.index)

    data = df[column].dropna()

    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        anomalies = z_scores > threshold
    elif method == 'iqr':
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        anomalies = (data < lower_bound) | (data > upper_bound)
    else:
        raise ValueError("Method must be 'zscore' or 'iqr'")

    result = pd.Series(False, index=df.index)
    result.loc[data.index] = anomalies
    return result

def calculate_air_quality_index(tvoc, eco2):
    if tvoc < 220: tvoc_score = 1
    elif tvoc < 660: tvoc_score = 2
    else: tvoc_score = 3

    if eco2 < 1000: eco2_score = 1
    elif eco2 < 2000: eco2_score = 2
    else: eco2_score = 3

    combined_score = (tvoc_score + eco2_score) / 2
    if combined_score <= 1.5: quality = "Good"
    elif combined_score <= 2.5: quality = "Moderate"
    else: quality = "Poor"

    return {'score': combined_score, 'quality': quality, 'tvoc_score': tvoc_score, 'eco2_score': eco2_score}

def export_analysis_report(df: pd.DataFrame, filename=None):
    if filename is None:
        filename = f"sensor_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_stats = df.describe()
    if 'TVOC' in df.columns and 'eCO2' in df.columns:
        aq = calculate_air_quality_index(df['TVOC'].mean(), df['eCO2'].mean())
        summary_stats.loc['air_quality_score'] = [aq['score'] if c in ['TVOC', 'eCO2'] else np.nan for c in summary_stats.columns]
    return summary_stats

def resample_data(df: pd.DataFrame, frequency='1H'):
    if 'datetime' not in df.columns:
        return df
    df_resampled = df.set_index('datetime').resample(frequency).mean(numeric_only=True).reset_index()
    df_resampled['Time'] = df_resampled['datetime'].dt.strftime('%H:%M:%S')
    return df_resampled
