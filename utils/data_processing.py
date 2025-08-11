import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import streamlit as st

def load_sensor_data(file_path="./data/PSF_sensors_all.csv"):
    """Load sensor data from the specified CSV file"""
    try:
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            st.info("Please ensure the file `PSF_sensors_all.csv` exists in the `./data/` directory.")
            return None
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Validate the data
        is_valid, message = validate_sensor_data(df)
        
        if not is_valid:
            st.error(f"Data validation failed: {message}")
            return None
        
        # Clean and process the data
        df_clean = clean_sensor_data(df)
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error loading sensor data: {str(e)}")
        return None

def validate_sensor_data(df):
    """Validate uploaded sensor data format for PSF sensors"""
    required_columns = ['Time', 'Temperature', 'Humidity', 'Pressure']
    expected_columns = ['Time', 'TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Temperature', 'Humidity', 'Pressure', 'Resistance']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check Time format (should be hh:mm:ss)
    try:
        # Try to parse a few time values
        pd.to_datetime(df['Time'].head(10), format='%H:%M:%S', errors='raise')
    except:
        try:
            # Try alternative format with seconds
            pd.to_datetime(df['Time'].head(10), errors='raise')
        except:
            return False, "Invalid Time format. Expected format: HH:MM:SS (e.g., 14:30:15)"
    
    # Check if numeric columns are actually numeric
    numeric_columns = ['TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Temperature', 'Humidity', 'Pressure', 'Resistance']
    for col in numeric_columns:
        if col in df.columns:
            # Check if column can be converted to numeric
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            if non_numeric > 0:
                return False, f"Column '{col}' contains {non_numeric} non-numeric values"
    
    # Check for reasonable data ranges
    if 'Temperature' in df.columns:
        temp_range = df['Temperature'].describe()
        if temp_range['min'] < -50 or temp_range['max'] > 70:
            return False, "Temperature values outside reasonable range (-50°C to 70°C)"
    
    if 'Humidity' in df.columns:
        humidity_range = df['Humidity'].describe()
        if humidity_range['min'] < 0 or humidity_range['max'] > 100:
            return False, "Humidity values outside valid range (0% to 100%)"
    
    return True, "Data validation successful"

def clean_sensor_data(df):
    """Clean and preprocess sensor data for PSF format"""
    df_clean = df.copy()
    
    # Convert Time column to datetime for easier plotting
    # Assuming all readings are from the same day, use today's date
    base_date = datetime.now().date()
    
    try:
        # First try standard HH:MM:SS format
        df_clean['datetime'] = pd.to_datetime(
            base_date.strftime('%Y-%m-%d') + ' ' + df_clean['Time'],
            format='%Y-%m-%d %H:%M:%S'
        )
    except:
        try:
            # Try more flexible parsing
            df_clean['datetime'] = pd.to_datetime(
                base_date.strftime('%Y-%m-%d') + ' ' + df_clean['Time']
            )
        except:
            # If all else fails, create a sequential datetime
            df_clean['datetime'] = pd.date_range(
                start=datetime.combine(base_date, datetime.min.time()),
                periods=len(df_clean),
                freq='15T'  # Assume 15-minute intervals
            )
    
    # Sort by time
    df_clean = df_clean.sort_values('datetime')
    
    # Remove exact duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Convert numeric columns and handle data types
    numeric_columns = ['TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Temperature', 'Humidity', 'Pressure', 'Resistance']
    existing_numeric_cols = [col for col in numeric_columns if col in df_clean.columns]
    
    for col in existing_numeric_cols:
        # Convert to numeric, replacing non-numeric values with NaN
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Interpolate missing values
        df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # Fill any remaining NaN values with column mean
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Remove extreme outliers (beyond 4 standard deviations for sensor data)
    for col in existing_numeric_cols:
        if df_clean[col].std() > 0:  # Avoid division by zero
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            
            # More conservative outlier removal for sensor data
            lower_bound = mean - 4 * std
            upper_bound = mean + 4 * std
            
            # Count outliers before removal
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            if outliers > 0:
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def get_data_summary(df):
    """Generate comprehensive data summary"""
    summary = {
        'total_records': len(df),
        'time_span_hours': 0,
        'missing_data_pct': 0,
        'parameter_ranges': {}
    }
    
    if 'datetime' in df.columns:
        time_span = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
        summary['time_span_hours'] = time_span
    
    # Calculate missing data percentage
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    summary['missing_data_pct'] = (missing_cells / total_cells) * 100
    
    # Get parameter ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['parameter_ranges'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    return summary

def detect_anomalies(df, column, method='zscore', threshold=3):
    """Detect anomalies in sensor data"""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    
    data = df[column].dropna()
    
    if method == 'zscore':
        # Z-score method
        z_scores = np.abs((data - data.mean()) / data.std())
        anomalies = z_scores > threshold
    
    elif method == 'iqr':
        # Interquartile range method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = (data < lower_bound) | (data > upper_bound)
    
    else:
        raise ValueError("Method must be 'zscore' or 'iqr'")
    
    # Create boolean series for entire dataframe
    result = pd.Series(False, index=df.index)
    result.loc[data.index] = anomalies
    
    return result

def calculate_air_quality_index(tvoc, eco2):
    """Calculate a simple air quality index based on TVOC and eCO2"""
    
    # TVOC scoring (1-3 scale)
    if tvoc < 220:
        tvoc_score = 1
    elif tvoc < 660:
        tvoc_score = 2
    else:
        tvoc_score = 3
    
    # eCO2 scoring (1-3 scale)
    if eco2 < 1000:
        eco2_score = 1
    elif eco2 < 2000:
        eco2_score = 2
    else:
        eco2_score = 3
    
    # Combined score (1-3 scale)
    combined_score = (tvoc_score + eco2_score) / 2
    
    # Determine quality level
    if combined_score <= 1.5:
        quality = "Good"
    elif combined_score <= 2.5:
        quality = "Moderate"
    else:
        quality = "Poor"
    
    return {
        'score': combined_score,
        'quality': quality,
        'tvoc_score': tvoc_score,
        'eco2_score': eco2_score
    }

def export_analysis_report(df, filename=None):
    """Export analysis report as CSV"""
    if filename is None:
        filename = f"sensor_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Create summary statistics
    summary_stats = df.describe()
    
    # Add air quality metrics if available
    if 'TVOC' in df.columns and 'eCO2' in df.columns:
        avg_tvoc = df['TVOC'].mean()
        avg_eco2 = df['eCO2'].mean()
        aq_index = calculate_air_quality_index(avg_tvoc, avg_eco2)
        
        # Add air quality row to summary
        summary_stats.loc['air_quality_score'] = [aq_index['score'] if col in ['TVOC', 'eCO2'] else np.nan 
                                                  for col in summary_stats.columns]
    
    return summary_stats

def resample_data(df, frequency='1H'):
    """Resample sensor data to different time frequencies"""
    if 'datetime' not in df.columns:
        return df
    
    df_resampled = df.set_index('datetime').resample(frequency).mean()
    df_resampled = df_resampled.reset_index()
    
    # Update Time column to match resampled datetime
    df_resampled['Time'] = df_resampled['datetime'].dt.strftime('%H:%M:%S')
    
    return df_resampled