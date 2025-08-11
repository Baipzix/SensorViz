import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path

from utils.data_processing import (
    validate_sensor_data,
    clean_sensor_data,
    load_sensor_data,
)
from utils.visualization import (
    plot_time_series,
    plot_correlation_matrix,
    create_dashboard_plots,   # now auto-handles all available parameters
    plot_sensor_comparison,
)

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Forest Gas Sensor Data Analyzer --Pacific Spirit Forest",
    layout="wide"
)

# ---------------- Session state & auto-load ----------------
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None

# Auto-load ./data/PSF_sensors_all.csv if present
if st.session_state.sensor_data is None:
    default_path = Path(__file__).resolve().parent / "data" / "PSF_sensors_all.CSV"
    if default_path.exists():
        df_auto = load_sensor_data(str(default_path))
        if df_auto is not None and not df_auto.empty:
            st.session_state.sensor_data = df_auto

# ---------------- Header ----------------
st.title("Forest Gas Sensor Data Analyzer")
st.markdown("""
Analyze outdoor sensor data including temperature, humidity, pressure, TVOC, eCO2, and gas sensor readings.
""")

# ---------------- Sidebar Navigation (Dashboard above Analysis) ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📈 Dashboard", "📊 Analysis", "ℹ️ About"],
    index=0
)

# =====================================================================
# DASHBOARD
# =====================================================================
if page == "📈 Dashboard":
    st.header("📈 Dashboard")

    # ---- Upload panel at top of Dashboard ----
    with st.expander("📤 Upload Custom CSV", expanded=(st.session_state.sensor_data is None)):
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Expected columns: Time, TVOC, eCO2, RawH2, RawEthanol, Temperature, Humidity, Pressure, Resistance"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                is_valid, message = validate_sensor_data(df)
                if is_valid:
                    st.success(message)
                    df_clean = clean_sensor_data(df)
                    st.session_state.sensor_data = df_clean
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    if st.session_state.sensor_data is None:
        st.info("No data loaded. Place `PSF_sensors_all.csv` under `./data/` or upload a CSV above.")
    else:
        df = st.session_state.sensor_data

        # ---- Current Conditions (latest reading) ----

        # ---- Dataset Overview (moved from Analysis) ----
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", f"{len(df):,}")
        with col2:
            if 'Temperature' in df.columns:
                st.metric("🌡️ Avg Temperature", f"{df['Temperature'].mean():.1f}°C")
            else:
                st.metric("🌡️ Avg Temperature", "N/A")
        with col3:
            if 'Humidity' in df.columns:
                st.metric("💧 Avg Humidity", f"{df['Humidity'].mean():.1f}%")
            else:
                st.metric("💧 Avg Humidity", "N/A")
        with col4:
            if 'TVOC' in df.columns:
                st.metric("🌬️ Avg TVOC", f"{df['TVOC'].mean():.0f} ppb")
            else:
                st.metric("🌬️ Avg TVOC", "N/A")

        # ---- Current Conditions (latest reading) ----
        st.subheader("🌡️ Current Conditions (Latest Reading)")
        latest_data = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'Temperature' in df.columns:
                temp_delta = latest_data['Temperature'] - df['Temperature'].mean()
                st.metric("🌡️ Temperature", f"{latest_data['Temperature']:.1f}°C", delta=f"{temp_delta:.1f}°C")
            else:
                st.metric("🌡️ Temperature", "N/A")
        with col2:
            if 'Humidity' in df.columns:
                humid_delta = latest_data['Humidity'] - df['Humidity'].mean()
                st.metric("💧 Humidity", f"{latest_data['Humidity']:.1f}%", delta=f"{humid_delta:.1f}%")
            else:
                st.metric("💧 Humidity", "N/A")
        with col3:
            if 'Pressure' in df.columns:
                pressure_delta = latest_data['Pressure'] - df['Pressure'].mean()
                st.metric("🌪️ Pressure", f"{latest_data['Pressure']:.1f} hPa", delta=f"{pressure_delta:.1f} hPa")
            else:
                st.metric("🌪️ Pressure", "N/A")
        with col4:
            if 'TVOC' in df.columns:
                tvoc_delta = latest_data['TVOC'] - df['TVOC'].mean()
                st.metric("🌬️ TVOC", f"{latest_data['TVOC']:.0f} ppb", delta=f"{tvoc_delta:.0f} ppb")
            else:
                st.metric("🌬️ TVOC", "N/A")
            




        # ---- Multi-parameter overview (uniform style, all available params) ----
        st.subheader("🎛️ Multi-Parameter Overview")
        # Order you prefer; only keep those that exist
        param_order = ['Temperature', 'Humidity', 'Pressure', 'TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Resistance']
        available_params = [p for p in param_order if p in df.columns]
        dashboard_fig = create_dashboard_plots(df, params=available_params, ncols=3)
        st.plotly_chart(dashboard_fig, use_container_width=True)

        # ---- Recent trends ----
        st.subheader("⏰ Recent Trends")
        trend_records = st.selectbox("Select number of recent readings", [50, 100, 200, 500], index=1)
        recent_data = df.tail(trend_records)
        y_params = [p for p in ['Temperature', 'Humidity', 'Pressure', 'TVOC'] if p in df.columns]
        if y_params:
            fig_recent = plot_time_series(recent_data, y_params, f"Recent Trends - Last {trend_records} readings")
            if fig_recent:
                st.plotly_chart(fig_recent, use_container_width=True)

        # ---- Data quality ----
        st.subheader("✅ Data Quality Metrics")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with col_q2:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Records", duplicates)
        with col_q3:
            st.metric("Total Readings", f"{len(df):,}")

# =====================================================================
# ANALYSIS
# =====================================================================
elif page == "📊 Analysis":
    st.header("📊 Comprehensive Data Analysis")

    if st.session_state.sensor_data is None:
        st.warning("⚠️ Please load data first (go to **Dashboard** and upload a CSV or place a file under `./data/`).")
    else:
        df = st.session_state.sensor_data
        # ---- Time series visualization ----
        st.subheader("📉 Time Series Visualization")
        environmental_params = [c for c in ['Temperature', 'Humidity', 'Pressure'] if c in df.columns]
        gas_params = [c for c in ['TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Resistance'] if c in df.columns]
        param_category = st.radio("Select parameter category", ["Environmental", "Gas Sensors", "Custom Selection"], horizontal=True)

        if param_category == "Environmental":
            selected_params = environmental_params
        elif param_category == "Gas Sensors":
            selected_params = gas_params
        else:
            available_params = [c for c in df.columns if c not in ['Time', 'datetime']]
            default_sel = [p for p in ['Temperature', 'TVOC'] if p in available_params][:2]
            selected_params = st.multiselect("Select parameters to visualize", available_params, default=default_sel)

        if selected_params:
            fig = plot_time_series(df, selected_params, f"{param_category} Parameters Over Time")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # ---- Time range filter ----
        st.subheader("⏰ Time Range Analysis")
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            start_time = st.time_input("Start Time", value=datetime.strptime("00:00:00", "%H:%M:%S").time())
        with col_time2:
            end_time = st.time_input("End Time", value=datetime.strptime("23:59:59", "%H:%M:%S").time())

        try:
            times = pd.to_datetime(df['Time'].astype(str).str.strip(), format='%H:%M:%S', errors='coerce').dt.time
            mask = (times >= start_time) & (times <= end_time)
            filtered_df = df[mask].copy()
        except Exception:
            filtered_df = df.copy()

        if len(filtered_df) > 0:
            st.write(f"⏰ Showing data from {start_time} to {end_time} ({len(filtered_df)} records)")

            # ---- Statistical summary ----
            st.subheader("📋 Statistical Summary")
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

            # ---- Distribution analysis ----
            st.subheader("📊 Distribution Analysis")
            available_for_dist = [c for c in filtered_df.columns if c not in ['Time', 'datetime']]
            if available_for_dist:
                param_to_analyze = st.selectbox("Select parameter for distribution analysis", available_for_dist)
                col_hist, col_box = st.columns(2)
                with col_hist:
                    st.plotly_chart(px.histogram(filtered_df, x=param_to_analyze, nbins=30,
                                                 title=f'Distribution of {param_to_analyze}'), use_container_width=True)
                with col_box:
                    st.plotly_chart(px.box(filtered_df, y=param_to_analyze,
                                           title=f'Box Plot of {param_to_analyze}'), use_container_width=True)

            # ---- Correlation analysis ----
            st.subheader("🔗 Parameter Correlations")
            fig_corr = plot_correlation_matrix(filtered_df)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough numeric columns to compute correlations.")

            # ---- Sensor Performance (moved here from Air Quality) ----
            st.subheader("⚙️ Sensor Performance")
            col_perf1, col_perf2 = st.columns(2)

            with col_perf1:
                choices = [c for c in ['TVOC', 'eCO2', 'Temperature', 'Humidity'] if c in filtered_df.columns]
                if 'Resistance' in filtered_df.columns and choices:
                    target = st.selectbox("Compare Resistance with:", choices, key="resistance_comparison")
                    fig_resistance = px.scatter(filtered_df, x='Resistance', y=target,
                                                title=f'Resistance vs {target}', opacity=0.6)
                    st.plotly_chart(fig_resistance, use_container_width=True)
                else:
                    st.info("Need Resistance and one of TVOC/eCO2/Temperature/Humidity.")

            with col_perf2:
                fig_raw = plot_sensor_comparison(filtered_df)
                if fig_raw:
                    st.plotly_chart(fig_raw, use_container_width=True)
                else:
                    st.info("Need at least two of RawH2, RawEthanol, Resistance for comparison.")
        else:
            st.warning("No data available for the selected time range.")

# =====================================================================
# ABOUT
# =====================================================================
elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    st.markdown("""
    - **Dashboard**: upload data, see current conditions, multi-parameter overview, trends, and data quality.
    - **Analysis**: rich exploration (time series, ranges, correlations) + **Sensor Performance**.
    - The app auto-loads `./data/PSF_sensors_all.csv` if present.
    Expected format:
    ```csv
    Time,TVOC,eCO2,RawH2,RawEthanol,Temperature,Humidity,Pressure,Resistance
    09:00:00,125.5,450.2,15243,18654,22.1,58.3,1013.2,52341
    09:15:00,128.3,455.8,15189,18702,22.4,57.9,1013.0,52198
    ```
    """)

# ---------------- Sidebar dataset status & actions ----------------
if st.session_state.sensor_data is not None:
    df = st.session_state.sensor_data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current Dataset")
    st.sidebar.write(f"**Records**: {len(df):,}")
    if 'datetime' in df.columns:
        hours = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
        st.sidebar.write(f"**Time Span**: {hours:.1f} hours")
        st.sidebar.write(f"**Latest Reading**: {df['Time'].iloc[-1]}")
    else:
        st.sidebar.write(f"**Time Range**: {df['Time'].min()} - {df['Time'].max()}")

    st.sidebar.markdown("#### 📈 Ranges")
    if 'Temperature' in df.columns:
        st.sidebar.write(f"🌡️ Temp: {df['Temperature'].min():.1f} - {df['Temperature'].max():.1f}°C")
    if 'TVOC' in df.columns:
        st.sidebar.write(f"🌬️ TVOC: {df['TVOC'].min():.0f} - {df['TVOC'].max():.0f} ppb")
    if 'eCO2' in df.columns:
        st.sidebar.write(f"💨 eCO2: {df['eCO2'].min():.0f} - {df['eCO2'].max():.0f} ppm")

    # Clear + Download
    if st.sidebar.button("🗑️ Clear Data"):
        st.session_state.sensor_data = None
        st.rerun()
    st.sidebar.download_button(
        label="💾 Download CSV",
        data=df.to_csv(index=False),
        file_name=f"sensor_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**🌡️ PSF Sensor Data Analyzer v1.1**")
st.sidebar.markdown("Built with ❤️ using Streamlit")
