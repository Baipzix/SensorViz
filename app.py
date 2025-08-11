import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from utils.data_processing import validate_sensor_data, clean_sensor_data, load_sensor_data
from utils.visualization import (
    plot_time_series, 
    plot_correlation_matrix, 
    create_dashboard_plots,
    plot_air_quality_timeline,
    plot_sensor_comparison
)

# Page configuration
st.set_page_config(
    page_title="Outdoor Sensor Data Analyzer",
    page_icon="🌡️",
    layout="wide"
)

# Initialize session state
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None

# Title and description
st.title("🌡️ Outdoor Sensor Data Analyzer")
st.markdown("""
Analyze outdoor sensor data including temperature, humidity, pressure, TVOC, eCO2, and gas sensor readings.
Upload your CSV files or load the default dataset to get started.
""")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select Page", ["📁 Data Loading", "📊 Analysis", "📈 Dashboard", "🌬️ Air Quality", "ℹ️ About"])

# Main content based on selected page
if page == "📁 Data Loading":
    st.header("📁 Data Loading & Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📂 Load Default Dataset")
        st.write("Load data from `./data/PSF_sensors_all.csv`")
        
        if st.button("📊 Load PSF Sensor Data", type="primary"):
            try:
                with st.spinner("Loading sensor data..."):
                    df = load_sensor_data()
                    if df is not None:
                        st.session_state.sensor_data = df
                        st.success("✅ PSF sensor data loaded successfully!")
                        st.balloons()
                        
                        # Show data preview
                        st.subheader("📋 Data Preview")
                        st.dataframe(df.head(10))
                        
                        # Data summary
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Records", len(df))
                        with col_b:
                            st.metric("Parameters", len(df.columns) - 2)  # Exclude Time and datetime
                        with col_c:
                            if 'datetime' in df.columns:
                                time_span = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
                                st.metric("Time Span", f"{time_span:.1f} hrs")
                    else:
                        st.error("❌ Could not load data file. Please check if `./data/PSF_sensors_all.csv` exists.")
                        
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("Make sure the file `./data/PSF_sensors_all.csv` exists and has the correct format.")
    
    with col2:
        st.subheader("📤 Upload Custom CSV File")
        st.write("Upload a CSV file with your sensor data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Upload CSV with columns: Time, TVOC, eCO2, RawH2, RawEthanol, Temperature, Humidity, Pressure, Resistance"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                is_valid, message = validate_sensor_data(df)
                
                if is_valid:
                    st.success(message)
                    df_clean = clean_sensor_data(df)
                    st.session_state.sensor_data = df_clean
                    
                    st.subheader("📋 Data Preview")
                    st.dataframe(df_clean.head(10))
                    
                    # Data summary
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Records", len(df_clean))
                    with col_b:
                        st.metric("Parameters", len(df_clean.columns) - 2)
                    with col_c:
                        if 'datetime' in df_clean.columns:
                            time_span = (df_clean['datetime'].max() - df_clean['datetime'].min()).total_seconds() / 3600
                            st.metric("Time Span", f"{time_span:.1f} hrs")
                    
                else:
                    st.error(message)
                    st.info("Expected columns: Time, TVOC, eCO2, RawH2, RawEthanol, Temperature, Humidity, Pressure, Resistance")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

elif page == "📊 Analysis":
    st.header("📊 Comprehensive Data Analysis")
    
    if st.session_state.sensor_data is None:
        st.warning("⚠️ Please load data first using the 'Data Loading' page!")
        st.info("Click 'Load PSF Sensor Data' to load the default dataset.")
    else:
        df = st.session_state.sensor_data
        
        # Quick stats overview
        st.subheader("📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", f"{len(df):,}")
        with col2:
            st.metric("🌡️ Avg Temperature", f"{df['Temperature'].mean():.1f}°C")
        with col3:
            st.metric("💧 Avg Humidity", f"{df['Humidity'].mean():.1f}%")
        with col4:
            st.metric("🌬️ Avg TVOC", f"{df['TVOC'].mean():.0f} ppb")
        
        # Parameter selection and visualization
        st.subheader("📉 Time Series Visualization")
        
        # Group parameters logically
        environmental_params = ['Temperature', 'Humidity', 'Pressure']
        gas_params = ['TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Resistance']
        
        param_category = st.radio(
            "Select parameter category",
            ["Environmental", "Gas Sensors", "Custom Selection"],
            horizontal=True
        )
        
        if param_category == "Environmental":
            selected_params = environmental_params
        elif param_category == "Gas Sensors":
            selected_params = gas_params
        else:
            available_params = [col for col in df.columns if col not in ['Time', 'datetime']]
            selected_params = st.multiselect(
                "Select parameters to visualize",
                available_params,
                default=['Temperature', 'TVOC'] if len(available_params) >= 2 else available_params[:1]
            )
        
        if selected_params:
            fig = plot_time_series(df, selected_params, f"{param_category} Parameters Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time range filtering
        st.subheader("⏰ Time Range Analysis")
        col_time1, col_time2 = st.columns(2)
        
        with col_time1:
            start_time = st.time_input(
                "Start Time",
                value=datetime.strptime("00:00:00", "%H:%M:%S").time()
            )
        
        with col_time2:
            end_time = st.time_input(
                "End Time", 
                value=datetime.strptime("23:59:59", "%H:%M:%S").time()
            )
        
        # Filter data based on time range
        if start_time <= end_time:
            # Convert Time column to datetime for filtering
            df_times = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
            
            filtered_df = df[
                (df_times >= start_time) & 
                (df_times <= end_time)
            ].copy()
            
            if len(filtered_df) > 0:
                st.write(f"⏰ Showing data from {start_time} to {end_time} ({len(filtered_df)} records)")
                
                # Statistical summary
                st.subheader("📋 Statistical Summary")
                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
                st.dataframe(filtered_df[numeric_cols].describe())
                
                # Distribution analysis
                st.subheader("📊 Distribution Analysis")
                available_for_dist = [col for col in filtered_df.columns if col not in ['Time', 'datetime']]
                param_to_analyze = st.selectbox("Select parameter for distribution analysis", available_for_dist)
                
                if param_to_analyze:
                    col_hist, col_box = st.columns(2)
                    
                    with col_hist:
                        fig_hist = px.histogram(
                            filtered_df, 
                            x=param_to_analyze,
                            nbins=30,
                            title=f'Distribution of {param_to_analyze}'
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col_box:
                        fig_box = px.box(
                            filtered_df,
                            y=param_to_analyze,
                            title=f'Box Plot of {param_to_analyze}'
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                
                # Correlation analysis
                st.subheader("🔗 Parameter Correlations")
                numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 1:
                    fig_corr = plot_correlation_matrix(filtered_df)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Gas sensor specific analysis
                st.subheader("🔬 Gas Sensor Analysis")
                gas_params_available = [param for param in gas_params if param in filtered_df.columns]
                
                if gas_params_available:
                    selected_gas = st.selectbox(
                        "Select gas parameter for detailed analysis", 
                        gas_params_available,
                        key="gas_analysis"
                    )
                    
                    col_gas1, col_gas2 = st.columns(2)
                    
                    with col_gas1:
                        fig_gas_time = plot_time_series(
                            filtered_df, [selected_gas], 
                            f'{selected_gas} Over Time'
                        )
                        st.plotly_chart(fig_gas_time, use_container_width=True)
                    
                    with col_gas2:
                        fig_gas_dist = px.histogram(
                            filtered_df,
                            x=selected_gas,
                            nbins=20,
                            title=f'{selected_gas} Distribution'
                        )
                        st.plotly_chart(fig_gas_dist, use_container_width=True)
            
            else:
                st.warning("No data available for the selected time range.")
        else:
            st.error("Start time must be before or equal to end time.")

elif page == "📈 Dashboard":
    st.header("📈 Real-time Dashboard")
    
    if st.session_state.sensor_data is None:
        st.warning("⚠️ Please load data first using the 'Data Loading' page!")
    else:
        df = st.session_state.sensor_data
        
        # Current conditions (latest reading)
        st.subheader("🌡️ Current Conditions (Latest Reading)")
        latest_data = df.iloc[-1]
        
        # Environmental parameters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_delta = latest_data['Temperature'] - df['Temperature'].mean()
            st.metric(
                "🌡️ Temperature", 
                f"{latest_data['Temperature']:.1f}°C",
                delta=f"{temp_delta:.1f}°C"
            )
        
        with col2:
            humid_delta = latest_data['Humidity'] - df['Humidity'].mean()
            st.metric(
                "💧 Humidity", 
                f"{latest_data['Humidity']:.1f}%",
                delta=f"{humid_delta:.1f}%"
            )
        
        with col3:
            pressure_delta = latest_data['Pressure'] - df['Pressure'].mean()
            st.metric(
                "🌪️ Pressure", 
                f"{latest_data['Pressure']:.1f} hPa",
                delta=f"{pressure_delta:.1f} hPa"
            )
        
        with col4:
            tvoc_delta = latest_data['TVOC'] - df['TVOC'].mean()
            st.metric(
                "🌬️ TVOC", 
                f"{latest_data['TVOC']:.0f} ppb",
                delta=f"{tvoc_delta:.0f} ppb"
            )
        
        # Gas sensor readings
        st.subheader("💨 Gas Sensor Readings")
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        
        with col_g1:
            eco2_delta = latest_data['eCO2'] - df['eCO2'].mean()
            st.metric(
                "eCO2", 
                f"{latest_data['eCO2']:.0f} ppm",
                delta=f"{eco2_delta:.0f} ppm"
            )
        
        with col_g2:
            st.metric("Raw H2", f"{latest_data['RawH2']:.0f}")
        
        with col_g3:
            st.metric("Raw Ethanol", f"{latest_data['RawEthanol']:.0f}")
        
        with col_g4:
            resistance_delta = latest_data['Resistance'] - df['Resistance'].mean()
            st.metric(
                "Resistance", 
                f"{latest_data['Resistance']:.0f} Ω",
                delta=f"{resistance_delta:.0f} Ω"
            )
        
        # Dashboard overview
        st.subheader("🎛️ Multi-Parameter Overview")
        dashboard_fig = create_dashboard_plots(df)
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Recent trends
        st.subheader("⏰ Recent Trends")
        trend_records = st.selectbox(
            "Select number of recent readings", 
            [50, 100, 200, 500],
            index=1
        )
        
        recent_data = df.tail(trend_records)
        
        if len(recent_data) > 0:
            fig_recent = plot_time_series(
                recent_data, 
                ['Temperature', 'Humidity', 'Pressure', 'TVOC'],
                f"Recent Trends - Last {trend_records} readings"
            )
            st.plotly_chart(fig_recent, use_container_width=True)
        
        # Data quality indicators
        st.subheader("✅ Data Quality Metrics")
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col_q2:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Records", duplicates)
        
        with col_q3:
            total_readings = len(df)
            st.metric("Total Readings", f"{total_readings:,}")

elif page == "🌬️ Air Quality":
    st.header("🌬️ Air Quality Analysis")
    
    if st.session_state.sensor_data is None:
        st.warning("⚠️ Please load data first using the 'Data Loading' page!")
    else:
        df = st.session_state.sensor_data
        
        # Air quality overview
        st.subheader("🎯 Current Air Quality Status")
        
        col_aq1, col_aq2, col_aq3 = st.columns(3)
        
        with col_aq1:
            avg_tvoc = df['TVOC'].mean()
            latest_tvoc = df['TVOC'].iloc[-1]
            
            if avg_tvoc < 220:
                quality = "Good"
                color = "green"
            elif avg_tvoc < 660:
                quality = "Moderate"
                color = "orange"
            else:
                quality = "Poor"
                color = "red"
            
            st.markdown(f"**TVOC Level**: :{color}[{quality}]")
            st.write(f"Current: {latest_tvoc:.0f} ppb")
            st.write(f"Average: {avg_tvoc:.0f} ppb")
        
        with col_aq2:
            avg_eco2 = df['eCO2'].mean()
            latest_eco2 = df['eCO2'].iloc[-1]
            
            if avg_eco2 < 1000:
                quality = "Good"
                color = "green"
            elif avg_eco2 < 2000:
                quality = "Moderate"
                color = "orange"
            else:
                quality = "Poor"
                color = "red"
            
            st.markdown(f"**eCO2 Level**: :{color}[{quality}]")
            st.write(f"Current: {latest_eco2:.0f} ppm")
            st.write(f"Average: {avg_eco2:.0f} ppm")
        
        with col_aq3:
            # Overall air quality score
            tvoc_score = 1 if avg_tvoc < 220 else (2 if avg_tvoc < 660 else 3)
            eco2_score = 1 if avg_eco2 < 1000 else (2 if avg_eco2 < 2000 else 3)
            overall_score = (tvoc_score + eco2_score) / 2
            
            if overall_score <= 1.5:
                overall_quality = "Good"
                color = "green"
            elif overall_score <= 2.5:
                overall_quality = "Moderate"
                color = "orange"
            else:
                overall_quality = "Poor"
                color = "red"
            
            st.markdown(f"**Overall Air Quality**: :{color}[{overall_quality}]")
            st.write(f"Score: {overall_score:.1f}/3.0")
        
        # Air quality timeline
        st.subheader("📈 Air Quality Timeline")
        fig_aq_timeline = plot_air_quality_timeline(df)
        st.plotly_chart(fig_aq_timeline, use_container_width=True)
        
        # Gas sensor correlations
        st.subheader("🔗 Gas Sensor Correlations")
        gas_columns = ['TVOC', 'eCO2', 'RawH2', 'RawEthanol', 'Resistance']
        available_gas_cols = [col for col in gas_columns if col in df.columns]
        
        if len(available_gas_cols) > 1:
            fig_gas_corr = plot_correlation_matrix(df[available_gas_cols])
            st.plotly_chart(fig_gas_corr, use_container_width=True)
        
        # Sensor performance analysis
        st.subheader("⚙️ Sensor Performance")
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            # Resistance vs other parameters
            resistance_param = st.selectbox(
                "Compare Resistance with:",
                ['TVOC', 'eCO2', 'Temperature', 'Humidity'],
                key="resistance_comparison"
            )
            
            fig_resistance = px.scatter(
                df,
                x='Resistance',
                y=resistance_param,
                title=f'Resistance vs {resistance_param}',
                opacity=0.6
            )
            st.plotly_chart(fig_resistance, use_container_width=True)
        
        with col_perf2:
            # Raw sensor readings comparison
            fig_raw_sensors = plot_sensor_comparison(df)
            st.plotly_chart(fig_raw_sensors, use_container_width=True)

elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🌡️ Outdoor Sensor Data Analyzer
    
    This application analyzes environmental data from your PSF outdoor sensor system, specifically designed for TVOC, eCO2, and environmental monitoring.
    
    ### 📊 Supported Measurements
    - **Time** (HH:MM:SS) - Measurement timestamp
    - **Temperature** (°C) - Environmental temperature
    - **Humidity** (%) - Relative humidity
    - **Pressure** (hPa) - Atmospheric pressure
    - **TVOC** (ppb) - Total Volatile Organic Compounds
    - **eCO2** (ppm) - Equivalent CO2 concentration
    - **RawH2** - Raw hydrogen sensor reading
    - **RawEthanol** - Raw ethanol sensor reading
    - **Resistance** (Ω) - Sensor resistance value
    
    ### 🛠️ Application Features
    - **📁 Data Loading**: Load from `./data/PSF_sensors_all.csv` or upload custom files
    - **🧹 Data Processing**: Automatic cleaning and validation
    - **📊 Multi-view Analysis**: Environmental, gas sensors, and custom parameter views
    - **📈 Real-time Dashboard**: Current conditions with trend indicators
    - **🌬️ Air Quality Assessment**: TVOC and eCO2 quality classification
    - **🔬 Sensor Performance**: Raw sensor analysis and correlations
    - **⏰ Time-based Filtering**: Analyze specific time periods
    
    ### 📋 Data File Location
    The application looks for your data at: `./data/PSF_sensors_all.csv`
    
    Expected format:
    ```csv
    Time,TVOC,eCO2,RawH2,RawEthanol,Temperature,Humidity,Pressure,Resistance
    09:00:00,125.5,450.2,15243,18654,22.1,58.3,1013.2,52341
    09:15:00,128.3,455.8,15189,18702,22.4,57.9,1013.0,52198
    ```
    
    ### 🎯 Key Applications
    - **Environmental Monitoring**: Track outdoor air quality
    - **Sensor Validation**: Monitor raw sensor performance
    - **Research Analysis**: Study parameter correlations
    - **Quality Control**: Identify sensor drift or anomalies
    - **Compliance**: Track air quality standards
    
    ### 📊 Air Quality Reference Levels
    
    **TVOC (Total Volatile Organic Compounds)**:
    - 🟢 Good: < 220 ppb
    - 🟡 Moderate: 220-660 ppb  
    - 🔴 Poor: > 660 ppb
    
    **eCO2 (Equivalent CO2)**:
    - 🟢 Good: < 1000 ppm
    - 🟡 Moderate: 1000-2000 ppm
    - 🔴 Poor: > 2000 ppm
    
    ### ⚙️ Technical Details
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualizations**: Plotly Express & Graph Objects
    - **Statistics**: SciPy, Scikit-learn
    
    ---
    
    💡 **Getting Started**: Use the 'Data Loading' page to load your PSF sensor data and begin analysis.
    """)

# Display current data status in sidebar
if st.session_state.sensor_data is not None:
    df = st.session_state.sensor_data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current Dataset")
    st.sidebar.write(f"**Records**: {len(df):,}")
    
    if 'datetime' in df.columns:
        time_span = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
        st.sidebar.write(f"**Time Span**: {time_span:.1f} hours")
        st.sidebar.write(f"**Latest Reading**: {df['Time'].iloc[-1]}")
    else:
        st.sidebar.write(f"**Time Range**: {df['Time'].min()} - {df['Time'].max()}")
    
    # Show parameter ranges
    st.sidebar.markdown("#### 📈 Current Ranges")
    st.sidebar.write(f"🌡️ Temp: {df['Temperature'].min():.1f} - {df['Temperature'].max():.1f}°C")
    st.sidebar.write(f"🌬️ TVOC: {df['TVOC'].min():.0f} - {df['TVOC'].max():.0f} ppb")
    st.sidebar.write(f"💨 eCO2: {df['eCO2'].min():.0f} - {df['eCO2'].max():.0f} ppm")
    
    # Quick actions
    if st.sidebar.button("🗑️ Clear Data"):
        st.session_state.sensor_data = None
        st.rerun()
    
    if st.sidebar.button("📥 Download Data"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Download as CSV",
            data=csv,
            file_name=f"sensor_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🌡️ PSF Sensor Data Analyzer v1.0**")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("*Optimized for TVOC, eCO2, and environmental sensors*")
