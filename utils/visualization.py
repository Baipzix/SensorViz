import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

def plot_time_series(df, columns, title="Time Series Data"):
    """Create interactive time series plot using datetime column"""
    fig = go.Figure()
    
    # Use 'datetime' column for x-axis if available, otherwise use Time
    time_col = 'datetime' if 'datetime' in df.columns else 'Time'
    
    # Color palette for multiple lines
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    for i, col in enumerate(columns):
        if col in df.columns:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{col}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_correlation_matrix(df):
    """Create correlation heatmap for numeric columns"""
    # Select only numeric columns, excluding datetime
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['datetime']]
    
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Parameter Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="Parameters",
        yaxis_title="Parameters"
    )
    
    return fig

def create_dashboard_plots(df):
    """Create comprehensive dashboard with environmental and gas sensor data"""
    
    # Create subplot titles for your specific parameters
    titles = ['Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)', 'TVOC (ppb)']
    
    # Time series for all main parameters
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Use datetime column for x-axis
    time_col = 'datetime' if 'datetime' in df.columns else 'Time'
    
    # Add traces for your specific parameters
    fig.add_trace(
        go.Scatter(
            x=df[time_col], 
            y=df['Temperature'], 
            name='Temperature', 
            line=dict(color='#FF6B6B', width=2),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df[time_col], 
            y=df['Humidity'], 
            name='Humidity', 
            line=dict(color='#4ECDC4', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df[time_col], 
            y=df['Pressure'], 
            name='Pressure', 
            line=dict(color='#45B7D1', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    if 'TVOC' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[time_col], 
                y=df['TVOC'], 
                name='TVOC', 
                line=dict(color='#96CEB4', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600, 
        title_text="Environmental Sensor Parameters Overview",
        title_x=0.5,
        title_font_size=16
    )
    
    return fig

def plot_air_quality_timeline(df):
    """Create air quality timeline with TVOC and eCO2"""
    if 'TVOC' not in df.columns or 'eCO2' not in df.columns:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['TVOC (ppb)', 'eCO2 (ppm)'],
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    time_col = 'datetime' if 'datetime' in df.columns else 'Time'
    
    # TVOC plot with quality zones
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df['TVOC'],
            mode='lines',
            name='TVOC',
            line=dict(color='#8E44AD', width=2),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add TVOC quality threshold lines
    fig.add_hline(y=220, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Threshold", row=1, col=1)
    fig.add_hline(y=660, line_dash="dash", line_color="red", 
                  annotation_text="Poor Threshold", row=1, col=1)
    
    # eCO2 plot with quality zones
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df['eCO2'],
            mode='lines',
            name='eCO2',
            line=dict(color='#E67E22', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add eCO2 quality threshold lines
    fig.add_hline(y=1000, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Threshold", row=2, col=1)
    fig.add_hline(y=2000, line_dash="dash", line_color="red", 
                  annotation_text="Poor Threshold", row=2, col=1)
    
    fig.update_layout(
        height=500,
        title_text="Air Quality Parameters Timeline",
        title_x=0.5
    )
    
    return fig

def plot_sensor_comparison(df):
    """Create comparison plot for raw sensor readings"""
    raw_sensors = ['RawH2', 'RawEthanol', 'Resistance']
    available_sensors = [col for col in raw_sensors if col in df.columns]
    
    if len(available_sensors) < 2:
        return None
    
    # Normalize data for comparison (0-1 scale)
    df_norm = df.copy()
    for col in available_sensors:
        col_min = df[col].min()
        col_max = df[col].max()
        df_norm[f'{col}_norm'] = (df[col] - col_min) / (col_max - col_min)
    
    fig = go.Figure()
    
    time_col = 'datetime' if 'datetime' in df.columns else 'Time'
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    for i, col in enumerate(available_sensors):
        fig.add_trace(go.Scatter(
            x=df_norm[time_col],
            y=df_norm[f'{col}_norm'],
            mode='lines',
            name=f'{col} (normalized)',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title="Raw Sensor Readings Comparison (Normalized)",
        xaxis_title="Time",
        yaxis_title="Normalized Value (0-1)",
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def plot_parameter_scatter(df, x_param, y_param, color_param=None):
    """Create scatter plot between two parameters"""
    if x_param not in df.columns or y_param not in df.columns:
        return None
    
    if color_param and color_param in df.columns:
        fig = px.scatter(
            df,
            x=x_param,
            y=y_param,
            color=color_param,
            title=f'{y_param} vs {x_param}',
            opacity=0.7,
            hover_data=['Time']
        )
    else:
        fig = px.scatter(
            df,
            x=x_param,
            y=y_param,
            title=f'{y_param} vs {x_param}',
            opacity=0.7,
            hover_data=['Time']
        )
    
    # Add trend line
    fig.add_trace(
        px.scatter(df, x=x_param, y=y_param, trendline="ols").data[1]
    )
    
    fig.update_layout(height=400)
    
    return fig

def plot_daily_patterns(df, parameter):
    """Plot daily patterns for a specific parameter"""
    if 'datetime' not in df.columns or parameter not in df.columns:
        return None
    
    # Extract hour from datetime
    df_pattern = df.copy()
    df_pattern['hour'] = df_pattern['datetime'].dt.hour
    
    # Group by hour and calculate statistics
    hourly_stats = df_pattern.groupby('hour')[parameter].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['mean'],
        mode='lines+markers',
        name='Mean',
        line=dict(color='#3498DB', width=3)
    ))
    
    # Add confidence band
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['mean'] + hourly_stats['std'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['mean'] - hourly_stats['std'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)',
        name='±1 Std Dev',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Daily Pattern: {parameter}',
        xaxis_title='Hour of Day',
        yaxis_title=parameter,
        height=400,
        xaxis=dict(tickmode='linear', dtick=2)
    )
    
    return fig

def create_summary_cards(df):
    """Create summary statistics for dashboard cards"""
    cards = {}
    
    if 'Temperature' in df.columns:
        cards['temperature'] = {
            'current': df['Temperature'].iloc[-1],
            'avg': df['Temperature'].mean(),
            'min': df['Temperature'].min(),
            'max': df['Temperature'].max()
        }
    
    if 'TVOC' in df.columns:
        cards['tvoc'] = {
            'current': df['TVOC'].iloc[-1],
            'avg': df['TVOC'].mean(),
            'min': df['TVOC'].min(),
            'max': df['TVOC'].max()
        }
    
    if 'eCO2' in df.columns:
        cards['eco2'] = {
            'current': df['eCO2'].iloc[-1],
            'avg': df['eCO2'].mean(),
            'min': df['eCO2'].min(),
            'max': df['eCO2'].max()
        }
    
    return cards