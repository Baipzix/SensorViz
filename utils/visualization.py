import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_time_series(df: pd.DataFrame, columns, title="Time Series Data"):
    fig = go.Figure()
    time_col = 'datetime' if 'datetime' in df.columns else 'Time'
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE']

    for i, col in enumerate(columns):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{col}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            ))

    if len(fig.data) == 0:
        return None

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_correlation_matrix(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'datetime']
    if len(numeric_cols) < 2:
        return None
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Parameter Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    fig.update_layout(height=500, xaxis_title="Parameters", yaxis_title="Parameters")
    return fig

def create_dashboard_plots(df: pd.DataFrame, params=None, ncols=3, vertical_spacing=0.22):
    """
    Build a uniform grid of <param vs time> plots with extra vertical space
    so x-axis labels don't collide with the next row's titles, and format
    x ticks to show time only (no date).
    """
    time_col = 'datetime' if 'datetime' in df.columns else 'Time'
    is_datetime = np.issubdtype(df[time_col].dtype, np.datetime64) if time_col in df.columns else False

    if params is None or len(params) == 0:
        params = [c for c in df.select_dtypes(include='number').columns if c != 'datetime']

    n = len(params)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title="No numeric parameters to plot.")
        return fig

    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=params,
        vertical_spacing=vertical_spacing,  # more open rows
        horizontal_spacing=0.07
    )

    # consistent style
    palette = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
               '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

    # Build traces
    for i, p in enumerate(params):
        r = (i // ncols) + 1
        c = (i % ncols) + 1
        if p in df.columns:
            color = palette[i % len(palette)]
            # Hover shows time only (if datetime)
            hover_x = "%{x|%H:%M:%S}" if is_datetime else "%{x}"
            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=df[p],
                    mode='lines',
                    name=p,
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertemplate=f"<b>{p}</b><br>Time: {hover_x}<br>Value: %{{y:.2f}}<extra></extra>"
                ),
                row=r, col=c
            )

    # Time-only ticks + axis titles
    for i in range(nrows * ncols):
        r = (i // ncols) + 1
        c = (i % ncols) + 1
        # Hide date part on datetime axes; for string Time, this is ignored gracefully
        fig.update_xaxes(
            title_text="Time",
            tickformat="%H:%M:%S" if is_datetime else None,
            title_standoff=6,     # keep title closer to ticks to reduce overlap
            row=r, col=c
        )
        fig.update_yaxes(title_text="Value", row=r, col=c)

    # Make subplot titles sit a bit higher to avoid crowding
    for ann in fig.layout.annotations:
        ann.font.size = 12
        ann.yshift = 10

    # Taller figure so rows are "more open"
    row_height = 290  # px per row (tweak as you like)
    fig.update_layout(
        height=max(420, nrows * row_height),
        title_text="Environmental & Gas Sensor Parameters Overview",
        title_x=0.5,
        title_font_size=16,
        margin=dict(t=60, b=30, l=10, r=10)
    )

    return fig

def plot_air_quality_timeline(df: pd.DataFrame):
    if 'TVOC' not in df.columns or 'eCO2' not in df.columns:
        return None

    fig = make_subplots(rows=2, cols=1, subplot_titles=['TVOC (ppb)', 'eCO2 (ppm)'], vertical_spacing=0.1, shared_xaxes=True)
    time_col = 'datetime' if 'datetime' in df.columns else 'Time'

    fig.add_trace(go.Scatter(x=df[time_col], y=df['TVOC'], mode='lines', name='TVOC', line=dict(color='#8E44AD', width=2), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[time_col], y=df['eCO2'], mode='lines', name='eCO2', line=dict(color='#E67E22', width=2), showlegend=False), row=2, col=1)

    # Thresholds (requires plotly >=5.0 for add_hline with row/col)
    try:
        fig.add_hline(y=220,  line_dash="dash", line_color="orange", annotation_text="Moderate Threshold", row=1, col=1)
        fig.add_hline(y=660,  line_dash="dash", line_color="red",    annotation_text="Poor Threshold",     row=1, col=1)
        fig.add_hline(y=1000, line_dash="dash", line_color="orange", annotation_text="Moderate Threshold", row=2, col=1)
        fig.add_hline(y=2000, line_dash="dash", line_color="red",    annotation_text="Poor Threshold",     row=2, col=1)
    except Exception:
        pass

    fig.update_layout(height=500, title_text="Air Quality Parameters Timeline", title_x=0.5)
    return fig

def plot_sensor_comparison(df: pd.DataFrame):
    raw_sensors = ['RawH2', 'RawEthanol', 'Resistance']
    available = [c for c in raw_sensors if c in df.columns]
    if len(available) < 2:
        return None

    df_norm = df.copy()
    for col in available:
        col_min, col_max = df[col].min(), df[col].max()
        rng = (col_max - col_min) if (col_max - col_min) != 0 else 1.0
        df_norm[f'{col}_norm'] = (df[col] - col_min) / rng

    time_col = 'datetime' if 'datetime' in df.columns else 'Time'
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    fig = go.Figure()
    for i, col in enumerate(available):
        fig.add_trace(go.Scatter(
            x=df_norm[time_col],
            y=df_norm[f'{col}_norm'],
            mode='lines',
            name=f'{col} (normalized)',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    fig.update_layout(title="Raw Sensor Readings Comparison (Normalized)", xaxis_title="Time", yaxis_title="Normalized Value (0-1)", height=400, yaxis=dict(range=[0, 1]))
    return fig

def plot_parameter_scatter(df: pd.DataFrame, x_param, y_param, color_param=None):
    if x_param not in df.columns or y_param not in df.columns:
        return None
    if color_param and color_param in df.columns:
        fig = px.scatter(df, x=x_param, y=y_param, color=color_param, title=f'{y_param} vs {x_param}', opacity=0.7, hover_data=['Time'])
    else:
        fig = px.scatter(df, x=x_param, y=y_param, title=f'{y_param} vs {x_param}', opacity=0.7, hover_data=['Time'])

    # Optional trendline (requires statsmodels)
    try:
        trend = px.scatter(df, x=x_param, y=y_param, trendline="ols").data[1]
        fig.add_trace(trend)
    except Exception:
        pass

    fig.update_layout(height=400)
    return fig

def plot_daily_patterns(df: pd.DataFrame, parameter):
    if 'datetime' not in df.columns or parameter not in df.columns:
        return None
    df_pattern = df.copy()
    df_pattern['hour'] = df_pattern['datetime'].dt.hour
    hourly_stats = df_pattern.groupby('hour')[parameter].agg(['mean', 'std', 'min', 'max']).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats['mean'], mode='lines+markers', name='Mean', line=dict(color='#3498DB', width=3)))
    fig.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats['mean'] + hourly_stats['std'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=hourly_stats['hour'], y=hourly_stats['mean'] - hourly_stats['std'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', name='±1 Std Dev', hoverinfo='skip'))
    fig.update_layout(title=f'Daily Pattern: {parameter}', xaxis_title='Hour of Day', yaxis_title=parameter, height=400, xaxis=dict(tickmode='linear', dtick=2))
    return fig

def create_summary_cards(df: pd.DataFrame):
    cards = {}
    if 'Temperature' in df.columns:
        cards['temperature'] = {'current': df['Temperature'].iloc[-1], 'avg': df['Temperature'].mean(), 'min': df['Temperature'].min(), 'max': df['Temperature'].max()}
    if 'TVOC' in df.columns:
        cards['tvoc'] = {'current': df['TVOC'].iloc[-1], 'avg': df['TVOC'].mean(), 'min': df['TVOC'].min(), 'max': df['TVOC'].max()}
    if 'eCO2' in df.columns:
        cards['eco2'] = {'current': df['eCO2'].iloc[-1], 'avg': df['eCO2'].mean(), 'min': df['eCO2'].min(), 'max': df['eCO2'].max()}
    return cards
