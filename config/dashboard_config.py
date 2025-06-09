"""
Dashboard configuration settings for the Casino Intelligence Hub.
Contains Streamlit and Power BI settings, chart configurations, and UI parameters.
"""

import os
from typing import Dict, Any, List

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'Casino Intelligence Hub',
    'page_icon': '🎰',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'port': int(os.getenv('STREAMLIT_PORT', 8501))
}

# Dashboard Layout Configuration
LAYOUT_CONFIG = {
    'sidebar_width': 300,
    'main_container_padding': 1,
    'chart_height': 400,
    'metric_container_height': 150,
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'warning': '#ff9800',
        'danger': '#d62728',
        'info': '#17a2b8'
    }
}

# Chart Configuration
CHART_CONFIG = {
    'churn_prediction': {
        'title': 'Player Churn Risk Distribution',
        'x_axis': 'Churn Probability',
        'y_axis': 'Number of Players',
        'color_scale': ['green', 'yellow', 'red']
    },
    'player_segments': {
        'title': 'Player Segments Distribution',
        'chart_type': 'pie',
        'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    },
    'revenue_trends': {
        'title': 'Revenue Trends Over Time',
        'x_axis': 'Date',
        'y_axis': 'Revenue ($)',
        'line_color': '#1f77b4'
    },
    'anomaly_detection': {
        'title': 'Anomaly Detection Results',
        'normal_color': '#2ca02c',
        'anomaly_color': '#d62728'
    }
}

# Import comprehensive casino analytics configuration
try:
    from .casino_analytics_config import CASINO_KPI_CONFIG
    KPI_CONFIG = CASINO_KPI_CONFIG
except ImportError:
    # Fallback to basic KPIs if casino analytics config not available
    KPI_CONFIG = {
        'metrics': [
            {
                'name': 'Total Players',
                'query': 'SELECT COUNT(DISTINCT user_id) FROM raw_casino_data',
                'format': '{:,}',
                'delta_query': 'SELECT COUNT(DISTINCT user_id) FROM raw_casino_data WHERE calc_date >= CURRENT_DATE - INTERVAL \'30 days\'',
                'icon': '👥'
            },
            {
                'name': 'Total Revenue (Bets)',
                'query': 'SELECT SUM(bet) FROM raw_casino_data WHERE bet > 0',
                'format': '${:,.0f}',
                'delta_query': 'SELECT SUM(bet) FROM raw_casino_data WHERE bet > 0 AND calc_date >= CURRENT_DATE - INTERVAL \'30 days\'',
                'icon': '💰'
            },
            {
                'name': 'House Edge',
                'query': 'SELECT (SUM(profit) / NULLIF(SUM(bet), 0)) * 100 FROM raw_casino_data WHERE bet > 0',
                'format': '{:.2f}%',
                'delta_query': 'SELECT (SUM(profit) / NULLIF(SUM(bet), 0)) * 100 FROM raw_casino_data WHERE bet > 0 AND calc_date >= CURRENT_DATE - INTERVAL \'30 days\'',
                'icon': '🏆'
            },
            {
                'name': 'Player Churn Rate (7d)',
                'query': 'SELECT ROUND((SUM(CASE WHEN is_churned_7d = 1 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100, 2) FROM player_churn_indicators',
                'format': '{:.1f}%',
                'delta_query': 'SELECT ROUND((SUM(CASE WHEN is_churned_14d = 1 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100, 2) FROM player_churn_indicators',
                'icon': '📉'
            }
        ]
    }

# Power BI Configuration
POWER_BI_CONFIG = {
    'data_source': {
        'server': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'casino_intelligence'),
        'connection_timeout': 30,
        'command_timeout': 300
    },
    'refresh_schedule': {
        'enabled': True,
        'frequency': 'hourly',
        'times': ['06:00', '12:00', '18:00']
    }
}

# Filter Configuration
FILTER_CONFIG = {
    'date_ranges': [
        {'label': 'Last 7 Days', 'days': 7},
        {'label': 'Last 30 Days', 'days': 30},
        {'label': 'Last 90 Days', 'days': 90},
        {'label': 'Last Year', 'days': 365}
    ],
    'player_segments': [
        'Champions',
        'Loyal Customers', 
        'Potential Loyalists',
        'New Customers',
        'At Risk',
        'Cannot Lose Them',
        'Hibernating',
        'Lost'
    ],
    'game_types': [
        'Slots',
        'Blackjack',
        'Roulette',
        'Poker',
        'Baccarat',
        'Other'
    ]
}

# Data Refresh Configuration
REFRESH_CONFIG = {
    'auto_refresh': True,
    'refresh_interval_seconds': 300,  # 5 minutes
    'cache_ttl_seconds': 600,         # 10 minutes
    'max_rows_display': 1000
} 