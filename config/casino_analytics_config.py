"""
Casino Analytics Configuration for Home Page Dashboard.
Defines key casino business metrics, KPIs, and analytics queries.
"""

import os
from typing import Dict, Any, List

# Casino Business KPIs Configuration
CASINO_KPI_CONFIG = {
    'metrics': [
        # Core Business Metrics
        {
            'name': 'Total Players',
            'category': 'Core',
            'query': 'SELECT COUNT(DISTINCT user_id) FROM raw_casino_data',
            'format': '{:,}',
            'delta_query': 'SELECT COUNT(DISTINCT user_id) FROM raw_casino_data WHERE calc_date >= (SELECT MAX(calc_date) - INTERVAL \'30 days\' FROM raw_casino_data)',
            'icon': '👥',
            'description': 'Total unique players in database'
        },
        {
            'name': 'Total Revenue (Bets)',
            'category': 'Core', 
            'query': 'SELECT SUM(bet) FROM raw_casino_data WHERE bet > 0',
            'format': '${:,.0f}',
            'delta_query': 'SELECT SUM(bet) FROM raw_casino_data WHERE bet > 0 AND calc_date >= (SELECT MAX(calc_date) - INTERVAL \'30 days\' FROM raw_casino_data)',
            'icon': '💰',
            'description': 'Total betting volume'
        },
        {
            'name': 'House Edge',
            'category': 'Core',
            'query': 'SELECT (SUM(profit) / SUM(bet)) * 100 FROM raw_casino_data WHERE bet > 0',
            'format': '{:.2f}%',
            'delta_query': 'SELECT (SUM(profit) / SUM(bet)) * 100 FROM raw_casino_data WHERE bet > 0 AND calc_date >= (SELECT MAX(calc_date) - INTERVAL \'30 days\' FROM raw_casino_data)',
            'icon': '🏆',
            'description': 'Casino profit margin percentage'
        },
        
        # Player Analytics Metrics  
        {
            'name': 'Player Churn Rate (7d)',
            'category': 'Player Analytics',
            'query': '''SELECT 
                CAST(
                    (SUM(CASE WHEN is_churned_7d = 1 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 
                    AS DECIMAL(10,2)
                ) 
                FROM player_churn_indicators''',
            'format': '{:.1f}%',
            'delta_query': '''SELECT 
                CAST(
                    (SUM(CASE WHEN is_churned_14d = 1 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 
                    AS DECIMAL(10,2)
                ) 
                FROM player_churn_indicators''',
            'icon': '📉',
            'description': 'Percentage of players who stopped playing in last 7 days'
        },
        {
            'name': 'Player Retention Rate (7d)',
            'category': 'Player Analytics',
            'query': '''SELECT 
                CAST(
                    (SUM(CASE WHEN is_churned_7d = 0 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 
                    AS DECIMAL(10,2)
                ) 
                FROM player_churn_indicators''',
            'format': '{:.1f}%',
            'delta_query': '''SELECT 
                CAST(
                    (SUM(CASE WHEN is_churned_14d = 0 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 
                    AS DECIMAL(10,2)
                ) 
                FROM player_churn_indicators''',
            'icon': '🎯',
            'description': 'Percentage of players still active in last 7 days'
        },
        {
            'name': 'Average Revenue Per User (ARPU)',
            'category': 'Player Analytics',
            'query': '''SELECT 
                CAST(SUM(bet) / COUNT(DISTINCT user_id) AS DECIMAL(15,2)) 
                FROM raw_casino_data WHERE bet > 0''',
            'format': '${:,.2f}',
            'delta_query': '''SELECT 
                CAST(SUM(bet) / COUNT(DISTINCT user_id) AS DECIMAL(15,2)) 
                FROM raw_casino_data WHERE bet > 0 AND calc_date >= (SELECT MAX(calc_date) - INTERVAL \'30 days\' FROM raw_casino_data)''',
            'icon': '💵',
            'description': 'Average revenue generated per player'
        },
        {
            'name': 'Customer Lifetime Value (CLV)',
            'category': 'Player Analytics',
            'query': '''SELECT 
                CAST(AVG(total_bet) AS DECIMAL(15,2)) 
                FROM player_features WHERE total_bet > 0''',
            'format': '${:,.2f}',
            'delta_query': '''SELECT 
                CAST(AVG(total_profit) AS DECIMAL(15,2)) 
                FROM player_features''',
            'icon': '💎',
            'description': 'Average lifetime value per player'
        },
        
        # Risk & Fraud Metrics
        {
            'name': 'High-Risk Players',
            'category': 'Risk Management',
            'query': '''SELECT 
                COUNT(*) 
                FROM player_churn_indicators 
                WHERE player_segment = 'High Value' AND is_churned_7d = 1''',
            'format': '{:,}',
            'delta_query': '''SELECT 
                COUNT(*) 
                FROM player_churn_indicators 
                WHERE player_segment = 'High Value' AND is_churned_14d = 1''',
            'icon': '⚠️',
            'description': 'High-value players at risk of churning'
        },
        {
            'name': 'Anomaly Detection Rate',
            'category': 'Risk Management',
            'query': '''SELECT 
                CAST(
                    (COUNT(CASE WHEN bet = 0 THEN 1 END)::float / COUNT(*)) * 100 
                    AS DECIMAL(10,2)
                ) 
                FROM raw_casino_data''',
            'format': '{:.1f}%',
            'delta_query': '''SELECT 
                CAST(
                    (COUNT(CASE WHEN bet = 0 THEN 1 END)::float / COUNT(*)) * 100 
                    AS DECIMAL(10,2)
                ) 
                FROM raw_casino_data 
                WHERE calc_date >= (SELECT MAX(calc_date) - INTERVAL '30 days' FROM raw_casino_data)''',
            'icon': '🔍',
            'description': 'Percentage of transactions flagged as anomalous (zero-bet transactions)'
        }
    ]
}

# Game Performance Analytics
GAME_ANALYTICS_CONFIG = {
    'queries': {
        'game_performance': '''
            SELECT 
                game_type,
                COUNT(*) as total_transactions,
                COUNT(DISTINCT user_id) as unique_players,
                SUM(bet) as total_bets,
                SUM(won) as total_winnings,
                SUM(profit) as casino_profit,
                CAST(AVG(bet) AS DECIMAL(10,2)) as avg_bet_size,
                CAST((SUM(profit) / NULLIF(SUM(bet), 0)) * 100 AS DECIMAL(10,2)) as house_edge_pct,
                CAST(SUM(bet) / COUNT(DISTINCT user_id) AS DECIMAL(15,2)) as arpu_by_game
            FROM raw_casino_data 
            WHERE bet > 0
            GROUP BY game_type
            ORDER BY total_bets DESC
        ''',
        
        'daily_trends': '''
            SELECT 
                calc_date,
                COUNT(*) as transactions,
                COUNT(DISTINCT user_id) as active_players,
                SUM(bet) as daily_revenue,
                SUM(won) as daily_winnings,
                SUM(profit) as daily_profit,
                CAST(AVG(bet) AS DECIMAL(10,2)) as avg_bet_size
            FROM raw_casino_data 
            WHERE bet > 0 
            AND calc_date >= (SELECT MAX(calc_date) - INTERVAL '30 days' FROM raw_casino_data)
            GROUP BY calc_date
            ORDER BY calc_date DESC
        ''',
        
        'player_segments_analysis': '''
            SELECT 
                player_segment,
                COUNT(*) as player_count,
                CAST(AVG(total_bet) AS DECIMAL(15,2)) as avg_total_bet,
                CAST(AVG(total_won) AS DECIMAL(15,2)) as avg_total_won,
                CAST(AVG(total_profit) AS DECIMAL(15,2)) as avg_profit_per_player,
                CAST(AVG(total_transactions) AS DECIMAL(10,0)) as avg_transactions,
                CAST(AVG(days_active) AS DECIMAL(10,1)) as avg_days_active,
                CAST(
                    (SUM(CASE WHEN is_churned_7d = 1 THEN 1 ELSE 0 END)::float / COUNT(*)) * 100 
                    AS DECIMAL(10,1)
                ) as churn_rate_pct
            FROM player_churn_indicators
            GROUP BY player_segment
            ORDER BY avg_total_bet DESC
        '''
    }
}

# Chart Configuration for Casino Analytics
CASINO_CHART_CONFIG = {
    'player_churn_risk': {
        'title': 'Player Churn Risk Distribution',
        'chart_type': 'pie',
        'colors': {
            'Low Risk': '#2ca02c',
            'Medium Risk': '#ff7f0e',
            'High Risk': '#d62728'
        }
    },
    'game_performance': {
        'title': 'Game Performance by Revenue',
        'chart_type': 'bar',
        'x_axis': 'Game Type',
        'y_axis': 'Total Bets ($)',
        'color_scale': 'viridis'
    },
    'daily_revenue_trends': {
        'title': 'Daily Revenue Trends (Last 30 Days)',
        'chart_type': 'line',
        'x_axis': 'Date',
        'y_axis': 'Revenue ($)',
        'line_color': '#1f77b4'
    },
    'player_value_segments': {
        'title': 'Player Value Segmentation',
        'chart_type': 'bar',
        'x_axis': 'Player Segment',
        'y_axis': 'Average CLV ($)',
        'color_column': 'churn_rate_pct'
    }
}

# Data Quality Monitoring
DATA_QUALITY_CONFIG = {
    'checks': [
        {
            'name': 'Zero Bet Transactions',
            'query': 'SELECT COUNT(*) FROM raw_casino_data WHERE bet = 0',
            'threshold': 1000000,  # Alert if more than 1M zero bets
            'severity': 'warning'
        },
        {
            'name': 'Negative Profits',
            'query': 'SELECT COUNT(*) FROM raw_casino_data WHERE profit < -1000',
            'threshold': 100000,   # Alert if more than 100K large losses
            'severity': 'critical'
        },
        {
            'name': 'Data Freshness',
            'query': 'SELECT MAX(calc_date) FROM raw_casino_data',
            'threshold': 30,       # Alert if data older than 30 days
            'severity': 'warning'
        }
    ]
}

# Dashboard Layout Configuration
CASINO_LAYOUT_CONFIG = {
    'home_page_sections': [
        {
            'section': 'Key Performance Indicators',
            'metrics': ['Total Players', 'Total Revenue (Bets)', 'House Edge', 'Player Churn Rate (7d)'],
            'layout': 'columns',
            'columns': 4
        },
        {
            'section': 'Player Analytics',
            'metrics': ['Player Retention Rate (7d)', 'Average Revenue Per User (ARPU)', 'Customer Lifetime Value (CLV)'],
            'layout': 'columns', 
            'columns': 3
        },
        {
            'section': 'Risk Management',
            'metrics': ['High-Risk Players', 'Anomaly Detection Rate'],
            'layout': 'columns',
            'columns': 2
        },
        {
            'section': 'Game Performance Analytics',
            'charts': ['game_performance', 'daily_revenue_trends'],
            'layout': 'rows'
        },
        {
            'section': 'Player Behavior Analytics', 
            'charts': ['player_churn_risk', 'player_value_segments'],
            'layout': 'columns',
            'columns': 2
        }
    ]
}

# Export configuration
__all__ = [
    'CASINO_KPI_CONFIG',
    'GAME_ANALYTICS_CONFIG', 
    'CASINO_CHART_CONFIG',
    'DATA_QUALITY_CONFIG',
    'CASINO_LAYOUT_CONFIG'
] 