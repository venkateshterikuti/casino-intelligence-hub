"""
Main Streamlit Dashboard for Casino Intelligence Hub.
Provides interactive analytics and visualizations for player behavior analysis.
"""

import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.dashboard_config import STREAMLIT_CONFIG, LAYOUT_CONFIG, KPI_CONFIG
from src.database.connection import db_manager
from src.utils.logger import get_logger

# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=STREAMLIT_CONFIG['initial_sidebar_state']
)

logger = get_logger(__name__)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stAlert {
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .main-header {
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CasinoDashboard:
    """Main dashboard class for Casino Intelligence Hub."""
    
    def __init__(self):
        self.db_manager = db_manager
        
    def _get_time_condition(self, time_period: str) -> str:
        """Generate SQL time condition based on selected period."""
        if time_period == "Last 7 Days":
            return "r.calc_date >= (SELECT MAX(calc_date) - INTERVAL '7 days' FROM raw_casino_data)"
        elif time_period == "Last 30 Days":
            return "r.calc_date >= (SELECT MAX(calc_date) - INTERVAL '30 days' FROM raw_casino_data)"
        elif time_period == "Last 90 Days":
            return "r.calc_date >= (SELECT MAX(calc_date) - INTERVAL '90 days' FROM raw_casino_data)"
        else:  # "All Time"
            return "1=1"  # Always true condition
            
    def _apply_time_filter_to_kpi(self, query: str, time_condition: str, metric_name: str) -> str:
        """Apply time filtering to KPI queries based on the metric type."""
        # Convert to lowercase for easier matching
        query_lower = query.lower()
        
        # Skip time filtering for certain metrics that don't make sense with time periods
        skip_time_filter = ['customer lifetime value', 'clv', 'player churn rate', 'player retention rate', 'high-risk players', 'anomaly detection rate']
        if any(skip in metric_name.lower() for skip in skip_time_filter):
            return query
            
        # If query references raw_casino_data, add time filter
        if 'raw_casino_data' in query_lower:
            # Use simple time condition without alias complications
            simple_time_condition = time_condition.replace('r.calc_date', 'calc_date')
            
            # Properly handle WHERE clause
            if 'where' in query_lower:
                # Find WHERE position and add our condition
                where_pos = query_lower.find('where')
                # Get the existing WHERE condition
                after_where = query[where_pos + 5:].strip()
                before_where = query[:where_pos].strip()
                return f"{before_where} WHERE {simple_time_condition} AND ({after_where})"
            else:
                # Add WHERE clause - find insertion point
                insertion_keywords = ['group by', 'order by', 'limit', 'having']
                for keyword in insertion_keywords:
                    if keyword in query_lower:
                        pos = query_lower.find(keyword)
                        before_keyword = query[:pos].strip()
                        after_keyword = query[pos:]
                        return f"{before_keyword} WHERE {simple_time_condition} {after_keyword}"
                
                # No special keywords found, add at the end
                return f"{query.strip()} WHERE {simple_time_condition}"
        
        return query
        
    def load_kpi_data(self, time_period: str = 'All Time') -> dict:
        """Load KPI data from database using new casino analytics with time filtering."""
        kpi_data = {}
        time_condition = self._get_time_condition(time_period)
        
        try:
            # Try to use new casino analytics config
            try:
                from config.casino_analytics_config import CASINO_KPI_CONFIG
                metrics = CASINO_KPI_CONFIG['metrics']
                logger.info("✅ Using comprehensive casino analytics config")
            except ImportError:
                # Fallback to old config
                metrics = KPI_CONFIG['metrics']
                logger.info("⚠️ Using fallback KPI config")
                
            for metric in metrics:
                try:
                    # Apply time filtering to queries
                    modified_query = self._apply_time_filter_to_kpi(metric['query'], time_condition, metric['name'])
                    
                    # Execute main query
                    result = self.db_manager.execute_query(modified_query)
                    if not result.empty:
                        value = result.iloc[0, 0]
                        if value is not None:
                            kpi_data[metric['name']] = {
                                'value': value,
                                'format': metric['format'],
                                'icon': metric['icon'],
                                'category': metric.get('category', 'General'),
                                'description': metric.get('description', '') + f' ({time_period})'
                            }
                            
                            # Execute delta query if available (also time-filtered)
                            if 'delta_query' in metric:
                                try:
                                    modified_delta_query = self._apply_time_filter_to_kpi(metric['delta_query'], time_condition, metric['name'])
                                    delta_result = self.db_manager.execute_query(modified_delta_query)
                                    if not delta_result.empty:
                                        current_value = value
                                        previous_value = delta_result.iloc[0, 0]
                                        if previous_value and previous_value != 0:
                                            delta = ((current_value - previous_value) / previous_value) * 100
                                            kpi_data[metric['name']]['delta'] = delta
                                except Exception as e:
                                    logger.warning(f"Failed to calculate delta for {metric['name']}: {e}")
                except Exception as e:
                    logger.error(f"Failed to load KPI {metric['name']}: {e}")
                            
        except Exception as e:
            logger.error(f"Failed to load KPI data: {e}")
            st.error("Failed to load KPI data. Please check database connection.")
            
        return kpi_data
    
    def display_kpis(self, kpi_data: dict):
        """Display KPIs in the dashboard organized by category."""
        if not kpi_data:
            st.warning("No KPI data available")
            return
        
        # Group KPIs by category
        categories = {}
        for metric_name, data in kpi_data.items():
            category = data.get('category', 'General')
            if category not in categories:
                categories[category] = []
            categories[category].append((metric_name, data))
        
        # Display each category
        for category, metrics in categories.items():
            st.markdown(f"### 📊 {category} Metrics")
            
            # Create columns for metrics in this category
            cols = st.columns(len(metrics))
            
            for i, (metric_name, data) in enumerate(metrics):
                with cols[i]:
                    # Format value with None check
                    value = data.get('value', 0)
                    if value is None:
                        value = 0
                    
                    try:
                        formatted_value = data['format'].format(value)
                    except:
                        formatted_value = str(value)
                    
                    # Calculate delta if available
                    delta = data.get('delta')
                    delta_str = f"{delta:+.1f}%" if delta is not None else None
                    
                    # Display metric
                    st.metric(
                        label=f"{data['icon']} {metric_name}",
                        value=formatted_value,
                        delta=delta_str,
                        help=data.get('description', '')
                    )
            
            st.markdown("---")
    
    def load_churn_data(self, time_period: str = 'All Time') -> pd.DataFrame:
        """Load churn prediction data from player_churn_indicators with time filtering."""
        time_condition = self._get_time_condition(time_period)
        
        # For churn data, we filter by players who were active in the time period
        query = f"""
            SELECT 
                CASE 
                    WHEN is_churned_7d = 1 THEN 'High Risk'
                    WHEN is_churned_14d = 1 THEN 'Medium Risk'
                    ELSE 'Low Risk'
                END as risk_category,
                COUNT(*) as player_count
            FROM player_churn_indicators p
            WHERE EXISTS (
                SELECT 1 FROM raw_casino_data r 
                WHERE r.user_id = p.user_id 
                AND r.bet > 0 
                AND {time_condition}
            )
            GROUP BY 
                CASE 
                    WHEN is_churned_7d = 1 THEN 'High Risk'
                    WHEN is_churned_14d = 1 THEN 'Medium Risk'
                    ELSE 'Low Risk'
                END
            ORDER BY 
                CASE 
                    CASE 
                        WHEN is_churned_7d = 1 THEN 'High Risk'
                        WHEN is_churned_14d = 1 THEN 'Medium Risk'
                        ELSE 'Low Risk'
                    END 
                    WHEN 'High Risk' THEN 1 
                    WHEN 'Medium Risk' THEN 2 
                    ELSE 3 
                END
        """
        
        try:
            return self.db_manager.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to load churn data: {e}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['risk_category', 'player_count'])
    
    def load_segment_data(self, time_period: str = 'All Time') -> pd.DataFrame:
        """Load player segmentation data from player_churn_indicators with time filtering."""
        time_condition = self._get_time_condition(time_period)
        
        # For segment data, we filter by players who were active in the time period
        query = f"""
            SELECT 
                player_segment as rfm_segment,
                COUNT(*) as player_count,
                CAST(AVG(total_profit) AS DECIMAL(15,2)) as avg_clv,
                CAST(AVG(days_active) AS DECIMAL(10,1)) as avg_recency,
                CAST(AVG(total_transactions) AS DECIMAL(10,0)) as avg_frequency,
                CAST(AVG(total_bet) AS DECIMAL(15,2)) as avg_monetary
            FROM player_churn_indicators p
            WHERE EXISTS (
                SELECT 1 FROM raw_casino_data r 
                WHERE r.user_id = p.user_id 
                AND r.bet > 0 
                AND {time_condition}
            )
            GROUP BY player_segment
            ORDER BY player_count DESC
        """
        
        try:
            return self.db_manager.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to load segment data: {e}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['rfm_segment', 'player_count', 'avg_clv', 'avg_recency', 'avg_frequency', 'avg_monetary'])
    
    def load_revenue_trends(self, player_segment: str = 'All', time_period: str = 'All Time') -> pd.DataFrame:
        """Load revenue trend data from raw_casino_data with segment and time filtering."""
        # Get time period filter
        time_condition = self._get_time_condition(time_period)
        
        # Use actual date range from data instead of CURRENT_DATE
        base_query = """
            SELECT 
                r.calc_date as date,
                SUM(r.bet) as daily_revenue,
                COUNT(DISTINCT r.user_id) as active_players,
                AVG(r.bet) as avg_bet_size
            FROM raw_casino_data r
        """
        
        # Add segment filtering if specified
        if player_segment != 'All':
            base_query += """
            JOIN player_churn_indicators p ON r.user_id = p.user_id
            WHERE r.bet > 0 AND p.player_segment = :segment AND """ + time_condition
            params = {'segment': player_segment}
        else:
            base_query += """
            WHERE r.bet > 0 AND """ + time_condition
            params = {}
        
        query = base_query + """
            GROUP BY r.calc_date
            ORDER BY r.calc_date
        """
        
        try:
            return self.db_manager.execute_query(query, params)
        except Exception as e:
            logger.error(f"Failed to load revenue trends: {e}")
            # Return empty DataFrame with correct structure  
            return pd.DataFrame(columns=['date', 'daily_revenue', 'active_players', 'avg_bet_size'])
    
    def load_top_games(self, player_segment: str = 'All', time_period: str = 'All Time') -> pd.DataFrame:
        """Load top games by revenue with segment and time filtering."""
        # Get time period filter
        time_condition = self._get_time_condition(time_period)
        
        base_query = """
            SELECT 
                r.game_type,
                SUM(r.bet) as total_revenue,
                COUNT(*) as total_transactions,
                COUNT(DISTINCT r.user_id) as unique_players,
                CAST(AVG(r.bet) AS DECIMAL(10,2)) as avg_bet_size,
                CAST((SUM(r.profit) / NULLIF(SUM(r.bet), 0)) * 100 AS DECIMAL(5,2)) as house_edge_pct
            FROM raw_casino_data r
        """
        
        # Add segment filtering if specified
        if player_segment != 'All':
            base_query += """
            JOIN player_churn_indicators p ON r.user_id = p.user_id
            WHERE r.bet > 0 AND p.player_segment = :segment AND """ + time_condition
            params = {'segment': player_segment}
        else:
            base_query += """
            WHERE r.bet > 0 AND """ + time_condition
            params = {}
        
        query = base_query + """
            GROUP BY r.game_type
            ORDER BY total_revenue DESC
            LIMIT 10
        """
        
        try:
            return self.db_manager.execute_query(query, params)
        except Exception as e:
            logger.error(f"Failed to load top games: {e}")
            return pd.DataFrame(columns=['game_type', 'total_revenue', 'total_transactions', 'unique_players', 'avg_bet_size', 'house_edge_pct'])
    
    def load_at_risk_players(self) -> pd.DataFrame:
        """Load detailed at-risk player data."""
        query = """
            SELECT 
                user_id,
                CAST(total_bet AS DECIMAL(15,2)) as total_revenue,
                CAST(total_profit AS DECIMAL(15,2)) as total_profit,
                total_transactions,
                days_active,
                last_activity_date,
                player_segment,
                CASE 
                    WHEN is_churned_7d = 1 THEN 'High Risk - 7d Inactive'
                    WHEN is_churned_14d = 1 THEN 'Medium Risk - 14d Inactive'
                    ELSE 'Active'
                END as risk_level
            FROM player_churn_indicators
            WHERE (is_churned_7d = 1 OR is_churned_14d = 1) 
                AND player_segment = 'High Value'
            ORDER BY total_bet DESC
            LIMIT 100
        """
        
        try:
            return self.db_manager.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to load at-risk players: {e}")
            return pd.DataFrame(columns=['user_id', 'total_revenue', 'total_profit', 'total_transactions', 'days_active', 'last_activity_date', 'player_segment', 'risk_level'])
    
    def create_churn_chart(self, churn_data: pd.DataFrame, time_period: str = 'All Time'):
        """Create churn risk distribution chart."""
        if churn_data.empty:
            st.warning("No churn data available")
            return
            
        # Group by risk category
        risk_summary = churn_data.groupby('risk_category')['player_count'].sum().reset_index()
        
        # Create pie chart
        fig = px.pie(
            risk_summary,
            values='player_count',
            names='risk_category',
            title=f'Player Churn Risk Distribution ({time_period})',
            color_discrete_map={
                'Low Risk': '#2ca02c',
                'Medium Risk': '#ff7f0e', 
                'High Risk': '#d62728'
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_segment_chart(self, segment_data: pd.DataFrame, time_period: str = 'All Time'):
        """Create player segmentation chart."""
        if segment_data.empty:
            st.warning("No segmentation data available")
            return
            
        # Create bar chart
        fig = px.bar(
            segment_data,
            x='rfm_segment',
            y='player_count',
            title=f'Player Segments Distribution ({time_period})',
            color='avg_clv',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            xaxis_title='Player Segment',
            yaxis_title='Number of Players'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_revenue_chart(self, revenue_data: pd.DataFrame, segment: str = 'All', time_period: str = 'All Time'):
        """Create revenue trends chart."""
        if revenue_data.empty:
            st.warning("No revenue data available (check if historical data exists)")
            return
            
        # Create line chart with dual y-axis
        fig = go.Figure()
        
        # Daily revenue (primary y-axis)
        fig.add_trace(go.Scatter(
            x=revenue_data['date'],
            y=revenue_data['daily_revenue'],
            mode='lines+markers',
            name='Daily Revenue',
            line=dict(color='#1f77b4', width=2),
            yaxis='y'
        ))
        
        # Active players (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=revenue_data['date'],
            y=revenue_data['active_players'],
            mode='lines+markers',
            name='Active Players',
            line=dict(color='#ff7f0e', width=2),
            yaxis='y2'
        ))
        
        segment_text = f'{segment} Players' if segment != 'All' else 'All Players'
        title = f'Daily Revenue Trends - {segment_text} ({time_period})'
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis=dict(title='Revenue ($)', side='left'),
            yaxis2=dict(title='Active Players', side='right', overlaying='y'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_top_games_chart(self, games_data: pd.DataFrame, segment: str = 'All', time_period: str = 'All Time'):
        """Create top games by revenue chart."""
        if games_data.empty:
            st.warning("No games data available")
            return
            
        # Create bar chart
        fig = px.bar(
            games_data,
            x='game_type',
            y='total_revenue',
            color='house_edge_pct',
            title=f'Top Games by Revenue - {segment if segment != "All" else "All"} Players ({time_period})',
            labels={
                'total_revenue': 'Total Revenue ($)',
                'game_type': 'Game Type',
                'house_edge_pct': 'House Edge (%)'
            },
            color_continuous_scale='RdYlGn',
            text='total_revenue'
        )
        
        # Format text labels
        fig.update_traces(
            texttemplate='$%{text:,.0f}',
            textposition='outside'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_data_table(self, data: pd.DataFrame, title: str):
        """Display data in an interactive table."""
        if not data.empty:
            st.subheader(title)
            st.dataframe(data, use_container_width=True)
        else:
            st.warning(f"No data available for {title}")

def main():
    """Main dashboard function."""
    # Header
    st.markdown("<h1 class='main-header'>🎰 Casino Intelligence Hub</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = CasinoDashboard()
    
    # Sidebar for navigation and filters
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        st.markdown("---")
        
        # Time period filter
        st.subheader("📅 Time Period")
        time_period = st.selectbox(
            "Select time period:",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            index=1,  # Default to 30 days
            help="Filter data by time period"
        )
        
        # Player segment filter (FUNCTIONAL NOW!)
        st.subheader("👥 Player Segments")
        segment_filter = st.selectbox(
            "Filter revenue and games by segment:",
            ["All", "High Value", "Medium Value", "Low Value"],
            help="This filter affects revenue trends and game analysis below"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dashboard Info")
        st.info(f"""
        **Current Filters**: 
        • **Time Period**: {time_period}
        • **Player Segment**: {segment_filter} Players
        
        This affects:
        • Revenue trend charts
        • Top games analysis
        • Performance metrics
        """)
    
    # Main content area
    try:
        # Test database connection
        if not dashboard.db_manager.test_connection():
            st.error("❌ Database connection failed. Please check your configuration.")
            st.stop()
        
        # Load and display KPIs (TIME-FILTERED!)
        st.subheader(f"📈 Key Performance Indicators ({time_period})")
        kpi_data = dashboard.load_kpi_data(time_period)
        dashboard.display_kpis(kpi_data)
        
        st.markdown("---")
        
        # Create three columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"⚠️ Churn Risk Analysis ({time_period})")
            churn_data = dashboard.load_churn_data(time_period)
            dashboard.create_churn_chart(churn_data, time_period)
        
        with col2:
            st.subheader(f"🎯 Player Segmentation ({time_period})")
            segment_data = dashboard.load_segment_data(time_period)
            dashboard.create_segment_chart(segment_data, time_period)
        
        # Revenue analysis section (FILTERED BY SEGMENT)
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader(f"💰 Revenue Trends - {segment_filter} Players ({time_period})")
            revenue_data = dashboard.load_revenue_trends(segment_filter, time_period)
            dashboard.create_revenue_chart(revenue_data, segment_filter, time_period)
        
        with col4:
            st.subheader(f"🎮 Top Games by Revenue - {segment_filter} Players ({time_period})")
            games_data = dashboard.load_top_games(segment_filter, time_period)
            dashboard.create_top_games_chart(games_data, segment_filter, time_period)
        
        # At-Risk Players section
        st.markdown("---")
        st.subheader("⚠️ High-Value At-Risk Players")
        
        col5, col6 = st.columns([2, 1])
        
        with col5:
            at_risk_data = dashboard.load_at_risk_players()
            if not at_risk_data.empty:
                st.dataframe(
                    at_risk_data,
                    use_container_width=True,
                    column_config={
                        "user_id": "Player ID",
                        "total_revenue": st.column_config.NumberColumn("Total Revenue", format="$%d"),
                        "total_profit": st.column_config.NumberColumn("Total Profit", format="$%d"),
                        "total_transactions": "Transactions",
                        "days_active": "Days Active",
                        "last_activity_date": "Last Activity",
                        "risk_level": "Risk Level"
                    }
                )
            else:
                st.warning("No at-risk player data available")
        
        with col6:
            if not at_risk_data.empty:
                st.metric("At-Risk Players", len(at_risk_data))
                st.metric("Total At-Risk Revenue", f"${at_risk_data['total_revenue'].sum():,.0f}")
                st.metric("Avg Revenue/Player", f"${at_risk_data['total_revenue'].mean():,.0f}")
        
        # Data tables section
        st.markdown("---")
        st.subheader(f"📋 Detailed Data ({time_period})")
        
        # Tabs for different data views
        tab1, tab2, tab3, tab4 = st.tabs([
            f"Churn Data ({time_period})", 
            f"Segments ({time_period})", 
            f"Revenue Trends ({time_period})", 
            f"Top Games ({time_period})"
        ])
        
        with tab1:
            if not churn_data.empty:
                st.dataframe(churn_data, use_container_width=True)
            else:
                st.warning("No churn data available")
        
        with tab2:
            if not segment_data.empty:
                st.dataframe(segment_data, use_container_width=True)
            else:
                st.warning("No segment data available")
        
        with tab3:
            if not revenue_data.empty:
                st.dataframe(revenue_data, use_container_width=True)
            else:
                st.warning("No revenue data available")
        
        with tab4:
            if not games_data.empty:
                st.dataframe(games_data, use_container_width=True)
            else:
                st.warning("No games data available")
            
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        st.error(f"❌ An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Casino Intelligence Hub** | Powered by PostgreSQL, Python & Streamlit | "
        f"Data as of: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main() 