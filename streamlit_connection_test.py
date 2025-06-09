import streamlit as st
import psycopg2

st.title("Casino Intelligence Hub - Connection Test")

try:
    conn = st.connection("postgresql", type="sql")
    df = conn.query("SELECT COUNT(*) as total_records FROM raw_casino_data;")
    
    st.success("Connected successfully!")
    st.write(f"Total records: {df.iloc[0]['total_records']:,}")
    
    # Test a sample query
    sample_df = conn.query("""
        SELECT user_id, game_type, bet_amount, calc_date 
        FROM raw_casino_data 
        ORDER BY calc_date DESC 
        LIMIT 5
    """)
    
    st.subheader("Sample Data:")
    st.dataframe(sample_df)
    
except Exception as e:
    st.error(f"Connection failed: {str(e)}")
    st.write("Please check:")
    st.write("1. Your local PostgreSQL is running")
    st.write("2. Port 5432 is forwarded in your router")
    st.write("3. Windows Firewall allows PostgreSQL")
    st.write("4. secrets.toml has correct IP and password")
