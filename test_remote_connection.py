#!/usr/bin/env python3
"""
Test Remote Connection to Local PostgreSQL
==========================================

This script helps test that your local PostgreSQL is accessible
from external sources like Streamlit Cloud.
"""

import psycopg2
import requests
import socket
import sys
import getpass
from urllib.parse import urlparse

def get_public_ip():
    """Get current public IP"""
    try:
        response = requests.get('https://ipinfo.io/ip', timeout=5)
        return response.text.strip()
    except:
        return None

def test_port_open(host, port):
    """Test if a port is open on the host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def test_database_connection(host, port, database, username, password):
    """Test database connection"""
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=username,
            password=password,
            connect_timeout=10
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM raw_casino_data;")
        count = cursor.fetchone()[0]
        
        conn.close()
        return True, version, count
    except Exception as e:
        return False, str(e), 0

def test_from_external_service():
    """Test connection using an external service (simulates Streamlit Cloud)"""
    print("Testing connection simulation...")
    
    public_ip = get_public_ip()
    if not public_ip:
        print("❌ Could not determine public IP")
        return False
    
    print(f"📍 Your public IP: {public_ip}")
    
    # Test port accessibility
    print("🔍 Testing if PostgreSQL port is accessible...")
    if test_port_open(public_ip, 5432):
        print("✅ Port 5432 is accessible from outside")
    else:
        print("❌ Port 5432 is not accessible from outside")
        print("   Check your router port forwarding and firewall settings")
        return False
    
    # Test database connection
    print("🔍 Testing database connection...")
    password = getpass.getpass("Enter your PostgreSQL password: ")
    
    success, result, count = test_database_connection(
        public_ip, 5432, "casino_intelligence", "postgres", password
    )
    
    if success:
        print(f"✅ Database connection successful!")
        print(f"   PostgreSQL Version: {result}")
        print(f"   Records in casino data: {count:,}")
        return True
    else:
        print(f"❌ Database connection failed: {result}")
        return False

def create_connection_test_for_streamlit():
    """Create a simple connection test that can be run on Streamlit Cloud"""
    test_code = '''
import streamlit as st
import psycopg2

st.title("Casino Intelligence Hub - Connection Test")

try:
    conn = st.connection("postgresql", type="sql")
    df = conn.query("SELECT COUNT(*) as total_records FROM raw_casino_data;")
    
    st.success(f"✅ Connected successfully!")
    st.write(f"Total records: {df.iloc[0]['total_records']:,}")
    
    # Test a sample query
    sample_df = conn.query("""
        SELECT player_id, game_type, bet_amount, timestamp 
        FROM raw_casino_data 
        ORDER BY timestamp DESC 
        LIMIT 5
    """)
    
    st.subheader("Sample Data:")
    st.dataframe(sample_df)
    
except Exception as e:
    st.error(f"❌ Connection failed: {str(e)}")
    st.write("Please check:")
    st.write("1. Your local PostgreSQL is running")
    st.write("2. Port 5432 is forwarded in your router")
    st.write("3. Windows Firewall allows PostgreSQL")
    st.write("4. secrets.toml has correct IP and password")
'''
    
    with open('streamlit_connection_test.py', 'w') as f:
        f.write(test_code)
    
    print("✅ Created streamlit_connection_test.py")
    print("   Upload this to your Streamlit Cloud app to test the connection")

def check_router_instructions():
    """Display router port forwarding instructions"""
    print("\n📋 Router Port Forwarding Setup:")
    print("=" * 50)
    print("1. Access your router admin panel (usually 192.168.1.1 or 192.168.0.1)")
    print("2. Look for 'Port Forwarding' or 'Virtual Server' settings")
    print("3. Add a new rule:")
    print(f"   - External Port: 5432")
    print(f"   - Internal IP: {socket.gethostbyname(socket.gethostname())}")
    print(f"   - Internal Port: 5432")
    print(f"   - Protocol: TCP")
    print(f"   - Enable/Active: Yes")
    print("4. Save and restart your router")

def check_current_setup():
    """Check current network and PostgreSQL setup"""
    print("🔍 Checking Current Setup")
    print("=" * 30)
    
    # Local IP
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"💻 Local IP: {local_ip}")
    
    # Public IP
    public_ip = get_public_ip()
    print(f"🌍 Public IP: {public_ip}")
    
    # PostgreSQL local port
    if test_port_open('localhost', 5432):
        print("✅ PostgreSQL running locally on port 5432")
    else:
        print("❌ PostgreSQL not accessible on localhost:5432")
    
    # Test local connection
    try:
        password = getpass.getpass("Enter PostgreSQL password to test local connection: ")
        success, result, count = test_database_connection(
            'localhost', 5432, 'casino_intelligence', 'postgres', password
        )
        if success:
            print(f"✅ Local database connection works ({count:,} records)")
        else:
            print(f"❌ Local database connection failed: {result}")
    except KeyboardInterrupt:
        print("⏭️  Skipped local connection test")

def main():
    print("🧪 Remote PostgreSQL Connection Tester")
    print("=" * 40)
    
    while True:
        print("\nChoose an option:")
        print("1. Check current setup")
        print("2. Test external connection (full test)")
        print("3. Show router setup instructions")
        print("4. Create Streamlit connection test file")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            check_current_setup()
        elif choice == '2':
            if test_from_external_service():
                print("\n🎉 Your local PostgreSQL is ready for Streamlit Cloud!")
            else:
                print("\n🛠️  Setup still needs work")
        elif choice == '3':
            check_router_instructions()
        elif choice == '4':
            create_connection_test_for_streamlit()
        elif choice == '5':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 