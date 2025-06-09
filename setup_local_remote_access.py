#!/usr/bin/env python3
"""
Setup Local PostgreSQL for Remote Access from Streamlit Cloud
============================================================

This script helps configure your local PostgreSQL to be accessible
from Streamlit Cloud while maintaining security.
"""

import os
import sys
import subprocess
import psycopg2
import getpass
import socket
import requests
from pathlib import Path

def get_postgres_config_path():
    """Find PostgreSQL configuration files"""
    possible_paths = [
        r"C:\Program Files\PostgreSQL\17\data",
        r"C:\Program Files\PostgreSQL\16\data", 
        r"C:\Program Files\PostgreSQL\15\data",
        r"C:\PostgreSQL\17\data",
        r"C:\PostgreSQL\16\data"
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "postgresql.conf")):
            return path
    
    return None

def get_public_ip():
    """Get your public IP address"""
    try:
        response = requests.get('https://ipinfo.io/ip', timeout=5)
        return response.text.strip()
    except:
        try:
            response = requests.get('https://api.ipify.org', timeout=5)
            return response.text.strip()
        except:
            return "Unable to determine"

def test_local_connection():
    """Test local PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="casino_intelligence",
            user="venkatesh",
            password="casino123"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_casino_data;")
        count = cursor.fetchone()[0]
        print(f"✅ Local connection successful! Found {count:,} records")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Local connection failed: {e}")
        return False

def backup_config_file(file_path):
    """Create backup of configuration file"""
    backup_path = f"{file_path}.backup"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"✅ Backup created: {backup_path}")

def update_postgresql_conf(config_path):
    """Update postgresql.conf for remote access"""
    conf_file = os.path.join(config_path, "postgresql.conf")
    
    if not os.path.exists(conf_file):
        print(f"❌ Configuration file not found: {conf_file}")
        return False
    
    backup_config_file(conf_file)
    
    # Read current config
    with open(conf_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Update configuration
    updated_lines = []
    listen_addresses_updated = False
    port_updated = False
    
    for line in lines:
        if line.strip().startswith('#listen_addresses') or line.strip().startswith('listen_addresses'):
            updated_lines.append("listen_addresses = '*'\t\t# Allow connections from any IP\n")
            listen_addresses_updated = True
        elif line.strip().startswith('#port') or (line.strip().startswith('port') and 'postgresql' not in line.lower()):
            updated_lines.append("port = 5432\t\t\t\t# Standard PostgreSQL port\n")
            port_updated = True
        else:
            updated_lines.append(line)
    
    # Add settings if not found
    if not listen_addresses_updated:
        updated_lines.append("\n# Remote access configuration\n")
        updated_lines.append("listen_addresses = '*'\t\t# Allow connections from any IP\n")
    
    if not port_updated:
        updated_lines.append("port = 5432\t\t\t\t# Standard PostgreSQL port\n")
    
    # Write updated config
    with open(conf_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print("✅ postgresql.conf updated for remote access")
    return True

def update_pg_hba_conf(config_path):
    """Update pg_hba.conf for remote authentication"""
    hba_file = os.path.join(config_path, "pg_hba.conf")
    
    if not os.path.exists(hba_file):
        print(f"❌ Authentication file not found: {hba_file}")
        return False
    
    backup_config_file(hba_file)
    
    # Read current config
    with open(hba_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add Streamlit Cloud access rule if not already present
    streamlit_rule = "host    casino_intelligence    venkatesh    0.0.0.0/0    md5"
    
    if streamlit_rule not in content:
        # Add the rule before local connections
        lines = content.split('\n')
        insert_index = -1
        
        for i, line in enumerate(lines):
            if 'host' in line and 'all' in line and '127.0.0.1/32' in line:
                insert_index = i
                break
        
        if insert_index != -1:
            lines.insert(insert_index, "# Streamlit Cloud access")
            lines.insert(insert_index + 1, streamlit_rule)
        else:
            lines.append("\n# Streamlit Cloud access")
            lines.append(streamlit_rule)
        
        # Write updated config
        with open(hba_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("✅ pg_hba.conf updated for remote authentication")
    else:
        print("✅ pg_hba.conf already configured for remote access")
    
    return True

def restart_postgresql():
    """Restart PostgreSQL service"""
    try:
        subprocess.run(['sc', 'stop', 'postgresql-x64-17'], 
                      capture_output=True, text=True, check=False)
        print("🔄 Stopping PostgreSQL service...")
        
        import time
        time.sleep(3)
        
        result = subprocess.run(['sc', 'start', 'postgresql-x64-17'], 
                               capture_output=True, text=True, check=True)
        print("✅ PostgreSQL service restarted")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to restart PostgreSQL: {e}")
        print("You may need to run this script as Administrator")
        return False

def create_streamlit_secrets():
    """Create Streamlit secrets configuration"""
    secrets_content = f'''[connections.postgresql]
dialect = "postgresql"
host = "YOUR_PUBLIC_IP_HERE"
port = 5432
database = "casino_intelligence"
username = "venkatesh"
password = "casino123"
'''
    
    os.makedirs('.streamlit', exist_ok=True)
    
    with open('.streamlit/secrets.toml', 'w') as f:
        f.write(secrets_content)
    
    print("✅ Created .streamlit/secrets.toml template")

def check_firewall():
    """Check Windows Firewall rules for PostgreSQL"""
    try:
        result = subprocess.run([
            'netsh', 'advfirewall', 'firewall', 'show', 'rule', 
            'name="PostgreSQL"'
        ], capture_output=True, text=True)
        
        if "No rules match" in result.stdout:
            print("⚠️  No firewall rule found for PostgreSQL")
            return False
        else:
            print("✅ Firewall rule exists for PostgreSQL")
            return True
    except Exception as e:
        print(f"⚠️  Could not check firewall: {e}")
        return False

def create_firewall_rule():
    """Create Windows Firewall rule for PostgreSQL"""
    try:
        subprocess.run([
            'netsh', 'advfirewall', 'firewall', 'add', 'rule',
            'name="PostgreSQL"',
            'dir=in',
            'action=allow',
            'protocol=TCP',
            'localport=5432'
        ], check=True)
        print("✅ Created firewall rule for PostgreSQL port 5432")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create firewall rule (need Administrator privileges)")
        return False

def main():
    print("🚀 Setting up Local PostgreSQL for Streamlit Cloud Access")
    print("=" * 60)
    
    # Get public IP
    public_ip = get_public_ip()
    print(f"📍 Your public IP: {public_ip}")
    
    # Test local connection first
    print("\n1. Testing local database connection...")
    if not test_local_connection():
        print("❌ Please fix local connection issues first")
        return
    
    # Find PostgreSQL config
    print("\n2. Finding PostgreSQL configuration...")
    config_path = get_postgres_config_path()
    if not config_path:
        print("❌ Could not find PostgreSQL configuration directory")
        print("Please check your PostgreSQL installation")
        return
    print(f"✅ Found config at: {config_path}")
    
    # Update configurations
    print("\n3. Updating PostgreSQL configuration...")
    if not update_postgresql_conf(config_path):
        return
    
    if not update_pg_hba_conf(config_path):
        return
    
    # Check/create firewall rule
    print("\n4. Checking Windows Firewall...")
    if not check_firewall():
        print("Creating firewall rule...")
        create_firewall_rule()
    
    # Restart PostgreSQL
    print("\n5. Restarting PostgreSQL...")
    if not restart_postgresql():
        print("⚠️  Please restart PostgreSQL service manually as Administrator")
    
    # Create Streamlit secrets
    print("\n6. Creating Streamlit configuration...")
    create_streamlit_secrets()
    
    print("\n🎉 Setup Complete!")
    print("=" * 60)
    print(f"📋 Next Steps:")
    print(f"1. Configure your router to forward port 5432 to this computer")
    print(f"2. Update .streamlit/secrets.toml with your public IP: {public_ip}")
    print(f"3. Add your PostgreSQL password to secrets.toml")
    print(f"4. Test connection from Streamlit Cloud")
    print(f"\n⚠️  Security Notes:")
    print(f"- Only your casino_intelligence database is exposed")
    print(f"- Use a strong PostgreSQL password")
    print(f"- Consider setting up SSL certificates for production")
    print(f"- Monitor your database access logs")

if __name__ == "__main__":
    main()