#!/usr/bin/env python3
"""
🎰 Casino Intelligence Hub Dashboard Launcher

Simple script to start the Streamlit dashboard with proper macOS configuration.
Handles Safari connection issues and provides alternative browser options.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking requirements...")
    
    # Check if we're in the right directory
    if not Path("dashboards/streamlit/app.py").exists():
        print("❌ Error: Not in casino project directory")
        print("   Please run from: /Users/venkatesh/Projects/casino")
        sys.exit(1)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        print("⚠️  Warning: Virtual environment not activated")
        print("   Run: source venv/bin/activate")
    
    print("✅ Requirements check passed")

def start_streamlit():
    """Start Streamlit with optimal macOS settings."""
    print("🚀 Starting Casino Intelligence Dashboard...")
    
    # Streamlit configuration for macOS
    env = os.environ.copy()
    env.update({
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
        'STREAMLIT_SERVER_HEADLESS': 'false',
        'STREAMLIT_SERVER_PORT': '8501',
        'STREAMLIT_SERVER_ADDRESS': '0.0.0.0'
    })
    
    # Start Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'dashboards/streamlit/app.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--server.headless', 'false',
        '--browser.gatherUsageStats', 'false'
    ]
    
    print(f"📡 Starting on: http://localhost:8501")
    print(f"🌐 Network URL: http://0.0.0.0:8501")
    
    try:
        # Start Streamlit process
        process = subprocess.Popen(cmd, env=env)
        
        # Wait a moment for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        # Try to open in browser
        urls_to_try = [
            'http://localhost:8501',
            'http://127.0.0.1:8501',
            'http://0.0.0.0:8501'
        ]
        
        print("\n🌐 Opening dashboard in browser...")
        print("📱 Try these URLs if Safari doesn't work:")
        
        browser_opened = False
        for url in urls_to_try:
            print(f"   • {url}")
            if not browser_opened:
                try:
                    webbrowser.open(url)
                    browser_opened = True
                except:
                    continue
        
        print("\n" + "="*60)
        print("🎰 CASINO INTELLIGENCE DASHBOARD RUNNING")
        print("="*60)
        print(f"📍 Local Access: http://localhost:8501")
        print(f"🔗 Direct IP: http://127.0.0.1:8501")
        print("\n💡 Browser Tips for macOS:")
        print("   • Safari: Try http://127.0.0.1:8501")
        print("   • Chrome: Usually works with any URL above")
        print("   • Firefox: Try http://localhost:8501")
        print("\n⌨️  Press Ctrl+C to stop the dashboard")
        print("="*60)
        
        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            process.terminate()
            print("✅ Dashboard stopped successfully")
            
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you're in the project directory")
        print("   2. Activate virtual environment: source venv/bin/activate")
        print("   3. Check if port 8501 is available")
        sys.exit(1)

def main():
    """Main function."""
    print("🎰 Casino Intelligence Hub - Dashboard Launcher")
    print("="*50)
    
    check_requirements()
    start_streamlit()

if __name__ == "__main__":
    main() 