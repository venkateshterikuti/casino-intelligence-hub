@echo off
echo ================================================================
echo Finishing PostgreSQL Remote Setup (Administrator Required)
echo ================================================================

echo.
echo 1. Creating Windows Firewall rule for PostgreSQL...
netsh advfirewall firewall add rule name="PostgreSQL" dir=in action=allow protocol=TCP localport=5432
if %ERRORLEVEL% EQU 0 (
    echo ✅ Firewall rule created successfully
) else (
    echo ❌ Failed to create firewall rule
)

echo.
echo 2. Restarting PostgreSQL service...
sc stop postgresql-x64-17
timeout /t 3 /nobreak >nul
sc start postgresql-x64-17
if %ERRORLEVEL% EQU 0 (
    echo ✅ PostgreSQL service restarted successfully
) else (
    echo ❌ Failed to restart PostgreSQL service
)

echo.
echo 3. Checking PostgreSQL service status...
sc query postgresql-x64-17

echo.
echo ================================================================
echo Setup Complete!
echo ================================================================
echo.
echo Next Steps:
echo 1. Configure router port forwarding (see LOCAL_DATABASE_HOSTING_GUIDE.md)
echo 2. Test external connection with: python test_remote_connection.py
echo 3. Deploy to Streamlit Cloud
echo.
echo Your public IP: 172.59.201.228
echo.
pause 