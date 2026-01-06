import requests
import time
import sys
import threading
import subprocess
import os
import signal

BASE_URL = "http://localhost:11111/api/v1/safetymonitor/0"

def log(msg):
    print(f"[TEST] {msg}")

def check_connected():
    try:
        resp = requests.get(f"{BASE_URL}/connected")
        resp.raise_for_status()
        return resp.json()["Value"]
    except Exception as e:
        log(f"Error checking connected status: {e}")
        return None

def check_issafe():
    try:
        resp = requests.get(f"{BASE_URL}/issafe")
        resp.raise_for_status()
        return resp.json()["Value"]
    except Exception as e:
        log(f"Error checking safe status: {e}")
        return None

def connect(client_id, ip_spoof=None):
    params = {'ClientID': client_id, 'Connected': 'True'}
    headers = {}
    if ip_spoof:
        # Note: Werkzeug/Flask remote_addr isn't easily spoofed via headers unless behind proxy setup
        # But for this test we might just rely on different ClientIDs.
        # The refactored code uses (client_ip, client_id) as key.
        # Running locally, IP will be 127.0.0.1 for all requests.
        # So we really rely on ClientID to distinguish sessions if IP is same.
        pass
    
    try:
        resp = requests.put(f"{BASE_URL}/connected", data=params)
        resp.raise_for_status()
        return True
    except Exception as e:
        log(f"Error connecting client {client_id}: {e}")
        return False

def disconnect(client_id):
    params = {'ClientID': client_id, 'Connected': 'False'}
    try:
        resp = requests.put(f"{BASE_URL}/connected", data=params)
        resp.raise_for_status()
        return True
    except Exception as e:
        log(f"Error disconnecting client {client_id}: {e}")
        return False

def run_tests():
    # Start Server
    log("Starting server subprocess...")
    server_process = subprocess.Popen([sys.executable, "alpaca_safety_monitor.py"], 
                                      stdout=subprocess.DEVNULL, 
                                      stderr=subprocess.DEVNULL)
    
    try:
        # Wait for server to be ready
        log("Waiting for server...")
        for _ in range(30):
            try:
                # We can check simple endpoint like /management/apiversions or even /connected
                resp = requests.get(f"{BASE_URL}/connected")
                if resp.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        else:
            log("Server not reachable")
            sys.exit(1)

        log("Server ready. Starting tests.")

        # Scenario A: Single Client
        log("--- Scenario A: Single Client ---")
        
        # Ensure initially disconnected
        if check_connected():
            log("WARN: Initially connected? Disconnecting all potential clients...")
        
        log("Connecting Client A (ID 1)...")
        connect(1)
        
        connected = check_connected()
        log(f"Is Connected? {connected}")
        if not connected:
            log("FAIL: Client A should be connected")
            sys.exit(1)
            
        log("Disconnecting Client A...")
        disconnect(1)
        
        connected = check_connected()
        log(f"Is Connected? {connected}")
        if connected:
            log("FAIL: Should be disconnected after Client A leaves")
            sys.exit(1)

        log("Scenario A Passed")

        # Scenario B: Shared Session
        log("--- Scenario B: Shared Session ---")
        
        log("Connecting Client A (ID 1)...")
        connect(1)
        
        log("Connecting Client B (ID 2)...")
        connect(2)
        
        log("Disconnecting Client A...")
        disconnect(1)
        
        connected = check_connected()
        log(f"Is Connected? {connected}")
        if not connected:
            log("FAIL: Should still be connected (Client B is active)")
            sys.exit(1)
            
        log("Disconnecting Client B...")
        disconnect(2)
        
        connected = check_connected()
        log(f"Is Connected? {connected}")
        if connected:
            log("FAIL: Should be disconnected after Client B leaves")
            sys.exit(1)
            
        log("Scenario B Passed")

        # Scenario C: Fail-safe
        log("--- Scenario C: Fail-safe ---")
        # Ensure fully disconnected
        if check_connected():
             log("FAIL: System should be disconnected")
             sys.exit(1)
             
        is_safe = check_issafe()
        log(f"Is Safe (while disconnected)? {is_safe}")
        if is_safe:
            log("FAIL: Should report Unsafe (False) when disconnected")
            sys.exit(1)
            
        log("Scenario C Passed")
        log("ALL TESTS PASSED")
        
    finally:
        log("Terminating server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    run_tests()
