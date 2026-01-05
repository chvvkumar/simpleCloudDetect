import time
import sys
# Ensure the alpaca module is in the path if running from the root of the project
# sys.path.append("path/to/alpyca/folder") 

try:
    from alpaca.safetymonitor import SafetyMonitor
    from alpaca.exceptions import *
except ImportError:
    print("Error: Could not import 'alpaca'. Ensure the alpyca package is installed or in your Python path.")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEVICE_ADDRESS = "allskypi5.lan:11111"  # Updated based on your log
DEVICE_NUMBER = 0                   
PROTOCOL = "http"                   
# ==============================================================================

def log(message, level="INFO"):
    print(f"[{level}] {message}")

def test_safety_monitor():
    log(f"Initializing SafetyMonitor at {PROTOCOL}://{DEVICE_ADDRESS} Device #{DEVICE_NUMBER}")
    
    try:
        # 1. Initialize the Device
        # ----------------------------------------------------------------------
        safetymon = SafetyMonitor(DEVICE_ADDRESS, DEVICE_NUMBER, PROTOCOL)
        log("Device object initialized.")

        # 2. Test Connection (Connect)
        # ----------------------------------------------------------------------
        log("Attempting to connect...")
        try:
            safetymon.Connected = True
            
            # Wait for connection to complete (handling async behavior)
            attempts = 0
            while attempts < 10:
                if safetymon.Connected:
                    break
                if hasattr(safetymon, 'Connecting') and safetymon.Connecting:
                    log("Device is connecting...", "WAIT")
                time.sleep(1)
                attempts += 1
            
            if safetymon.Connected:
                log("Successfully connected to device.")
            else:
                log("Failed to connect: Timed out.", "ERROR")
                return

        except Exception as e:
            log(f"Exception during connection: {e}", "ERROR")
            return

        # 3. Test Standard Device Properties (Metadata)
        # ----------------------------------------------------------------------
        log("--- Querying Device Information ---")
        try:
            name = safetymon.Name
            log(f"Name:             {name}")
        except Exception as e:
            log(f"Failed to read Name: {e}", "WARN")

        try:
            desc = safetymon.Description
            log(f"Description:      {desc}")
        except Exception as e:
            log(f"Failed to read Description: {e}", "WARN")

        try:
            driver_info = safetymon.DriverInfo
            log(f"Driver Info:      {driver_info}")
        except Exception as e:
            log(f"Failed to read DriverInfo: {e}", "WARN")

        try:
            driver_version = safetymon.DriverVersion
            log(f"Driver Version:   {driver_version}")
        except Exception as e:
            log(f"Failed to read DriverVersion: {e}", "WARN")
        
        try:
            interface_version = safetymon.InterfaceVersion
            log(f"Interface Ver:    {interface_version}")
        except Exception as e:
            log(f"Failed to read InterfaceVersion: {e}", "WARN")

        # 4. Test SafetyMonitor Specific Property: IsSafe
        # ----------------------------------------------------------------------
        log("--- Testing Safety State ---")
        try:
            # Poll safety state a few times
            for i in range(3):
                is_safe = safetymon.IsSafe
                status_str = "SAFE" if is_safe else "UNSAFE"
                log(f"Check {i+1}: Environment is {status_str} (IsSafe={is_safe})")
                time.sleep(1)
        except NotConnectedException:
            log("Error: Device reported Not Connected while reading IsSafe.", "ERROR")
        except DriverException as e:
            log(f"Driver Error reading IsSafe: {e}", "ERROR")
        except Exception as e:
            log(f"Unexpected error reading IsSafe: {e}", "ERROR")

        # 5. Test Supported Actions (Optional)
        # ----------------------------------------------------------------------
        log("--- Querying Supported Actions ---")
        try:
            actions = safetymon.SupportedActions
            if actions:
                log(f"Supported Actions: {actions}")
            else:
                log("No custom actions supported.")
        except Exception as e:
            log(f"Failed to read SupportedActions: {e}", "WARN")

        # 6. Test Disconnection (Updated for Async)
        # ----------------------------------------------------------------------
        log("--- Disconnecting ---")
        try:
            # Send disconnect request
            safetymon.Connected = False
            
            # Wait for disconnection to complete
            attempts = 0
            disconnected = False
            
            while attempts < 10:
                # Check if device is done "Connecting" (which handles disconnects too)
                if hasattr(safetymon, 'Connecting') and safetymon.Connecting:
                    log("Device is disconnecting...", "WAIT")
                    time.sleep(1)
                elif not safetymon.Connected:
                    disconnected = True
                    break
                else:
                    # Not connecting, but still reports Connected=True?
                    # Give it a moment to reflect state
                    time.sleep(1)
                
                attempts += 1

            if disconnected:
                log("Successfully disconnected.")
            else:
                log("Warning: Device still reports Connected=True after timeout.", "WARN")
                
        except Exception as e:
            log(f"Exception during disconnection: {e}", "ERROR")

    except AlpacaRequestException as e:
        log(f"Communication Error (Alpaca Request Failed): {e}", "CRITICAL")
    except KeyboardInterrupt:
        log("Test interrupted by user.", "WARN")
    except Exception as e:
        log(f"An unexpected error occurred: {e}", "CRITICAL")
    finally:
        log("Test sequence completed.")

if __name__ == "__main__":
    test_safety_monitor()