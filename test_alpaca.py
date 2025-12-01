#!/usr/bin/env python3
"""
Test script for ASCOM Alpaca SafetyMonitor
Validates API compliance and basic functionality
"""

import requests
import sys
import time
from typing import Dict, Any

BASE_URL = "http://localhost:11111"
DEVICE_NUM = 0


class AlpacaAPITester:
    """Test ASCOM Alpaca SafetyMonitor API compliance"""
    
    def __init__(self, base_url: str = BASE_URL, device_num: int = DEVICE_NUM):
        self.base_url = base_url
        self.device_num = device_num
        self.passed = 0
        self.failed = 0
        self.client_transaction_id = 1
    
    def get_next_transaction_id(self) -> int:
        """Get next client transaction ID"""
        tx_id = self.client_transaction_id
        self.client_transaction_id += 1
        return tx_id
    
    def test_get(self, endpoint: str, expected_type: type = None, 
                 should_fail: bool = False) -> bool:
        """Test GET endpoint"""
        url = f"{self.base_url}/api/v1/safetymonitor/{self.device_num}/{endpoint}"
        params = {"ClientTransactionID": self.get_next_transaction_id()}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            # Check response structure
            if not all(key in data for key in ["ClientTransactionID", "ServerTransactionID", 
                                                "ErrorNumber", "ErrorMessage"]):
                print(f"  ❌ Missing required response fields")
                return False
            
            # Check for expected failure
            if should_fail:
                if data["ErrorNumber"] == 0:
                    print(f"  ❌ Expected error but got success")
                    return False
                print(f"  ✅ Correctly returned error: {data['ErrorMessage']}")
                return True
            
            # Check for success
            if data["ErrorNumber"] != 0:
                print(f"  ❌ Error: {data['ErrorMessage']} (Code: {data['ErrorNumber']})")
                return False
            
            # Check value type if specified
            if expected_type and "Value" in data:
                if not isinstance(data["Value"], expected_type):
                    print(f"  ❌ Expected {expected_type.__name__}, got {type(data['Value']).__name__}")
                    return False
            
            print(f"  ✅ Success - Value: {data.get('Value', 'N/A')}")
            return True
            
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return False
    
    def test_put(self, endpoint: str, data: Dict[str, Any], 
                 should_fail: bool = False) -> bool:
        """Test PUT endpoint"""
        url = f"{self.base_url}/api/v1/safetymonitor/{self.device_num}/{endpoint}"
        form_data = {
            "ClientTransactionID": self.get_next_transaction_id(),
            **data
        }
        
        try:
            response = requests.put(url, data=form_data, timeout=5)
            resp_data = response.json()
            
            # Check response structure
            if not all(key in resp_data for key in ["ClientTransactionID", "ServerTransactionID", 
                                                     "ErrorNumber", "ErrorMessage"]):
                print(f"  ❌ Missing required response fields")
                return False
            
            # Check for expected failure
            if should_fail:
                if resp_data["ErrorNumber"] == 0:
                    print(f"  ❌ Expected error but got success")
                    return False
                print(f"  ✅ Correctly returned error: {resp_data['ErrorMessage']}")
                return True
            
            # Check for success
            if resp_data["ErrorNumber"] != 0:
                print(f"  ❌ Error: {resp_data['ErrorMessage']} (Code: {resp_data['ErrorNumber']})")
                return False
            
            print(f"  ✅ Success")
            return True
            
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return False
    
    def run_test(self, name: str, test_func) -> None:
        """Run a test and track results"""
        print(f"\n{name}")
        if test_func():
            self.passed += 1
        else:
            self.failed += 1
    
    def test_management_api(self) -> None:
        """Test management API endpoints"""
        print("\n" + "="*60)
        print("MANAGEMENT API TESTS")
        print("="*60)
        
        # Test API versions
        try:
            response = requests.get(f"{self.base_url}/management/apiversions", timeout=5)
            versions = response.json()
            if 1 in versions:
                print("\n✅ API Versions - Supports v1")
                self.passed += 1
            else:
                print("\n❌ API Versions - Missing v1")
                self.failed += 1
        except Exception as e:
            print(f"\n❌ API Versions - Exception: {e}")
            self.failed += 1
        
        # Test description
        try:
            response = requests.get(f"{self.base_url}/management/v1/description", timeout=5)
            desc = response.json()
            if "ServerName" in desc:
                print(f"✅ Server Description - {desc['ServerName']}")
                self.passed += 1
            else:
                print("❌ Server Description - Missing ServerName")
                self.failed += 1
        except Exception as e:
            print(f"❌ Server Description - Exception: {e}")
            self.failed += 1
        
        # Test configured devices
        try:
            response = requests.get(f"{self.base_url}/management/v1/configureddevices", timeout=5)
            devices = response.json()
            if isinstance(devices, list) and len(devices) > 0:
                print(f"✅ Configured Devices - Found {len(devices)} device(s)")
                self.passed += 1
            else:
                print("❌ Configured Devices - No devices found")
                self.failed += 1
        except Exception as e:
            print(f"❌ Configured Devices - Exception: {e}")
            self.failed += 1
    
    def test_common_endpoints(self) -> None:
        """Test common device endpoints"""
        print("\n" + "="*60)
        print("COMMON DEVICE ENDPOINT TESTS")
        print("="*60)
        
        self.run_test("Test: Name", 
            lambda: self.test_get("name", str))
        
        self.run_test("Test: Description", 
            lambda: self.test_get("description", str))
        
        self.run_test("Test: DriverInfo", 
            lambda: self.test_get("driverinfo", str))
        
        self.run_test("Test: DriverVersion", 
            lambda: self.test_get("driverversion", str))
        
        self.run_test("Test: InterfaceVersion", 
            lambda: self.test_get("interfaceversion", int))
        
        self.run_test("Test: SupportedActions", 
            lambda: self.test_get("supportedactions", list))
        
        self.run_test("Test: Connected (GET)", 
            lambda: self.test_get("connected", bool))
        
        self.run_test("Test: Connecting (GET)", 
            lambda: self.test_get("connecting", bool))
    
    def test_connection_workflow(self) -> None:
        """Test connection/disconnection workflow"""
        print("\n" + "="*60)
        print("CONNECTION WORKFLOW TESTS")
        print("="*60)
        
        # Ensure disconnected first
        self.run_test("Test: Disconnect", 
            lambda: self.test_put("connected", {"Connected": "false"}))
        
        time.sleep(1)
        
        # Connect
        self.run_test("Test: Connect", 
            lambda: self.test_put("connected", {"Connected": "true"}))
        
        time.sleep(2)  # Wait for initial detection
        
        # Verify connected
        self.run_test("Test: Verify Connected", 
            lambda: self.test_get("connected", bool))
    
    def test_safetymonitor_endpoints(self) -> None:
        """Test SafetyMonitor-specific endpoints"""
        print("\n" + "="*60)
        print("SAFETYMONITOR SPECIFIC TESTS")
        print("="*60)
        
        self.run_test("Test: IsSafe (while connected)", 
            lambda: self.test_get("issafe", bool))
        
        self.run_test("Test: DeviceState", 
            lambda: self.test_get("devicestate", list))
    
    def test_error_conditions(self) -> None:
        """Test error handling"""
        print("\n" + "="*60)
        print("ERROR HANDLING TESTS")
        print("="*60)
        
        # Test invalid device number
        old_device = self.device_num
        self.device_num = 99
        self.run_test("Test: Invalid Device Number", 
            lambda: self.test_get("name", should_fail=True))
        self.device_num = old_device
        
        # Test deprecated methods
        self.run_test("Test: CommandBlind (deprecated)", 
            lambda: self.test_put("commandblind", {"Command": "test", "Raw": "false"}, 
                                 should_fail=True))
        
        # Test unsupported action
        self.run_test("Test: Unsupported Action", 
            lambda: self.test_put("action", {"Action": "UnsupportedAction", "Parameters": ""}, 
                                 should_fail=True))
        
        # Disconnect and test IsSafe (should fail)
        self.test_put("connected", {"Connected": "false"})
        time.sleep(1)
        self.run_test("Test: IsSafe (while disconnected)", 
            lambda: self.test_get("issafe", should_fail=True))
    
    def run_all_tests(self) -> None:
        """Run all tests"""
        print("="*60)
        print("ASCOM ALPACA SAFETYMONITOR API COMPLIANCE TESTS")
        print("="*60)
        print(f"Testing: {self.base_url}")
        print(f"Device Number: {self.device_num}")
        
        # Check if server is running
        try:
            response = requests.get(f"{self.base_url}/management/apiversions", timeout=5)
            print(f"✅ Server is responding")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("\nPlease ensure the Alpaca server is running:")
            print("  python alpaca_safety_monitor.py")
            sys.exit(1)
        
        # Run test suites
        self.test_management_api()
        self.test_common_endpoints()
        self.test_connection_workflow()
        self.test_safetymonitor_endpoints()
        self.test_error_conditions()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        total = self.passed + self.failed
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        
        if self.failed == 0:
            print("\n🎉 All tests passed! ASCOM Alpaca API is compliant.")
            sys.exit(0)
        else:
            print(f"\n⚠️  {self.failed} test(s) failed.")
            sys.exit(1)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ASCOM Alpaca SafetyMonitor API")
    parser.add_argument("--url", default=BASE_URL, 
                       help=f"Base URL (default: {BASE_URL})")
    parser.add_argument("--device", type=int, default=DEVICE_NUM,
                       help=f"Device number (default: {DEVICE_NUM})")
    
    args = parser.parse_args()
    
    tester = AlpacaAPITester(args.url, args.device)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
