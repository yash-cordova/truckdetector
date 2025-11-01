from pyModbusTCP.client import ModbusClient
from threading import Thread, RLock
import time
import json
import os

class PLCThread(Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # Make thread daemon so it stops when main thread stops
        
        # Load PLC configuration
        self.config = self.load_config()
        
        # Initialize ModbusClient with config settings
        self.client = ModbusClient(
            host=self.config['plc_connection']['host'],
            port=self.config['plc_connection']['port'],
            auto_open=self.config['plc_connection']['auto_open'],
            auto_close=self.config['plc_connection']['auto_close']
        )
        
        # Initialize states
        self.connection_status = False
        self.running = True
        self.callback = None
        self.last_light_state = None
        self.lock = RLock()  # Reentrant lock for thread safety (allows nested locking)

    def _is_client_open(self):
        """Helper method to check if client is open (handles both property and method)"""
        try:
            if hasattr(self.client, 'is_open'):
                # Check if it's callable (method) or not (property)
                attr = getattr(self.client, 'is_open')
                if callable(attr):
                    return attr()
                else:
                    return bool(attr)
            return False
        except Exception:
            return False

    def load_config(self):
        """Load PLC configuration from main config.json file"""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                # Extract PLC-related config sections
                return {
                    "plc_connection": full_config.get("plc_connection", {
                        "host": "127.0.0.1",
                        "port": 502,
                        "auto_open": True,
                        "auto_close": True
                    }),
                    "plc_settings": full_config.get("plc_settings", {
                        "check_interval": 1
                    }),
                    "light_settings": full_config.get("light_settings", {
                        "green_light_coil": [True, False, False, False, False, False, False, False],
                        "red_light_coil": [False, True, False, False, False, False, False, False],
                        "yellow_light_coil": [False, False, True, False, False, False, False, False]
                    })
                }
        except Exception as e:
            print(f"Error loading config.json: {e}")
            # Return default config if file not found
            return {
                "plc_connection": {
                    "host": "127.0.0.1",
                    "port": 502,
                    "auto_open": True,
                    "auto_close": True
                },
                "plc_settings": {
                    "check_interval": 1
                },
                "light_settings": {
                    "green_light_coil": [True, False, False, False, False, False, False, False],
                    "red_light_coil": [False, True, False, False, False, False, False, False],
                    "yellow_light_coil": [False, False, True, False, False, False, False, False]
                }
            }

    def set_callback(self, callback):
        """Set callback function to notify about PLC status changes"""
        self.callback = callback

    def check_connection(self):
        """Check PLC connection status"""
        with self.lock:
            try:
                # Check if connection is open using helper method
                is_open = self._is_client_open()
                
                # Try to read from the PLC to verify connection
                input_result = self.client.read_discrete_inputs(0, 7)
                is_connected = input_result is not None and is_open
                
                if self.connection_status != is_connected:
                    self.connection_status = is_connected
                    if self.callback:
                        self.callback({
                            'type': 'connection',
                            'status': is_connected
                        })
                
                return is_connected
            except Exception as e:
                print(f"PLC Connection Error: {str(e)}")
                if self.connection_status:
                    self.connection_status = False
                    if self.callback:
                        self.callback({
                            'type': 'connection',
                            'status': False
                        })
                return False

    def update_light_state(self, state):
        """Update light state - state can be 'red', 'yellow', or 'green'"""
        with self.lock:
            # Only update if state has changed
            if self.last_light_state != state:
                self.last_light_state = state
                
                if not self.connection_status:
                    # Try to reconnect
                    try:
                        # Handle is_open as property or method
                        is_open = self._is_client_open()
                        if not is_open:
                            self.client.open()
                        self.connection_status = self.check_connection()
                    except Exception as e:
                        print(f"PLC Reconnection Error: {str(e)}")
                
                if state == 'red':
                    self.redlight()
                elif state == 'yellow':
                    self.yellowlight()
                elif state == 'green':
                    self.greenlight()

    def run(self):
        """Main PLC thread loop - checks connection periodically"""
        while self.running:
            try:
                # Check PLC connection periodically
                self.check_connection()
            except Exception as e:
                print(f"PLC Error: {str(e)}")
            
            time.sleep(self.config['plc_settings']['check_interval'])

    def redlight(self):
        """Turn on red light on PLC"""
        try:
            if self._is_client_open():
                self.client.write_multiple_coils(0, self.config['light_settings']['red_light_coil'])
                print("PLC: Red light ON")
            else:
                print("PLC: Red light - Connection closed, attempting to reconnect...")
                try:
                    self.client.open()
                    if self._is_client_open():
                        self.client.write_multiple_coils(0, self.config['light_settings']['red_light_coil'])
                        print("PLC: Red light ON (after reconnect)")
                except Exception as reconnect_error:
                    print(f"PLC: Failed to reconnect - {str(reconnect_error)}")
        except Exception as e:
            print(f"Red light error: {str(e)}")

    def greenlight(self):
        """Turn on green light on PLC"""
        try:
            if self._is_client_open():
                self.client.write_multiple_coils(0, self.config['light_settings']['green_light_coil'])
                print("PLC: Green light ON")
            else:
                print("PLC: Green light - Connection closed, attempting to reconnect...")
                try:
                    self.client.open()
                    if self._is_client_open():
                        self.client.write_multiple_coils(0, self.config['light_settings']['green_light_coil'])
                        print("PLC: Green light ON (after reconnect)")
                except Exception as reconnect_error:
                    print(f"PLC: Failed to reconnect - {str(reconnect_error)}")
        except Exception as e:
            print(f"Green light error: {str(e)}")

    def yellowlight(self):
        """Turn on yellow/orange light on PLC"""
        try:
            if self._is_client_open():
                self.client.write_multiple_coils(0, self.config['light_settings']['yellow_light_coil'])
                print("PLC: Yellow light ON")
            else:
                print("PLC: Yellow light - Connection closed, attempting to reconnect...")
                try:
                    self.client.open()
                    if self._is_client_open():
                        self.client.write_multiple_coils(0, self.config['light_settings']['yellow_light_coil'])
                        print("PLC: Yellow light ON (after reconnect)")
                except Exception as reconnect_error:
                    print(f"PLC: Failed to reconnect - {str(reconnect_error)}")
        except Exception as e:
            print(f"Yellow light error: {str(e)}")

    def stop(self):
        """Stop the PLC thread"""
        self.running = False
        try:
            if self._is_client_open():
                self.client.close()
        except Exception as e:
            print(f"Error closing PLC connection: {str(e)}")

