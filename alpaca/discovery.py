"""
ASCOM Alpaca UDP Discovery Protocol Handler
"""
import socket
import threading
import logging
import json

logger = logging.getLogger(__name__)


class AlpacaDiscovery:
    """ASCOM Alpaca UDP Discovery Protocol Handler"""
    
    DISCOVERY_PORT = 32227
    DISCOVERY_MESSAGE = b"alpacadiscovery1"
    
    def __init__(self, alpaca_port: int):
        self.alpaca_port = alpaca_port
        self.socket = None
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the discovery service"""
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to all interfaces on discovery port
            self.socket.bind(('', self.DISCOVERY_PORT))
            
            self.running = True
            self.thread = threading.Thread(target=self._discovery_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Alpaca Discovery service started on UDP port {self.DISCOVERY_PORT}")
        except Exception as e:
            logger.error(f"Failed to start discovery service: {e}")
            logger.warning("Discovery will not be available, but HTTP API will still work")
    
    def stop(self):
        """Stop the discovery service"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("Alpaca Discovery service stopped")
    
    def _discovery_loop(self):
        """Main discovery loop"""
        logger.info("Discovery loop started, waiting for discovery requests...")
        
        while self.running:
            try:
                # Receive discovery request
                data, addr = self.socket.recvfrom(1024)
                
                if data == self.DISCOVERY_MESSAGE:
                    logger.info(f"Discovery request from {addr[0]}:{addr[1]}")
                    self._send_discovery_response(addr)
                else:
                    logger.debug(f"Ignored non-discovery message from {addr}: {data}")
                    
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logger.error(f"Error in discovery loop: {e}")
    
    def _send_discovery_response(self, addr):
        """Send discovery response to client"""
        try:
            # Build discovery response per ASCOM spec
            response = {
                "AlpacaPort": self.alpaca_port
            }
            
            response_json = json.dumps(response).encode('utf-8')
            self.socket.sendto(response_json, addr)
            logger.info(f"Sent discovery response to {addr[0]}:{addr[1]} - Port: {self.alpaca_port}")
            
        except Exception as e:
            logger.error(f"Failed to send discovery response: {e}")
