import docker
import threading
import time
import logging

logger = logging.getLogger("discovery")

class Backend:
    def __init__(self, id, ip, port, backend_type):
        self.id = id
        self.ip = ip
        self.port = port
        self.type = backend_type # "nvidia" or "amd"
        self.active_requests = 0

    def __repr__(self):
        return f"<Backend {self.type} {self.ip}:{self.port} (Active: {self.active_requests})>"

class BackendRegistry:
    def __init__(self):
        self._backends = {} # Changed to Dict {id: Backend}
        self._lock = threading.Lock()
        self._client = None
        self._stop_event = threading.Event()

    def start(self):
        """Starts the discovery thread."""
        try:
            self._client = docker.from_env()
            # Initial load
            self._refresh()
            
            # Start watcher thread
            self.thread = threading.Thread(target=self._watch_loop, daemon=True)
            self.thread.start()
            logger.info("Discovery service started.")
        except Exception as e:
            logger.error(f"Failed to initialize Docker discovery: {e}")

    def get_instances(self, backend_type=None):
        """Returns list of Backend objects, optionally filtered by type."""
        with self._lock:
            if backend_type:
                return [b for b in self._backends.values() if b.type == backend_type]
            return list(self._backends.values())

    def _refresh(self):
        """Scans all running containers and updates the list."""
        if not self._client: return

        try:
            containers = self._client.containers.list(filters={"label": "ollama.backend=true"})
            new_map = {}
            
            for c in containers:
                try:
                    labels = c.labels
                    b_type = labels.get("ollama.type")
                    if not b_type: continue

                    # Get IP from first network
                    networks = c.attrs.get('NetworkSettings', {}).get('Networks', {})
                    if not networks: continue
                    
                    net_info = next(iter(networks.values()))
                    ip = net_info.get('IPAddress')
                    
                    if ip:
                        # Check if exists to preserve stats
                        existing = None
                        with self._lock:
                            existing = self._backends.get(c.id)

                        if existing:
                            existing.ip = ip # Update IP just in case
                            new_map[c.id] = existing
                        else:
                            b = Backend(c.id, ip, 11434, b_type)
                            new_map[c.id] = b
                except Exception as e:
                    logger.error(f"Error parsing container {c.name}: {e}")

            with self._lock:
                self._backends = new_map
            
            logger.info(f"Updated registry: {len(new_map)} instances found.")
            for b_id, b in new_map.items():
                logger.info(f"  - {b}")

        except Exception as e:
            logger.error(f"Error refreshing containers: {e}")

    def _watch_loop(self):
        """Listens for Docker events."""
        if not self._client: return
        
        # Simple polling fallback if events stream is tricky/blocking
        # For robustness in this MVP, we'll just poll every 5 seconds + listen to events if possible.
        # Docker-py events API is blocking.
        
        for event in self._client.events(decode=True):
            if self._stop_event.is_set(): break
            
            # Filter for container events
            if event.get('Type') == 'container':
                action = event.get('Action')
                if action in ['start', 'die', 'stop', 'destroy']:
                    # We could check actor attributes to see if it's our container
                    # But a full refresh is safest and cheap enough for small scale
                    attrs = event.get('Actor', {}).get('Attributes', {})
                    if attrs.get('ollama.backend') == 'true':
                        logger.info(f"Detected Docker event: {action}. Refreshing...")
                        self._refresh()

# Global Singleton
registry = BackendRegistry()
