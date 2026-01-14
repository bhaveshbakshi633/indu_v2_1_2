#!/usr/bin/env python3
"""
Startup Health Check - Check external services and announce status via G1 speaker.
Creates /tmp/startup_complete file when done to signal health to docker-compose.
Stays running (doesn't exit) so other services can depend on it.
"""

import os
import sys
import time
import yaml
import requests
import paramiko
import logging
from pathlib import Path

# Add lib path for announcer_utils
sys.path.insert(0, '/ros2_ws/install/audio_player/lib/audio_player')

from announcer_utils import AnnouncerTTS

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('startup_health_check')

# Health check completion marker
HEALTH_MARKER = '/tmp/startup_complete'

class StartupHealthCheck:
    def __init__(self):
        self.config_path = '/ros2_ws/config/external_services.yaml'
        self.config = self._load_config()
        self.announcer = None
        self.max_retries = 3
        self.retry_delay = 10
        
    def _load_config(self):
        """Config file load karo"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            return {'services': {}, 'announcements': {}}
    
    def _wait_for_audio_services(self, timeout=60):
        """Audio services ready hone ka wait karo"""
        logger.info("Waiting for audio services...")
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                resp = requests.get('http://localhost:5050/health', timeout=2)
                if resp.status_code == 200:
                    logger.info("Audio receiver ready!")
                    return True
            except:
                pass
            time.sleep(1)
        
        logger.error("Audio services not ready after timeout")
        return False
    
    def _init_announcer(self):
        """Announcer initialize karo"""
        # Config se chatterbox settings lo
        chatterbox_cfg = self.config.get('services', {}).get('chatterbox', {})
        host = chatterbox_cfg.get('host', '192.168.123.169')
        port = chatterbox_cfg.get('port', 8000)
        
        self.announcer = AnnouncerTTS(
            chatterbox_host=host,
            chatterbox_port=port,
            audio_receiver_url='http://localhost:5050/play_audio',
            edge_voice='en-IN-NeerjaExpressiveNeural',
            gain=4.0
        )
        logger.info(f"Announcer initialized (Chatterbox: {host}:{port})")
    
    def _announce(self, message):
        """G1 speaker pe announce karo"""
        if self.announcer:
            logger.info(f"Announcing: {message}")
            self.announcer.announce(message)
        else:
            logger.warning(f"Announcer not ready, skipping: {message}")
    
    def _check_service_health(self, service_name, service_config):
        """HTTP health check karo"""
        host = service_config.get('host')
        port = service_config.get('port')
        endpoint = service_config.get('health_endpoint', '/health')
        name = service_config.get('name', service_name)
        
        url = f"http://{host}:{port}{endpoint}"
        
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except Exception as e:
            logger.debug(f"{name} check failed: {e}")
        
        return False
    
    def _ssh_restart_service(self, service_config):
        """SSH se service restart karo"""
        ssh_cfg = service_config.get('ssh', {})
        host = service_config.get('host')
        user = ssh_cfg.get('user')
        password = ssh_cfg.get('password')
        start_cmd = ssh_cfg.get('start_cmd')
        stop_cmd = ssh_cfg.get('stop_cmd')
        name = service_config.get('name', 'Unknown')
        
        if not all([host, user, password, start_cmd]):
            logger.error(f"SSH config incomplete for {name}")
            return False
        
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=user, password=password, timeout=10)
            
            # Stop first if stop_cmd exists
            if stop_cmd:
                logger.info(f"Stopping {name}...")
                stdin, stdout, stderr = client.exec_command(stop_cmd)
                stdout.read()
                time.sleep(2)
            
            # Start service
            logger.info(f"Starting {name}...")
            stdin, stdout, stderr = client.exec_command(start_cmd)
            stdout.read()
            
            client.close()
            return True
            
        except Exception as e:
            logger.error(f"SSH restart failed for {name}: {e}")
            return False
    
    def check_service(self, service_name, service_config):
        """Service check karo, restart if needed"""
        name = service_config.get('name', service_name)
        announcements = self.config.get('announcements', {})
        
        # Check if already healthy
        if self._check_service_health(service_name, service_config):
            msg = announcements.get('server_ok', '{name} server running').format(name=name)
            self._announce(msg)
            logger.info(f"{name}: OK")
            return True
        
        # Not healthy - announce and try restart
        msg = announcements.get('server_down', '{name} server not responding').format(name=name)
        self._announce(msg)
        logger.warning(f"{name}: DOWN")
        
        # SSH restart attempt
        msg = announcements.get('restart_attempt', 'Attempting to restart {name}').format(name=name)
        self._announce(msg)
        
        if not self._ssh_restart_service(service_config):
            msg = announcements.get('restart_failed', 'Failed to start {name}. Please check manually.').format(name=name)
            self._announce(msg)
            return False
        
        # Retry health check
        for attempt in range(self.max_retries):
            logger.info(f"Waiting for {name} to start (attempt {attempt + 1})...")
            time.sleep(self.retry_delay)
            
            if self._check_service_health(service_name, service_config):
                msg = announcements.get('restart_success', '{name} started successfully').format(name=name)
                self._announce(msg)
                logger.info(f"{name}: Started successfully")
                return True
        
        # Failed after retries
        msg = announcements.get('restart_failed', 'Failed to start {name}. Please check manually.').format(name=name)
        self._announce(msg)
        logger.error(f"{name}: Failed to start after {self.max_retries} retries")
        return False
    
    def run(self):
        """Main health check sequence"""
        logger.info("="*50)
        logger.info("STARTUP HEALTH CHECK STARTED")
        logger.info("="*50)
        
        # Remove old marker
        if os.path.exists(HEALTH_MARKER):
            os.remove(HEALTH_MARKER)
        
        # Wait for audio services
        if not self._wait_for_audio_services():
            logger.error("Audio services not available, cannot announce")
            # Continue anyway, but announcements won't work
        
        # Initialize announcer
        self._init_announcer()
        
        # Announce start
        announcements = self.config.get('announcements', {})
        self._announce(announcements.get('checking', 'Checking external servers'))
        
        # Check all services
        services = self.config.get('services', {})
        all_ok = True
        
        for service_name, service_config in services.items():
            if not self.check_service(service_name, service_config):
                all_ok = False
                # Continue checking other services even if one fails
        
        if all_ok:
            self._announce(announcements.get('all_ready', 'All external servers ready'))
            logger.info("All services healthy!")
            
            # Create health marker file
            Path(HEALTH_MARKER).touch()
            logger.info(f"Health marker created: {HEALTH_MARKER}")
            
            # Announce system ready
            time.sleep(1)
            self._announce(announcements.get('system_ready', 'System startup complete. Ready for operation.'))
            
            logger.info("="*50)
            logger.info("STARTUP HEALTH CHECK COMPLETE")
            logger.info("="*50)
            
            # Stay running so docker-compose healthcheck can verify
            logger.info("Staying alive for healthcheck...")
            while True:
                time.sleep(60)
        else:
            logger.error("Some services failed to start!")
            # Don't create marker - healthcheck will fail
            # Stay running but unhealthy
            while True:
                time.sleep(60)

if __name__ == '__main__':
    checker = StartupHealthCheck()
    checker.run()
