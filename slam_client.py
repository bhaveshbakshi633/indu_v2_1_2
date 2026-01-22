#!/usr/bin/env python3
"""
SLAM Client for Naamika Voice Assistant
HTTP client for G1 robot's SLAM navigation endpoints

Features:
- Mapping control (start, stop with map name)
- Navigation control (relocate, goto, pause, resume)
- Waypoint management (save, list, goto, delete)
- Map management (list, load, delete)

Usage:
    from slam_client import get_slam_client

    client = get_slam_client()

    # Mapping
    client.start_mapping()
    client.stop_mapping("office")  # Saves as office.pcd

    # Navigation
    client.start_navigation("office")  # Loads office.pcd
    client.goto_waypoint("lobby")
    client.pause()
    client.resume()

    # Waypoints (must be in navigation mode)
    client.save_waypoint("lobby")
    client.list_waypoints()
    client.goto_waypoint("lobby")

    # Maps
    client.list_maps()
"""

import requests
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SLAMResponse:
    """Response from SLAM endpoints"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class SLAMClient:
    """HTTP client for G1 SLAM navigation"""

    def __init__(self, host: str = "172.16.2.242", port: int = 5052, timeout: float = 30.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.enabled = True

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> SLAMResponse:
        """Make HTTP request to SLAM endpoint"""
        if not self.enabled:
            return SLAMResponse(success=False, message="SLAM client disabled")

        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=self.timeout)
            else:
                response = requests.post(url, json=data or {}, timeout=self.timeout)

            result = response.json()

            # Check for success
            if response.status_code == 200 and result.get("status") in ("ok", "success"):
                return SLAMResponse(
                    success=True,
                    message=result.get("message", result.get("stdout", "Success")),
                    data=result
                )
            else:
                error_msg = result.get("message", result.get("stderr", result.get("stdout", "Unknown error")))
                return SLAMResponse(
                    success=False,
                    message=error_msg,
                    data=result
                )

        except requests.exceptions.ConnectionError:
            logger.error(f"[SLAM] Connection failed to {url}")
            return SLAMResponse(success=False, message="Cannot connect to G1 robot")
        except requests.exceptions.Timeout:
            logger.error(f"[SLAM] Request timeout: {url}")
            return SLAMResponse(success=False, message="Request timed out")
        except Exception as e:
            logger.error(f"[SLAM] Request failed: {e}")
            return SLAMResponse(success=False, message=str(e))

    # ============================================================
    # MAPPING
    # ============================================================

    def start_mapping(self) -> SLAMResponse:
        """Start SLAM mapping mode"""
        logger.info("[SLAM] Starting mapping mode")
        return self._request("POST", "/slam/start_mapping")

    def stop_mapping(self, map_name: str = "default_map") -> SLAMResponse:
        """Stop mapping and save map with given name"""
        logger.info(f"[SLAM] Stopping mapping, saving as '{map_name}'")
        return self._request("POST", "/slam/stop_mapping", {"map_name": map_name})

    # ============================================================
    # NAVIGATION
    # ============================================================

    def start_navigation(self, map_name: str = "default_map") -> SLAMResponse:
        """Start navigation mode (relocate on map)"""
        logger.info(f"[SLAM] Starting navigation with map '{map_name}'")
        return self._request("POST", "/slam/relocate", {"map_name": map_name})

    def goto_pose(self, x: float, y: float, z: float = 0.0,
                  qx: float = 0.0, qy: float = 0.0,
                  qz: float = 0.0, qw: float = 1.0) -> SLAMResponse:
        """Navigate to specific coordinates"""
        logger.info(f"[SLAM] Navigating to pose ({x:.2f}, {y:.2f})")
        return self._request("POST", "/slam/goto", {
            "x": x, "y": y, "z": z,
            "qx": qx, "qy": qy, "qz": qz, "qw": qw
        })

    def goto_waypoint(self, name: str) -> SLAMResponse:
        """Navigate to a saved waypoint"""
        logger.info(f"[SLAM] Navigating to waypoint '{name}'")
        return self._request("POST", "/slam/goto_waypoint", {"name": name})

    def pause(self) -> SLAMResponse:
        """Pause current navigation"""
        logger.info("[SLAM] Pausing navigation")
        return self._request("POST", "/slam/pause")

    def resume(self) -> SLAMResponse:
        """Resume paused navigation"""
        logger.info("[SLAM] Resuming navigation")
        return self._request("POST", "/slam/resume")

    def get_status(self) -> SLAMResponse:
        """Get current SLAM status including pose, waypoints, maps"""
        return self._request("GET", "/slam/status")

    def get_pose(self) -> SLAMResponse:
        """Get current robot pose"""
        return self._request("GET", "/slam/pose")

    # ============================================================
    # WAYPOINTS
    # ============================================================

    def save_waypoint(self, name: str) -> SLAMResponse:
        """Save current position as waypoint (must be in navigation mode)"""
        logger.info(f"[SLAM] Saving waypoint '{name}'")
        return self._request("POST", "/waypoint/save", {"name": name})

    def list_waypoints(self) -> SLAMResponse:
        """List all saved waypoints"""
        return self._request("GET", "/waypoint/list")

    def delete_waypoint(self, name: str) -> SLAMResponse:
        """Delete a waypoint"""
        logger.info(f"[SLAM] Deleting waypoint '{name}'")
        return self._request("POST", "/waypoint/delete", {"name": name})

    def get_waypoint_names(self) -> List[str]:
        """Get list of waypoint names (convenience method)"""
        response = self.list_waypoints()
        if response.success and response.data:
            return response.data.get("names", [])
        return []

    # ============================================================
    # MAPS
    # ============================================================

    def list_maps(self) -> SLAMResponse:
        """List all saved maps"""
        return self._request("GET", "/map/list")

    def delete_map(self, name: str) -> SLAMResponse:
        """Delete a map from registry"""
        logger.info(f"[SLAM] Deleting map '{name}'")
        return self._request("POST", "/map/delete", {"name": name})

    def get_map_names(self) -> List[str]:
        """Get list of map names (convenience method)"""
        response = self.list_maps()
        if response.success and response.data:
            return response.data.get("names", [])
        return []


# Singleton instance
_slam_client: Optional[SLAMClient] = None


def get_slam_client(host: str = "172.16.2.242", port: int = 5052) -> SLAMClient:
    """Get or create singleton SLAMClient instance"""
    global _slam_client
    if _slam_client is None:
        _slam_client = SLAMClient(host=host, port=port)
    return _slam_client


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== SLAM Client Test ===\n")

    client = get_slam_client()

    # Test status
    print("Getting SLAM status...")
    status = client.get_status()
    print(f"Status: {status}\n")

    # Test waypoint list
    print("Listing waypoints...")
    waypoints = client.list_waypoints()
    print(f"Waypoints: {waypoints}\n")

    # Test map list
    print("Listing maps...")
    maps = client.list_maps()
    print(f"Maps: {maps}\n")

    print("Test complete!")
