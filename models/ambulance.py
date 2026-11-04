"""
Ambulance Data Model
Represents an ambulance with location and availability status.
"""


class Ambulance:
    """Represents an ambulance in the emergency dispatch system."""

    def __init__(self, ambulance_id, current_node, is_available=True, speed=60):
        """
        Initialize an Ambulance.

        Args:
            ambulance_id (str): Unique identifier for the ambulance.
            current_node (str): Graph node where ambulance is currently located.
            is_available (bool): Whether ambulance is available for dispatch.
            speed (float): Speed of ambulance in km/h.
        """
        self.ambulance_id = ambulance_id
        self.current_node = current_node
        self.is_available = is_available
        self.speed = speed

    def dispatch(self):
        """Mark ambulance as dispatched (unavailable)."""
        self.is_available = False

    def release(self):
        """Mark ambulance as available again."""
        self.is_available = True

    def __repr__(self):
        status = "Available" if self.is_available else "Dispatched"
        return f"Ambulance({self.ambulance_id}, at={self.current_node}, {status})"
