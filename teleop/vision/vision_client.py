import zmq
import json
import numpy as np
import logging
# We assume vuer is installed in the main environment
try:
    from vuer.schemas import Text, Box, Group
except ImportError:
    # Fallback if running in an env without vuer (just for simple testing)
    Text, Box, Group = None, None, None

class VisionClient:
    def __init__(self, ip="127.0.0.1", port=55556):
        """
        Client to receive vision data from the independent Vision Service.
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        address = f"tcp://{ip}:{port}"
        print(f"[VisionClient] Connecting to {address}...")
        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
        # Only keep the last message to avoid lag
        self.socket.setsockopt(zmq.RCVHWM, 1)        # Set Receive High Water Mark to 1 to reduce buffer
        # self.socket.setsockopt(zmq.CONFLATE, 1)    # CONFLATE works mostly for PULL/PUSH, sometimes issues with SUB 
        
    def get_latest_overlays(self):
        """
        Check for new data. Returns a list of Vuer objects if new data exists, else None.
        Drains the queue to ensure we only get the absolute latest frame.
        """
        latest_msg = None
        while True:
            try:
                # Non-blocking receive
                msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                latest_msg = msg # Keep updating until queue is empty
            except zmq.Again:
                break
            except Exception as e:
                # print(f"[VisionClient] Error: {e}")
                break
        
        if latest_msg:
            detections = json.loads(latest_msg)
            return self._format_overlays(detections)
        return None

    def get_latest_raw_data(self):
        """
        Check for new data. Returns the raw detection list (dict) if new data exists, else None.
        Drains the queue to ensure we only get the absolute latest frame.
        """
        latest_msg = None
        while True:
            try:
                # Non-blocking receive
                msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                latest_msg = msg # Keep updating until queue is empty
            except zmq.Again:
                break
            except Exception as e:
               break
               
        if latest_msg:
            return json.loads(latest_msg)
        return None

    def _format_overlays(self, detections):
        """
        Convert raw detection JSON into Vuer Scene objects.
        TODO: Map 2D bounding boxes to 3D rays or HUD elements.
        """
        if Text is None:
            return []

        overlays = []
        
        # Example 1: A HUD (Head Up Display) text summary fixed in space relative to camera?
        # For now, we put it in the world frame just to demonstrate visibility.
        
        if not detections:
             return [Text(key="vision_hud", children=["NO OBJECTS"], position=[0.2, 0.2, 0.5], scale=0.05, color="red")]

        # Build a text summary string
        summary_lines = "VISION DETECTED:\n"
        for d in detections:
            label = d.get('label', 'obj')
            conf = d.get('conf', 0.0)
            summary_lines += f"{label}: {conf:.2f}\n"

        # Create a Floating Text Panel
        # In a real app, you might attach this to the head pose or hand.
        overlays.append(
            Text(
                key="vision_hud",
                children=[summary_lines],
                position=[0.2, 0.2, 0.5], # Slightly to the right and up, 0.5m away
                rotation=[0, np.pi, 0],   # Facing back towards the 'default' camera position
                scale=0.05,
                color="#00FF00",
                bg=True
            )
        )
        
        return overlays
