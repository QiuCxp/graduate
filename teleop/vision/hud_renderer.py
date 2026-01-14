import cv2
import numpy as np
import time

class HUDRenderer:
    """
    A futuristic HUD renderer for teleoperation.
    Responsible for drawing overlays (bounding boxes, tags, crosshairs) on the camera feed.
    """
    def __init__(self):
        # Color Palette (BGR)
        self.COLOR_PRIMARY = (0, 255, 255)    # Yellow/Cyan mix for Sci-Fi look
        self.COLOR_SECONDARY = (255, 100, 0)  # Orange for highlights
        self.COLOR_TEXT = (255, 255, 255)     # White
        self.COLOR_BG = (0, 0, 0)             # Black (for backgrounds)
        
        # Font settings
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.5
        self.THICKNESS = 1

    def draw_hud(self, image, detections):
        """
        Main entry point to draw all HUD elements on the image.
        :param image: BGR numpy array (the camera frame)
        :param detections: List of detection dicts {'box': [cx, cy, w, h], 'label': str, 'conf': float}
        :return: The processed image (drawn in-place)
        """
        if image is None:
            return image
            
        h, w, _ = image.shape
        
        # 1. Draw Static Elements (Minimalist)
        self._draw_crosshair(image, w, h)
        self._draw_status(image, w, h, len(detections))
        # self._draw_overlay_frame(image, w, h) # Removed for Minimal AR style
        
        # 2. Draw Dynamic Elements (Detections)
        if detections:
            # First pass: Draw all fills (so they are behind text)
            overlay = image.copy()
            has_fill = False
            
            for det in detections:
                 if self._draw_segmentation_fill(overlay, det, w, h):
                     has_fill = True
            
            if has_fill:
                alpha = 0.3  # Transparency factor
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Second pass: Draw contours and text
            for det in detections:
                self._draw_ar_highlight(image, det, w, h)
                
        return image

    def _draw_crosshair(self, img, w, h):
        """Draws a minimalist central dot."""
        cx, cy = w // 2, h // 2
        # Just a small glowing dot
        cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1) 
        # Faint outer ring
        cv2.circle(img, (cx, cy), 15, (255, 255, 255), 1)

    def _draw_status(self, img, w, h, obj_count):
        """Draws minimal status count."""
        if obj_count > 0:
            status_text = f"{obj_count} TARGETS"
            cv2.putText(img, status_text, (w//2 - 40, 30), self.FONT, 0.4, (255, 255, 255), 1)

    def _draw_segmentation_fill(self, overlay, det, w, h):
        """Draws the filled polygon on the overlay layer."""
        polygon = det.get('polygon', [])
        conf = det.get('conf', 0)
        
        if not polygon:
            return False

        # Convert normalized [0..1] polygon to pixel coords [w, h]
        pts = np.array(polygon, dtype=np.float32)
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = pts.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Color based on confidence (Cyan to White)
        color = self.COLOR_PRIMARY
        if conf > 0.8:
            color = (255, 200, 0) # Goldish
        
        cv2.fillPoly(overlay, [pts], color)
        return True

    def _draw_ar_highlight(self, img, det, w, h):
        """Draws the contour line and floating label."""
        polygon = det.get('polygon', [])
        box = det.get('box', [])
        label = det.get('label', '?')
        conf = det.get('conf', 0)

        color = self.COLOR_PRIMARY
        if conf > 0.8:
            color = (255, 200, 0)
            
        # 1. Draw Contour if available, else Box
        if polygon:
            pts = np.array(polygon, dtype=np.float32)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = pts.astype(np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw thin contour
            cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
            
            # Find top point for label
            # top_point = tuple(pts[pts[:, :, 1].argmin()][0])
            # Or use bounding box top center
        
        # Calculate Box coordinates for label positioning (fallback or primary)
        if len(box) == 4:
            cx, cy, bw, bh = box
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            # y2 = int((cy + bh/2) * h)
            
            # Draw minimalist corners if no polygon, or just to accent
            if not polygon:
                self._draw_simple_corners(img, x1, y1, x2-x1, int(bh*h), color)

            # 2. Draw Floating Label
            label_text = f"{label.upper()}"
            (tw, th), _ = cv2.getTextSize(label_text, self.FONT, 0.4, 1)
            
            # Position above object
            text_x = int((cx * w) - tw / 2)
            text_y = y1 - 10
            
            # Optional: Tiny background for readability
            # cv2.rectangle(img, (text_x - 2, text_y - th - 2), (text_x + tw + 2, text_y + 2), (0,0,0), -1)
            
            cv2.putText(img, label_text, (text_x, text_y), self.FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{int(conf*100)}%", (text_x + tw + 5, text_y), self.FONT, 0.3, color, 1, cv2.LINE_AA)

    def _draw_simple_corners(self, img, x, y, w, h, color):
        """Fallback simple corners."""
        l = min(w, h) // 5
        cv2.line(img, (x, y), (x + l, y), color, 1)
        cv2.line(img, (x, y), (x, y + l), color, 1)
        cv2.line(img, (x+w, y), (x+w-l, y), color, 1)
        cv2.line(img, (x+w, y), (x+w, y+l), color, 1)
        cv2.line(img, (x, y+h), (x+l, y+h), color, 1)
        cv2.line(img, (x, y+h), (x, y+h-l), color, 1)
        cv2.line(img, (x+w, y+h), (x+w-l, y+h), color, 1)
        cv2.line(img, (x+w, y+h), (x+w, y+h-l), color, 1)

        callout_x = x2 + 10
        callout_y = y1 - 20
        
        cv2.line(img, (x2, y1), (x2 + 10, y1 - 10), color, 1) # Diagonal
        cv2.line(img, (x2 + 10, y1 - 10), (callout_x + 80, y1 - 10), color, 1) # Horizontal
        
        label_text = f"{label.upper()} {int(conf*100)}%"
        
        # Text Background
        (text_w, text_h), _ = cv2.getTextSize(label_text, self.FONT, 0.4, 1)
        cv2.rectangle(img, (callout_x + 10, y1 - 25), (callout_x + 10 + text_w, y1 - 10), self.COLOR_BG, -1)
        
        # Text
        cv2.putText(img, label_text, (callout_x + 10, y1 - 14), self.FONT, 0.4, color, 1)
