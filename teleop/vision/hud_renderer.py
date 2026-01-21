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

    def render(self, image, detections, task_id=0): # For legacy compatibility with old entry point name if needed
        return self.render_main(image, detections, task_id)

    def draw_hud(self, image, detections):
        # Redirect old method to new logic (default to Task 0)
        return self.render_main(image, detections, task_id=0)

    def render_main(self, image, detections, task_id=0):
        """
        Main render loop.
        task_id: 
          1: Arrange Items (Cup -> Book)
          2: Crisis Response (Cup Rescue -> Place near Bottle)
          3: Grasp Fruit (Apple, Orange)
          4: Navigate
        """
        h, w = image.shape[:2]
        
        # --- Task Specific Dispatch ---
        if task_id == 1:
            return self.render_task_arrange(image, detections)
        elif task_id == 2:
            return self.render_task_crisis_response(image, detections)
        elif task_id == 3:
            return self.render_task_human_interaction(image, detections)

        # Define relevant objects for each task
        target_labels = []
        if task_id == 3:
            target_labels = ['apple', 'orange']
            
        # Filter detections for the general layer based on task
        # If task_id is 0 (Ready) or 4 (Nav), show everything.
        # If task_id is 3, show only targets.
        relevant_detections = detections
        if target_labels:
             relevant_detections = [d for d in detections if d.get('label', '').lower() in target_labels]

        # --- 1. General Drawing Layer (Standard HUD) ---
        # Static Elements
        self._draw_crosshair(image, w, h)
        self._draw_status(image, w, h, len(relevant_detections))
        
        # Dynamic Elements (Detections)
        if relevant_detections:
            # Pass 1: Segmentation Fills (Translucent)
            overlay = image.copy()
            has_fill = False
            for det in relevant_detections:
                 if self._draw_segmentation_fill(overlay, det, w, h):
                     has_fill = True
            
            if has_fill:
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Pass 2: Contours & Labels (AR Highlight)
            for det in relevant_detections:
                self._draw_ar_highlight(image, det, w, h)
        
        # --- 2. Task Specific Layer (Overlays) ---
        # if task_id == 3: # Old Grasp Fruit
        #    return self.render_task_grasp(image, detections)
        
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

    def draw_notification(self, image, task_id, elapsed_time):
        """
        Draw a transient notification for task selection.
        :param task_id: The ID of the selected task (1-4)
        :param elapsed_time: Time in seconds since the task was selected
        """
        if elapsed_time > 3.0: # Show for 3 seconds
            return
            
        h, w, _ = image.shape
        
        task_names = {
            1: "TASK 1: POS MOVE",
            2: "TASK 2: CLEAN UP", 
            3: "TASK 3: HUMAN SAFETY",
            4: "TASK 4: NAVIGATION"
        }
        
        text = task_names.get(task_id, f"TASK {task_id}")
        
        # Calculate fade out alpha
        alpha = 1.0
        if elapsed_time > 2.0:
            alpha = 1.0 - (elapsed_time - 2.0)
            
        # Text settings
        font_scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, self.FONT, font_scale, thickness)
        
        cx, cy = w // 2, h // 2
        
        # Draw background pill
        pad = 20
        # Create overlay for transparency
        overlay = image.copy()
        cv2.rectangle(overlay, 
                     (cx - tw//2 - pad, cy - th - pad), 
                     (cx + tw//2 + pad, cy + pad), 
                     (0, 0, 0), -1)
        
        # Blend background
        cv2.addWeighted(overlay, 0.6 * alpha, image, 1 - (0.6 * alpha), 0, image)
        
        # Draw text
        # Note: OpenCV putText doesn't support alpha directly easily without layers, 
        # but for HUD, simple white text is usually fine.
        # To strictly do alpha text, we'd need another mask, but let's keep it simple for perf.
        color = (255, 255, 255)
        # We can simulate alpha by blending color with background if needed, but white is fine.
        
    # [REMOVED DUPLICATE RENDER METHOD]
    # The primary render entry point is at the top of the class, enabling render_main to control the flow.
    # The mapping here was also inconsistent with the UI.


    def render_task_crisis_response(self, image, detections):
        """
        Task 2: Crisis Response Workflow (REVISED)
        Objects: Bottle(39), Cup(41), Book(73)
        Roles:
          - Bottle: Active Object (Crisis Candidate)
          - Cup: Anchor/Reference
        Logic:
        1. Check Bottle State: If Bottle.Height < Cup.Height -> Crisis Mode (RED ALERT).
        2. If Upright: Isolation Mode. Target = Near Cup.
        3. Check Obstacles: If Book in target zone -> REMOVE BOOK.
        4. Success: Bottle in zone & Book out of zone.
        """
        if not detections:
            return image

        h, w = image.shape[:2]
        
        # 1. Gather Object Data
        cup_info = None
        bottle_info = None
        book_info = None
        
        for det in detections:
            label = det.get('label', '').lower()
            box = det.get('box', [])
            if len(box) != 4: continue
            
            # Convert to pixels
            cx, cy = int(box[0]*w), int(box[1]*h)
            bw, bh = int(box[2]*w), int(box[3]*h)
            
            info = {
                'box': (cx, cy, bw, bh), # pixels
                'det': det,
                'bottom': (cx, cy + bh//2),
                'height': bh 
            }
            
            if label == 'cup': cup_info = info
            elif label == 'bottle': bottle_info = info
            elif label == 'book': book_info = info

        # --- STATE MACHINE ---
        
        # S0: Need Bottle + Cup for Reference/Target
        if not bottle_info:
            cv2.putText(image, "SEARCHING FOR BOTTLE...", (w//2-150, h//2), self.FONT, 0.8, (255, 255, 255), 2)
            return image
            
        # Check Dimensions for Topple Detection (Crisis)
        # Bottle Info
        bx, by, bw, bh = bottle_info['box']
        
        # Condition: Aspect Ratio Check (Robust)
        # If Width is close to Height or larger (w > 0.6*h), it's likely horizontal or tilted towards camera.
        # Normal standing bottle is usually h > 2.5*w (w < 0.4*h)
        # We relax the threshold from 0.9 to 0.6 to catch "foreshortened" toppled bottles.
        is_toppled = bw > (bh * 0.6)
        
        # --- STAGE 1: CRISIS MODE (Emergency) ---
        if is_toppled:
            # Effect: Red Alert (Persistent)
            
            # 1. Red Box on Bottle (Thick, Non-flashing)
            cv2.rectangle(image, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 0, 255), 3)
            
            # 2. Warning Labels (Top Right Corner, Non-flashing)
            # Create a background plate for high visibility
            msg1 = "!!! ALERT: BOTTLE SPILLED !!!"
            msg2 = "ACTION: UPRIGHT BOTTLE"
            
            # Top Right coords
            start_x = w - 350
            start_y = 50
            
            # Text background
            cv2.rectangle(image, (start_x - 10, start_y - 30), (w, start_y + 40), (0, 0, 150), -1) 
            
            cv2.putText(image, msg1, (start_x, start_y), self.FONT, 0.7, (255, 255, 255), 2)
            cv2.putText(image, msg2, (start_x, start_y + 25), self.FONT, 0.6, (200, 200, 200), 1)
            
            # 3. Block other overlays (Focus on this task)
            return image

        # --- STAGE 2: ISOLATION MODE (Normal Operation) ---
        # Bottle is upright. Now we need to move it near Cup.
        
        if not cup_info:
            # Bottle is upright, but we need Cup to define the zone
            cv2.putText(image, "BOTTLE SECURED. FIND CUP (ANCHOR)...", (w//2-200, h//2+200), self.FONT, 0.7, (255, 255, 255), 2)
            # Draw Bottle is calm yellow
            cv2.rectangle(image, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 255, 255), 2)
            return image

        # Cup Info (Ref)
        cx, cy, cw, ch = cup_info['box']
        
        # Draw Bottle (Active Object - Yellow)
        cv2.rectangle(image, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 255, 255), 2)
        cv2.putText(image, "TARGET: BOTTLE", (bx-bw//2, by-bh//2-10), self.FONT, 0.6, (0, 255, 255), 2)
             
        # Generate Isolation Zone (Relative to Cup)
        # Highlight Cup (Anchor)
        cv2.rectangle(image, (cx-cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), (255, 0, 0), 2)
        cv2.putText(image, "ANCHOR: CUP", (cx-cw//2, cy-ch//2-10), self.FONT, 0.5, (255, 0, 0), 1)

        # Definition: 20cm (approx 1 Cup width) to the Right (or Left?)
        # Let's say Right of Cup
        zone_offset = int(cw * 1.5)
        zone_x = cx + zone_offset
        zone_y = cy + ch // 2 # Ground level
        zone_w = int(bw * 1.5) # Based on Bottle width
        zone_h = int(bw * 1.0) # Flat elliptical zone
        
        target_center = (zone_x, zone_y)
        
        # --- STAGE 3: CLEARANCE & COMPLETION ---
        
        # Check 1: Is Book blocking the zone?
        is_blocked = False
        if book_info:
            book_box = book_info['box'] # cx, cy, w, h
            bk_x, bk_y, bk_w, bk_h = book_box
            
            z_left = zone_x - zone_w//2
            z_right = zone_x + zone_w//2
            b_left = bk_x - bk_w//2
            b_right = bk_x + bk_w//2
            
            # Check overlap
            if not (b_right < z_left or b_left > z_right):
                is_blocked = True
                # Draw Book ERROR
                cv2.rectangle(image, (b_left, bk_y-bk_h//2), (b_right, bk_y+bk_h//2), (0, 0, 255), 3)
                cv2.line(image, (b_left, bk_y-bk_h//2), (b_right, bk_y+bk_h//2), (0, 0, 255), 3)
                cv2.line(image, (b_right, bk_y-bk_h//2), (b_left, bk_y+bk_h//2), (0, 0, 255), 3)
                cv2.putText(image, "OBSTACLE: MOVE BOOK!", (b_left, bk_y-bk_h//2-10), self.FONT, 0.6, (0, 0, 255), 2)

        # Check 2: Is Bottle in Zone?
        bottle_bottom = bottle_info['bottom']
        dist_to_zone = np.linalg.norm(np.array(bottle_bottom) - np.array(target_center))
        in_zone_threshold = zone_w * 0.6
        is_in_zone = dist_to_zone < in_zone_threshold
        
        # --- RENDER ZONE & GUIDANCE ---
        
        if is_blocked:
             # Zone is BLOCKED (Red)
             cv2.ellipse(image, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 0, 255), 2)
             cv2.putText(image, "ZONE BLOCKED", (zone_x-60, zone_y+30), self.FONT, 0.6, (0, 0, 255), 2)
        elif is_in_zone:
             # SUCCESS!
             # Green Zone + Checkmark
             cv2.ellipse(image, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 255, 0), 4) # Thick Green
             # Big Checkmark
             cx, cy = w // 2, h // 2
             cv2.circle(image, (cx, cy), 50, (0, 255, 0), -1)
             pts = np.array([[cx - 20, cy], [cx - 5, cy + 20], [cx + 30, cy - 30]], np.int32)
             cv2.polylines(image, [pts], False, (255, 255, 255), 8, cv2.LINE_AA)
        else:
             # Guidance Mode
             # Draw Zone
             cv2.ellipse(image, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 255, 0), 2, cv2.LINE_AA)
             cv2.putText(image, "ISOLATION ZONE", (zone_x-60, zone_y+30), self.FONT, 0.5, (0, 255, 0), 1)
             
             # Draw Path Arrow
             self._draw_animated_dashed_line(image, bottle_bottom, target_center, (0, 255, 255))

        return image
        
        # --- RENDER ZONE & GUIDANCE ---
        
        if is_blocked:
             # Zone is BLOCKED (Red/Orange Warning Zone)
             cv2.ellipse(image, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 0, 255), 2)
             cv2.putText(image, "ZONE BLOCKED", (zone_x-60, zone_y+30), self.FONT, 0.6, (0, 0, 255), 2)
        elif is_in_zone:
             # SUCCESS!
             # Green Zone + Checkmark
             cv2.ellipse(image, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 255, 0), 4) # Thick Green
             # Big Checkmark
             cx, cy = w // 2, h // 2
             cv2.circle(image, (cx, cy), 50, (0, 255, 0), -1)
             pts = np.array([[cx - 20, cy], [cx - 5, cy + 20], [cx + 30, cy - 30]], np.int32)
             cv2.polylines(image, [pts], False, (255, 255, 255), 8, cv2.LINE_AA)
        else:
             # Guidance Mode (Blue/Green Zone + Path)
             # Draw Zone
             cv2.ellipse(image, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 255, 0), 2, cv2.LINE_AA)
             cv2.putText(image, "ISOLATION ZONE", (zone_x-60, zone_y+30), self.FONT, 0.5, (0, 255, 0), 1)
             
             # Draw Path Arrow
             self._draw_animated_dashed_line(image, cup_bottom, target_center, (0, 255, 255))

        return image

    def render_task_grasp(self, image, detections):
        """Task 1: Grasp Fruit (Apple, Orange) - Draw 3D brackets + Center Point"""
        if not detections:
            return image
            
        h, w = image.shape[:2]
        
        for det in detections:
            label = det.get('label', '').lower()
            if label not in ['apple', 'orange']:
                continue
                
            box = det['box']
            conf = det['conf']
            
            # center x, center y, width, height (normalized)
            cx_px = int(box[0] * w)
            cy_px = int(box[1] * h)
            w_px = int(box[2] * w)
            h_px = int(box[3] * h)
            
            # Color
            color = (0, 165, 255) if label == 'orange' else (0, 0, 255) # BGR: Orange or Red
            
            # Draw Corner Brackets (HUD Style) instead of full box
            self._draw_corners(image, cx_px, cy_px, w_px, h_px, color, 20)
            
            # Draw Grasp Target (Crosshair in center)
            cv2.drawMarker(image, (cx_px, cy_px), color, cv2.MARKER_CROSS, 20, 2)
            
            # Label
            cv2.putText(image, f"{label.upper()} {int(conf*100)}%", (cx_px - w_px//2, cy_px - h_px//2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return image

    def render_task_human_interaction(self, image, detections):
        """
        Task 3: Human-in-need, Teleoperation-required
        Scenario: Robot cannot operate autonomously near human.
        Objects: Person(0), Bottle(39, Risk Object), Cup(41, Safe Anchor)
        Zones:
         - Person Polygon + Semi-transparent Overlay
         - Safe Zone: Expanded Mask Area (Body + 20%)
        """
        h, w = image.shape[:2]
        
        person_info = None
        cup_info = None
        bottle_info = None
        
        # 1. Identify Objects
        for det in detections:
            label = det.get('label', '').lower()
            box = det.get('box', []) # cx, cy, w, h normalized
            if len(box) != 4: continue
            
            # Convert to pixels
            cx_px = int(box[0] * w)
            cy_px = int(box[1] * h)
            w_px = int(box[2] * w)
            h_px = int(box[3] * h)
            
            info = {
                'box': (cx_px, cy_px, w_px, h_px), # Pixel box [cx, cy, w, h]
                'norm_box': box,
                'center': (cx_px, cy_px),
                'label': label,
                'polygon': det.get('polygon', [])
            }
            
            if label == 'person':
                # Just take the first person found or largest confidence
                if person_info is None or det.get('conf') > person_info.get('conf', 0):
                    info['conf'] = det.get('conf', 0)
                    person_info = info
            elif label == 'cup':
                cup_info = info
            elif label == 'bottle':
                bottle_info = info

        # --- DRAWING ---
        
        # 1. Person & Risk Zone (Mask Based)
        is_high_risk = False
        person_mask_dilated = None
        
        if person_info:
            px, py, pw, ph = person_info['box']
            polygon = person_info['polygon']
            
            has_mask = False
            pts_px = None
            
            if len(polygon) > 0:
                # Convert polygon to pixels
                pts = np.array(polygon, dtype=np.float32)
                pts[:, 0] *= w
                pts[:, 1] *= h
                pts_px = pts.astype(np.int32).reshape((-1, 1, 2))
                
                # Create mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts_px], 255)
                
                # --- Expand the Zone (20% Dilation) ---
                # A 20% expansion of "area" or "size"? 
                # Usually means margin. Let's use 20% of the minor axis (width) as kernel size.
                kernel_size = int(min(pw, ph) * 0.2)
                if kernel_size % 2 == 0: kernel_size += 1
                kernel_size = max(3, kernel_size) # Min kernel
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                person_mask_dilated = cv2.dilate(mask, kernel, iterations=1)
                
                has_mask = True
                
                # --- Draw Semi-transparent Overlay ---
                overlay = image.copy()
                
                # 1. Draw Dilation (Restricted Zone) - Red tint
                # Use the dilated mask to paint red
                # Find contours of dilated mask to draw efficient filled poly
                dilated_contours, _ = cv2.findContours(person_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, dilated_contours, -1, (0, 0, 200), -1) # Dark Red Fill
                
                # 2. Draw Person (Actual) - Orange tint
                cv2.fillPoly(overlay, [pts_px], (0, 140, 255)) # Orange

                # Blend
                # Zone alpha 0.3, Person alpha 0.5 (additive logic approx)
                cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
                
                # Draw Boundaries
                cv2.drawContours(image, dilated_contours, -1, (0, 0, 255), 2) # Red Border
                cv2.polylines(image, [pts_px], True, (255, 255, 255), 1) # White Person Outline
                
                # Label
                top_idx = np.argmin(pts_px[:, 0, 1])
                top_pt = tuple(pts_px[top_idx][0])
                cv2.putText(image, "NO-GO ZONE (+20%)", (top_pt[0]-40, top_pt[1]-10), self.FONT, 0.5, (0, 0, 255), 2)

            else:
                 # Fallback to Box if no polygon
                 # (Original logic)
                 margin_w, margin_h = int(pw * 0.2), int(ph * 0.2)
                 p1 = (max(0, px - pw//2 - margin_w), max(0, py - ph//2 - margin_h))
                 p2 = (min(w, px + pw//2 + margin_w), min(h, py + ph//2 + margin_h))
                 cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
                 cv2.putText(image, "NO-GO ZONE (BOX)", (p1[0], p1[1]-5), self.FONT, 0.5, (0, 0, 255), 1)
                 
                 # Create a dummy box mask for collision
                 person_mask_dilated = np.zeros((h, w), dtype=np.uint8)
                 cv2.rectangle(person_mask_dilated, p1, p2, 255, -1)


        # 2. Check Risk (Bottle vs Human Zone Mask)
        if bottle_info and person_mask_dilated is not None:
            cx, cy = bottle_info['center'] # Use center point
            # Check if center is in mask
            # Ensure coords are bounds
            cx = max(0, min(w-1, cx))
            cy = max(0, min(h-1, cy))
            
            if person_mask_dilated[cy, cx] > 0:
                is_high_risk = True

        # --- LOGIC FLOW ---
        
        if is_high_risk:
            # === STAGE 1: HIGH RISK / EXTRACTION ===
            
            # High Visibility Warning
            cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 10) # Screen border red
            
            # Alert Banner
            alert_bg_color = (0, 0, 200)
            cv2.rectangle(image, (w//2 - 250, 50), (w//2 + 250, 150), alert_bg_color, -1)
            cv2.putText(image, "!!! HUMAN PROXIMITY ALERT !!!", (w//2 - 220, 90), self.FONT, 0.8, (255, 255, 255), 2)
            cv2.putText(image, "AUTONOMY DISABLED - TELEOP REQUIRED", (w//2 - 230, 130), self.FONT, 0.6, (255, 255, 0), 2)
            
            # Highlight Bottle as Risk Object
            if bottle_info:
                cx, cy, cw, ch = bottle_info['box']
                cv2.rectangle(image, (cx-cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), (0, 0, 255), 3)
                cv2.putText(image, "RISK OBJECT", (cx-cw//2, cy+ch//2+20), self.FONT, 0.6, (0, 0, 255), 2)
                
                # Draw Arrow AWAY from Person (Centroid based)
                if person_info:
                    px, py = person_info['center']
                    # Vector from Person -> Bottle
                    vec = np.array([cx - px, cy - py])
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                        # Target point for extraction (just 'away')
                        target = (int(cx + vec[0]*100), int(cy + vec[1]*100))
                        
                        # Draw arrow
                        cv2.arrowedLine(image, (cx, cy), target, (0, 255, 255), 4, tipLength=0.3)
                        cv2.putText(image, "MOVE AWAY", target, self.FONT, 0.6, (0, 255, 255), 2)

        elif bottle_info:
            # === STAGE 2: PLACEMENT (SAFE) ===
            
            # Bottle is safe, highlight it normally (Yellow)
            cx, cy, cw, ch = bottle_info['box']
            cv2.rectangle(image, (cx-cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), (0, 255, 255), 2)
            cv2.putText(image, "OBJECT SECURED", (cx-cw//2, cy-ch//2-10), self.FONT, 0.6, (0, 255, 255), 2)
            
            if cup_info:
                # We have an anchor (Cup) -> Show Safe Placement Zone
                bx, by, bw, bh = cup_info['box']
                
                # Draw Anchor
                cv2.rectangle(image, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (255, 0, 0), 2)
                cv2.putText(image, "ANCHOR: CUP", (bx-bw//2, by-bh//2-10), self.FONT, 0.5, (255, 0, 0), 1)
                
                # Define Safe Zone (e.g., Left of Cup)
                # Let's say 20cm Left
                zone_offset = int(bw * 2.0)
                zone_x = bx - zone_offset
                zone_y = by + bh // 2
                zone_w = int(cw * 1.5)
                zone_h = int(cw * 1.0)
                target_center = (zone_x, zone_y)
                
                # Check if Bottle is in Zone
                bottle_bottom = (cx, cy + ch//2)
                dist = np.linalg.norm(np.array(bottle_bottom) - np.array(target_center))
                in_zone = dist < (zone_w * 0.6)
                
                if in_zone:
                    # SUCCESS
                    cv2.ellipse(image, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 255, 0), 4)
                    
                    # Big Success Check
                    center_x, center_y = w // 2, h // 2
                    cv2.circle(image, (center_x, center_y), 50, (0, 255, 0), -1)
                    # Checkmark
                    pts = np.array([[center_x - 20, center_y], [center_x - 5, center_y + 20], [center_x + 30, center_y - 30]], np.int32)
                    cv2.polylines(image, [pts], False, (255, 255, 255), 8, cv2.LINE_AA)
                    cv2.putText(image, "TASK COMPLETE", (center_x - 70, center_y + 80), self.FONT, 0.8, (0, 255, 0), 2)
                else:
                    # Guidance
                    cv2.ellipse(image, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 255, 0), 2)
                    cv2.putText(image, "SAFE ZONE", (zone_x-40, zone_y+30), self.FONT, 0.5, (0, 255, 0), 1)
                    
                    # Path
                    self._draw_animated_dashed_line(image, bottle_bottom, target_center, (0, 255, 0))
            else:
                 # No cup found
                 cv2.putText(image, "FIND CUP (ANCHOR)...", (w//2-150, h//2+100), self.FONT, 0.7, (255, 255, 255), 2)
        
        else:
            # No bottle found
            cv2.putText(image, "SEARCHING FOR BOTTLE...", (w//2-150, h//2), self.FONT, 0.8, (255, 255, 255), 2)

        return image

    def _draw_corners(self, img, cx, cy, w, h, color, length):

        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = cx + w//2, cy + h//2
        
        # Top Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, 2)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, 2)
        # Top Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, 2)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, 2)
        # Bottom Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, 2)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, 2)
        # Bottom Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, 2)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, 2)

    def render_task_arrange(self, image, detections):
        """
        Task 1: Arrange Items (Move Bottle to Book's Right Side)
        - Bottle: Yellow fill + Yellow border
        - Target: Green perspective circle (to the right of Book)
        - Path: Bottle -> Target (Animated Yellow Dashed Line)
        - Success: Clear all + Green Checkmark
        """
        if not detections:
            return image

        h, w = image.shape[:2]
        bottle_info = None
        book_info = None

        # 1. Identify Objects
        for det in detections:
            label = det.get('label', '').lower()
            box = det.get('box', []) # [cx, cy, w, h] normalized
            if len(box) != 4:
                continue

            # Convert to pixels
            cx_px = int(box[0] * w)
            cy_px = int(box[1] * h)
            w_px = int(box[2] * w)
            h_px = int(box[3] * h)
            
            # Bottom Center (Ground Point)
            bottom_center = (cx_px, cy_px + h_px // 2)

            info = {
                'box': box,
                'px_box': (cx_px, cy_px, w_px, h_px),
                'bottom': bottom_center,
                'polygon': det.get('polygon', []),
                'det': det
            }

            if label == 'bottle':
                bottle_info = info
            elif label == 'book':
                book_info = info

        # 2. Logic & Rendering
        
        # DEBUG: Show what we see at the top
        debug_text = f"DEBUG: Seen {len(detections)} objs."
        if bottle_info: debug_text += " [BOTTLE]"
        if book_info: debug_text += " [BOOK]"
        cv2.putText(image, debug_text, (10, 60), self.FONT, 0.7, (0, 0, 255), 2)
        
        # 1. Always Draw Anchor (Book) if present
        if book_info:
             bx, by, bw, bh = book_info['px_box']
             cv2.rectangle(image, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (255, 0, 0), 2) # Blue
             cv2.putText(image, "ANCHOR: BOOK", (bx-bw//2, by-bh//2-10), self.FONT, 0.6, (255, 0, 0), 2)

        # 2. If no bottle, show waiting status
        if not bottle_info:
             if not book_info:
                 cv2.putText(image, "WAITING FOR BOTTLE & BOOK...", (w//2-100, h//2), self.FONT, 0.7, (255, 255, 255), 2)
             else:
                 cv2.putText(image, "WAITING FOR BOTTLE...", (w//2-100, h//2), self.FONT, 0.7, (255, 255, 255), 2)
             return image

        # We have a bottle.
        
        # Calculate Target (if book exists)
        target_center = None
        is_success = False
        
        if book_info:
            bx, by, bw, bh = book_info['px_box']
            # Book is already drawn above in step 1

            book_bottom_y = by + bh // 2
            book_left_x = bx - bw // 2
            
            # Offset: 5cm to the LEFT
            # We estimate 5cm based on book width assuming book is ~20cm? 
            # Let's say 5cm is 1/4 of book width for now.
            offset_px = int(bw * 0.4) 
            
            target_x = book_left_x - offset_px
            target_y = book_bottom_y  # Same ground plane
            target_center = (target_x, target_y)
            
            # Check Success
            bottle_bottom = bottle_info['bottom']
            dist = np.linalg.norm(np.array(bottle_bottom) - np.array(target_center))
            
            # Threshold: Half bottle width or roughly 50px
            threshold = bottle_info['px_box'][2] * 0.8
            if dist < threshold:
                is_success = True

        # --- Draw Logic ---
        
        if is_success:
            # CLEAN SCREEN + BIG CHECKMARK
            # Draw big green checkmark in center
            center_x, center_y = w // 2, h // 2
            # Draw Green Circle
            cv2.circle(image, (center_x, center_y), 60, (0, 255, 0), -1)
            # Draw Checkmark (White)
            # Simple 3 point polyline
            pts = np.array([
                [center_x - 30, center_y],
                [center_x - 10, center_y + 30],
                [center_x + 40, center_y - 40]
            ], np.int32)
            cv2.polylines(image, [pts], False, (255, 255, 255), 10, cv2.LINE_AA)
            return image

        # Not Success: Guide Mode
        
        # 1. Draw Bottle (Yellow Style)
        # Yellow Fill
        overlay = image.copy()
        if self._draw_segmentation_fill_custom(overlay, bottle_info['det'], w, h, (0, 255, 255)): # Yellow BGR
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Yellow Border
        cx, cy, cw, ch = bottle_info['px_box']
        cv2.rectangle(image, (cx-cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), (0, 255, 255), 2)
        cv2.putText(image, "BOTTLE", (cx-cw//2, cy-ch//2-10), self.FONT, 0.6, (0, 255, 255), 2)

        # 2. Draw Target & Path (if book present)
        if target_center:
            tx, ty = target_center
            target_radius = int(bottle_info['px_box'][2] * 0.6) # Slightly larger than bottle radius
            
            # Perspective Circle (Green)
            cv2.ellipse(image, (tx, ty), (target_radius, target_radius // 3), 
                       0, 0, 360, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 3. Animated Path (Bottle -> Target)
            p1 = bottle_info['bottom']
            p2 = target_center
            
            self._draw_animated_dashed_line(image, p1, p2, (0, 255, 255)) # Yellow Path

        return image

    def _draw_segmentation_fill_custom(self, overlay, det, w, h, color):
        """Helper to draw fill with custom color"""
        polygon = det.get('polygon', [])
        if not polygon: return False
        pts = np.array(polygon, dtype=np.float32)
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = pts.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        return True

    def _draw_animated_dashed_line(self, img, p1, p2, color):
        """Draws a flowing dashed line from p1 to p2."""
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        if dist < 1: return
        
        # Normalized vector
        vec = (np.array(p2) - np.array(p1)) / dist
        
        segment_length = 20
        gap_length = 15
        total_period = segment_length + gap_length
        
        # Animation offset (speed = 50px/sec)
        speed = 100
        offset = int(time.time() * speed) % total_period
        
        # Draw segments
        current_dist = -offset # Start behind to flow in
        while current_dist < dist:
            start_d = max(0, current_dist)
            end_d = min(dist, current_dist + segment_length)
            
            if start_d < end_d:
                pt_start = np.array(p1) + vec * start_d
                pt_end = np.array(p1) + vec * end_d
                cv2.line(img, tuple(pt_start.astype(int)), tuple(pt_end.astype(int)), color, 3, cv2.LINE_AA)
            
            current_dist += total_period
            
        # Draw ArrowHead at p2
        angle = np.arctan2(vec[1], vec[0])
        arrow_len = 20
        arrow_width = np.pi / 6
        p_tip = np.array(p2)
        p_left = p_tip - arrow_len * np.array([np.cos(angle - arrow_width), np.sin(angle - arrow_width)])
        p_right = p_tip - arrow_len * np.array([np.cos(angle + arrow_width), np.sin(angle + arrow_width)])
        cv2.line(img, tuple(p_tip.astype(int)), tuple(p_left.astype(int)), color, 3, cv2.LINE_AA)
        cv2.line(img, tuple(p_tip.astype(int)), tuple(p_right.astype(int)), color, 3, cv2.LINE_AA)

    def render_task_pour(self, image, detections):
        # ... (Legacy logic, now replaced by arrange but kept signature if needed or just remove) ...
        # Since I changed the call site, I can remove or ignore this.
        return image # Placeholder

    def draw_cup_bottle_path(self, image, detections):
        """
        Draws an arched path (Bezier curve) between Cup (ID 41) and Bottle (ID 39).
        Visualizes a natural 'pick and place' or 'transfer' trajectory.
        """
        if not detections:
            return image

        cup_info = None
        bottle_info = None
        h, w, _ = image.shape

        for det in detections:
            label = det.get('label', '').lower()
            box = det.get('box', [])
            if len(box) != 4:
                continue
            
            # box is [cx, cy, w, h] normalized
            cx_px = int(box[0] * w)
            cy_px = int(box[1] * h)
            w_px = int(box[2] * w)
            h_px = int(box[3] * h)
            
            # Define points: Bottom Center (Ground), Top Center (Action)
            bottom_center = (cx_px, cy_px + h_px // 2)
            top_center = (cx_px, cy_px - h_px // 2 - 20) # 20px offset above object
            
            info = {
                'box_w': w_px,
                'bottom': bottom_center,
                'top': top_center
            }

            if label == 'cup':
                cup_info = info
            elif label == 'bottle':
                bottle_info = info

        if cup_info and bottle_info:
            # 1. Perspective Circles (3D Rings under objects) - Draw ONLY Front Arc (0-180) to simulate occlusion
            # Cup: Green Ring
            cv2.ellipse(image, cup_info['bottom'], (cup_info['box_w']//2, cup_info['box_w']//4), 
                       0, 0, 180, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Bottle: Red Ring
            cv2.ellipse(image, bottle_info['bottom'], (bottle_info['box_w']//2, bottle_info['box_w']//4), 
                       0, 0, 180, (0, 0, 255), 2, cv2.LINE_AA)

            # 2. Path Calculation (Above Objects)
            # P0 = Cup Top, P2 = Bottle Top
            p0 = np.array(cup_info['top'])
            p2 = np.array(bottle_info['top'])
            
            dist = np.linalg.norm(p2 - p0)
            mid = (p0 + p2) / 2
            
            # Arch height proportional to distance
            arch_height = dist * 0.4
            p1 = mid + np.array([0, -arch_height])
            
            # Generate Cubic Bezier Points
            num_points = 60 # More points for smoother dashing
            curve_points = []
            for t in np.linspace(0, 1, num_points):
                pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                curve_points.append(pt.astype(np.int32))
                
            # pts = np.array(curve_points).reshape((-1, 1, 2)) # Not needed for dashed loop
            
            # --- Style: Blue Translucent Dashed Path ---
            overlay = image.copy()
            path_color = (235, 206, 135) # Sky Blue (BGR)
            path_thickness = 6
            
            # Draw dashed curve manually
            for i in range(len(curve_points) - 1):
                # 50% duty cycle: Draw 3 segments, skip 3
                if i % 6 < 3: 
                    cv2.line(overlay, tuple(curve_points[i]), tuple(curve_points[i+1]), path_color, path_thickness, cv2.LINE_AA)

            # Blend
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # --- Solid Arrowhead Style (V-Shape) ---
            if len(curve_points) >= 5:
                # Use a slightly longer baseline for the tangent to be stable
                end_vec = curve_points[-1] - curve_points[-5]
                angle = np.arctan2(end_vec[1], end_vec[0])
                
                # Geometry
                arrow_len = 50 
                arrow_width = np.pi / 4 
                
                p_tip = curve_points[-1]
                p_left = p_tip - arrow_len * np.array([np.cos(angle - arrow_width), np.sin(angle - arrow_width)])
                p_right = p_tip - arrow_len * np.array([np.cos(angle + arrow_width), np.sin(angle + arrow_width)])
                
                # Draw Wings (Solid White, Thicker)
                cv2.line(image, tuple(p_tip.astype(int)), tuple(p_left.astype(int)), (255, 255, 255), 4, cv2.LINE_AA)
                cv2.line(image, tuple(p_tip.astype(int)), tuple(p_right.astype(int)), (255, 255, 255), 4, cv2.LINE_AA)
                
        return image
                
        return image
                

