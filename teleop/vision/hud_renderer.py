import cv2
import numpy as np
import time
import os
from PIL import Image, ImageDraw, ImageFont

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

        # Chinese font support (best-effort)
        self._font_cache = {}
        self._font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Light.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]

        # Label translation (optional)
        self.LABEL_MAP = {
            "bottle": "瓶子",
            "cup": "杯子",
            "book": "书本",
            "person": "人",
            "apple": "苹果",
            "orange": "橙子",
        }

        # Task 1 trajectory trail (past path)
        self._task1_trail = []
        self._task1_trail_max = 120
        self._task1_last_point = None
        self._task1_phase_start = None
        self._task1_has_target = False
        self._task1_confirmed = False

        # Task 2 analysis state
        self._task2_phase_start = None
        self._task2_has_target = False
        self._task2_confirmed = False

        # Task 3 trail
        self._task3_trail = []
        self._task3_trail_max = 120
        self._task3_last_point = None
        self._task3_person_offsets = None
        self._task3_person_fixed_contour = None
        self._task3_person_stable_count = 0
        self._task3_person_miss_count = 0
        self._task3_enter_time = None
        self._task3_scan_start = None
        self._task3_scan_done = False
        self._task3_profile_template = self._load_patient_profile_template()
        self._task3_profile_image = self._load_patient_profile_image()
        self._task2_enter_time = None
        self._task2_scan_done = False
        self._task2_trail = []
        self._task2_trail_max = 120
        self._task2_last_point = None

    def _get_font(self, size):
        if size in self._font_cache:
            return self._font_cache[size]
        for path in self._font_paths:
            try:
                font = ImageFont.truetype(path, size)
                self._font_cache[size] = font
                return font
            except Exception:
                continue
        self._font_cache[size] = None
        return None

    def _put_text(self, img, text, org, font_scale=0.5, color=(255, 255, 255), thickness=1):
        """Draw text with Chinese support using PIL on ROI (fallback to cv2 if font missing)."""
        if text is None:
            return
        font_size = max(12, int(22 * font_scale))
        font = self._get_font(font_size)
        if font is None:
            cv2.putText(img, text, org, self.FONT, font_scale, color, 1, cv2.LINE_AA)
            return

        # Calculate text bbox
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x, y = org
        # Convert baseline org to top-left for PIL
        tl_x = x
        tl_y = max(0, y - text_h)

        # Clamp ROI
        x1 = max(0, tl_x)
        y1 = max(0, tl_y)
        x2 = min(img.shape[1], tl_x + text_w + 2)
        y2 = min(img.shape[0], tl_y + text_h + 2)
        if x1 >= x2 or y1 >= y2:
            return

        roi = img[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Draw single-pass to keep thin/clear
        draw.text((0, 0), text, font=font, fill=(color[2], color[1], color[0]))

        roi_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        roi[:] = roi_bgr

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
            status_text = f"目标数：{obj_count}"
            self._put_text(img, status_text, (w//2 - 40, 30), 0.4, (255, 255, 255), 1)

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
            label_key = label.lower()
            label_text = self.LABEL_MAP.get(label_key, label.upper())
            (tw, th), _ = cv2.getTextSize(label_text, self.FONT, 0.4, 1)
            
            # Position above object
            text_x = int((cx * w) - tw / 2)
            text_y = y1 - 10
            
            # Optional: Tiny background for readability
            # cv2.rectangle(img, (text_x - 2, text_y - th - 2), (text_x + tw + 2, text_y + 2), (0,0,0), -1)
            
            self._put_text(img, label_text, (text_x, text_y), 0.4, (255, 255, 255), 1)
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
            1: "任务1：位置移动",
            2: "任务2：清理",
            3: "任务3：人员安全",
            4: "任务4：导航"
        }
        
        text = task_names.get(task_id, f"任务{task_id}")
        
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
        self._put_text(image, text, (cx - tw//2, cy + th//2), font_scale, color, thickness)
        
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
        # Task enter time (for initial VR buffer)
        if self._task2_enter_time is None:
            self._task2_enter_time = time.time()
            self._task2_scan_done = False
            self._task2_trail = []
            self._task2_last_point = None

        if not detections:
            self._task2_phase_start = None
            self._task2_has_target = False
            self._task2_confirmed = False
            self._task2_trail = []
            self._task2_last_point = None
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
            return image
            
        # Check Dimensions for Topple Detection (Crisis)
        # Bottle Info
        bx, by, bw, bh = bottle_info['box']
        
        # Condition: Aspect Ratio Check (Robust)
        # If Width is close to Height or larger (w > 0.6*h), it's likely horizontal or tilted towards camera.
        # Normal standing bottle is usually h > 2.5*w (w < 0.4*h)
        # We relax the threshold from 0.9 to 0.6 to catch "foreshortened" toppled bottles.
        is_toppled = bw > (bh * 0.6)

        # Delay crisis display for VR buffer
        if is_toppled and (time.time() - self._task2_enter_time) < 10.0:
            return image
        
        # --- STAGE 1: CRISIS MODE (Emergency) ---
        if is_toppled:
            # Effect: Red Alert (Stylized)

            # Reset analysis flow while in danger
            self._task2_phase_start = None
            self._task2_has_target = False
            self._task2_confirmed = False

            # 1) Pulsing red border
            pulse = 0.6 + 0.4 * (np.sin(time.time() * 2.5) * 0.5 + 0.5)
            border_color = (0, 0, int(180 + 75 * pulse))
            cv2.rectangle(image, (0, 0), (w-1, h-1), border_color, 8)

            # Dim all other areas to emphasize crisis
            dim_overlay = image.copy()
            cv2.rectangle(dim_overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(dim_overlay, 0.45, image, 0.55, 0, image)

            # 2) Bottle highlight (red contour + overlay)
            if bottle_info.get('det', {}).get('polygon'):
                overlay = image.copy()
                self._draw_segmentation_fill_custom(overlay, bottle_info['det'], w, h, (0, 0, 255))
                cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
                pts = np.array(bottle_info['det']['polygon'], dtype=np.float32)
                pts[:, 0] *= w
                pts[:, 1] *= h
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                overlay = image.copy()
                cv2.rectangle(overlay, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
                cv2.rectangle(image, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 0, 255), 3)

            # 3) Center-top warning text
            warn_text = "检测到倾倒风险"
            action_text = "请先扶正瓶子"
            (tw, th), _ = cv2.getTextSize(warn_text, self.FONT, 1.4, 3)
            text_x = w // 2 - tw // 2
            text_y = int(h * 0.12)
            self._put_text(image, warn_text, (text_x, text_y), 1.4, (0, 0, 255), 3)
            self._put_text(image, action_text, (text_x, text_y + 40), 0.9, (0, 0, 255), 2)

            # 4) Block other overlays
            return image

        # --- STAGE 2: ISOLATION MODE (Normal Operation) ---
        # Bottle is upright. Now we need to move it near Cup.
        
        if not cup_info:
            # Bottle is upright, but we need Cup to define the zone
            self._task2_phase_start = None
            self._task2_has_target = False
            self._task2_confirmed = False
            # Draw Bottle (red contour + overlay)
            if bottle_info.get('det', {}).get('polygon'):
                overlay = image.copy()
                self._draw_segmentation_fill_custom(overlay, bottle_info['det'], w, h, (0, 0, 255))
                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
                pts = np.array(bottle_info['det']['polygon'], dtype=np.float32)
                pts[:, 0] *= w
                pts[:, 1] *= h
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                overlay = image.copy()
                cv2.rectangle(overlay, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
                cv2.rectangle(image, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 0, 255), 2)
            return image

        # Cup Info (Ref)
        cx, cy, cw, ch = cup_info['box']
        
        # Generate Isolation Zone (Relative to Cup)
        # Definition: 20cm (approx 1 Cup width) to the Right (or Left?)
        # Let's say Right of Cup
        zone_offset = int(cw * 1.5)
        zone_x = cx + zone_offset
        zone_y = cy + ch // 2 # Ground level
        zone_w = int(bw * 1.5) # Based on Bottle width
        zone_h = int(bw * 1.0) # Flat elliptical zone
        
        target_center = (zone_x, zone_y)
        
        # --- STAGE 3: CLEARANCE & COMPLETION ---

        # Analysis flow after danger resolved (run only once)
        if not self._task2_scan_done:
            if not self._task2_confirmed and not self._task2_has_target:
                self._task2_phase_start = time.time()
            self._task2_has_target = True

            # Timings: scan 3s -> analysis 2s -> confirm
            delay_t = 0.0
            phase1_t = 3.0
            phase2_t = 2.0
            elapsed = 0.0 if self._task2_phase_start is None else (time.time() - self._task2_phase_start)

            phase_text = None
            if elapsed < delay_t:
                phase_text = None
            elif elapsed < delay_t + phase1_t:
                phase_text = "正在扫描桌面"
                sweep_color = (255, 120, 0)
                period = 4.0
                t = (time.time() % period) / period
                if t <= 0.5:
                    scan_y = int((t * 2.0) * h)
                else:
                    scan_y = int((1.0 - (t - 0.5) * 2.0) * h)
                cv2.line(image, (0, scan_y), (w, scan_y), sweep_color, 4, cv2.LINE_AA)
            elif elapsed < delay_t + phase1_t + phase2_t:
                phase_text = "正在分析可放置区域"
            else:
                phase_text = "推荐位置已确认"
                self._task2_confirmed = True
                self._task2_scan_done = True

            if phase_text:
                (tw, th), _ = cv2.getTextSize(phase_text, self.FONT, 1.8, 4)
                text_x = w // 2 - tw // 2
                text_y = int(h * 0.12)
                self._put_text(image, phase_text, (text_x, text_y), 1.8, (255, 120, 0), 4)

            if not self._task2_confirmed:
                return image
        else:
            self._task2_confirmed = True

        # Draw Bottle (red contour + overlay)
        if bottle_info.get('det', {}).get('polygon'):
            overlay = image.copy()
            self._draw_segmentation_fill_custom(overlay, bottle_info['det'], w, h, (0, 0, 255))
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
            pts = np.array(bottle_info['det']['polygon'], dtype=np.float32)
            pts[:, 0] *= w
            pts[:, 1] *= h
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            overlay = image.copy()
            cv2.rectangle(overlay, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
            cv2.rectangle(image, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), (0, 0, 255), 2)
        self._put_text(image, "目标：瓶子", (bx-bw//2, by-bh//2-10), 0.6, (0, 0, 255), 2)
        
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
                # Draw Book ERROR (contour + X)
                if book_info.get('det', {}).get('polygon'):
                    pts = np.array(book_info['det']['polygon'], dtype=np.float32)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    pts = pts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (b_left, bk_y-bk_h//2), (b_right, bk_y+bk_h//2), (0, 0, 255), 3)

                # Draw X over the book area
                cv2.line(image, (b_left, bk_y-bk_h//2), (b_right, bk_y+bk_h//2), (0, 0, 255), 3)
                cv2.line(image, (b_right, bk_y-bk_h//2), (b_left, bk_y+bk_h//2), (0, 0, 255), 3)
                self._put_text(image, "障碍：移开书本！", (b_left, bk_y-bk_h//2-10), 0.6, (0, 0, 255), 2)

        # Check 2: Is Bottle in Zone?
        bottle_bottom = bottle_info['bottom']
        dist_to_zone = np.linalg.norm(np.array(bottle_bottom) - np.array(target_center))
        in_zone_threshold = zone_w * 0.6
        is_in_zone = dist_to_zone < in_zone_threshold
        
        # --- RENDER ZONE & GUIDANCE ---
        
        if is_blocked:
            # Zone is BLOCKED (Red, high opacity)
            block_overlay = image.copy()
            cv2.ellipse(block_overlay, target_center, (zone_w//2, zone_h//2), 0, 0, 360, (0, 0, 255), -1)
            cv2.addWeighted(block_overlay, 0.55, image, 0.45, 0, image)
            self._put_text(image, "区域被阻挡", (zone_x-80, zone_y+30), 0.7, (0, 0, 255), 2)

            # Obstacle guidance line (move away)
            bk_cx, bk_cy = book_info['box'][0], book_info['box'][1]
            bk_cx = int(bk_cx)
            bk_cy = int(bk_cy)
            vec = np.array([bk_cx - zone_x, bk_cy - zone_y], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-3:
                vec = vec / norm
            guide_end = (int(bk_cx + vec[0] * 140), int(bk_cy + vec[1] * 140))
            cv2.arrowedLine(image, (bk_cx, bk_cy), guide_end, (0, 255, 255), 3, tipLength=0.25)
            self._put_text(image, "移开遮挡物", (guide_end[0]-20, guide_end[1]-10), 0.7, (0, 255, 255), 2)
        elif is_in_zone:
                        # SUCCESS!
                        # Green Zone + Checkmark
                        target_w = int(zone_w * 0.65)
                        target_h = int(zone_h * 0.55)
                        cv2.ellipse(image, target_center, (target_w//2, target_h//2), 0, 0, 360, (0, 255, 0), 4) # Thick Green
                        # Big Checkmark
                        cx, cy = w // 2, h // 2
                        cv2.circle(image, (cx, cy), 50, (0, 255, 0), -1)
                        pts = np.array([[cx - 20, cy], [cx - 5, cy + 20], [cx + 30, cy - 30]], np.int32)
                        cv2.polylines(image, [pts], False, (255, 255, 255), 8, cv2.LINE_AA)
        else:
             # Guidance Mode
             # Draw smaller translucent target zone
             target_w = int(zone_w * 0.65)
             target_h = int(zone_h * 0.55)
             target_overlay = image.copy()
             cv2.ellipse(target_overlay, target_center, (target_w//2, target_h//2), 0, 0, 360, (0, 255, 0), -1, cv2.LINE_AA)
             cv2.addWeighted(target_overlay, 0.25, image, 0.75, 0, image)

             # Path: Bottle center -> Target (white solid curve) + blue trail
             p1 = (bx, by)
             p2 = target_center

             if self._task2_last_point is None or np.linalg.norm(np.array(p1) - np.array(self._task2_last_point)) > 2:
                 self._task2_trail.append(p1)
                 self._task2_last_point = p1
                 if len(self._task2_trail) > self._task2_trail_max:
                     self._task2_trail.pop(0)

             if len(self._task2_trail) >= 2:
                 trail_overlay = image.copy()
                 pts = np.array(self._task2_trail, dtype=np.int32).reshape((-1, 1, 2))
                 cv2.polylines(trail_overlay, [pts], False, (255, 200, 120), 3, cv2.LINE_AA)
                 cv2.addWeighted(trail_overlay, 0.35, image, 0.65, 0, image)

             self._draw_solid_curve(image, p1, p2, (255, 255, 255))

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
        Task 3: Patient Care Safety
        Scenario: Person is stationary; robot must avoid safety boundaries while moving.
        Visuals:
         - 40% expanded human contour (yellow)
         - 20% expanded human contour (red)
         - Dim background on yellow intrusion; red warning text on red intrusion
        """
        h, w = image.shape[:2]

        # Task 3 entry buffer (VR warm-up)
        if self._task3_enter_time is None:
            self._task3_enter_time = time.time()
            self._task3_scan_start = None
            self._task3_scan_done = False
        
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
            
            if label == 'cup':
                # Use cup center as anchor for patient silhouette
                if cup_info is None or det.get('conf') > cup_info.get('conf', 0):
                    info['conf'] = det.get('conf', 0)
                    cup_info = info
            elif label == 'bottle':
                bottle_info = info

        # --- SAFETY ZONES ---
        red_mask = None
        yellow_mask = None
        yellow_ring = None
        person_contour = None
        red_contour = None
        yellow_contour = None
        person_mask = None
        person_png_overlay = None
        person_png_bbox = None

        if cup_info:
            cx, cy, cw, ch = cup_info['box']

            pw = cw * 5
            ph = ch * 5


            live_pts = self._make_patient_profile_contour(cx, cy, pw, ph)
            person_contour = live_pts.astype(np.int32).reshape((-1, 1, 2))

                        # ---- NEW: no-freeze ROI cropping (allow PNG partially out of screen) ----
            # Compute bbox from contour (bbox can be outside screen)
            min_x = int(np.min(person_contour[:, 0, 0]))
            max_x = int(np.max(person_contour[:, 0, 0]))
            min_y = int(np.min(person_contour[:, 0, 1]))
            max_y = int(np.max(person_contour[:, 0, 1]))

            # Full bbox size (can be larger than screen)
            bbox_w = max(1, max_x - min_x)
            bbox_h = max(1, max_y - min_y)

            person_png_overlay = None
            person_png_bbox = None
            person_mask = None
            red_mask = None
            yellow_mask = None
            yellow_ring = None
            red_contour = None
            yellow_contour = None

            if self._task3_profile_image is not None and bbox_w > 1 and bbox_h > 1:
                # Resize the PNG to the full bbox size FIRST (even if partly out of screen)
                full_overlay = cv2.resize(
                    self._task3_profile_image,
                    (bbox_w, bbox_h),
                    interpolation=cv2.INTER_AREA
                )

                # Intersection ROI with screen (screen coords)
                sx1 = max(0, min_x)
                sy1 = max(0, min_y)
                sx2 = min(w, max_x)
                sy2 = min(h, max_y)

                if sx2 > sx1 and sy2 > sy1:
                    # Corresponding ROI in resized PNG (overlay coords)
                    ox1 = sx1 - min_x
                    oy1 = sy1 - min_y
                    ox2 = ox1 + (sx2 - sx1)
                    oy2 = oy1 + (sy2 - sy1)

                    # Crop the visible portion only
                    person_png_overlay = full_overlay[oy1:oy2, ox1:ox2].copy()
                    person_png_bbox = (sx1, sy1, sx2, sy2)

                    # Build mask from cropped overlay
                    if person_png_overlay.shape[2] == 4:
                        alpha = person_png_overlay[:, :, 3]
                        local_mask = (alpha > 10).astype(np.uint8) * 255
                    else:
                        gray = cv2.cvtColor(person_png_overlay, cv2.COLOR_BGR2GRAY)
                        local_mask = (gray > 10).astype(np.uint8) * 255

                    # Place local mask into screen-sized person_mask
                    person_mask = np.zeros((h, w), dtype=np.uint8)
                    person_mask[sy1:sy2, sx1:sx2] = local_mask

                    # Expand 20% (red) and 40% (yellow) from visible silhouette mask
                    # NOTE: kernel size is derived from visible ROI size (so it won't explode to full screen)
                    vis_w = max(1, sx2 - sx1)
                    vis_h = max(1, sy2 - sy1)
                    base = max(3, int(min(vis_w, vis_h) * 0.05))
                    if base % 2 == 0:
                        base += 1
                    k20 = base
                    k40 = max(k20 + 2, int(k20 * 2))
                    if k40 % 2 == 0:
                        k40 += 1

                    kernel20 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k20, k20))
                    kernel40 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k40, k40))

                    red_mask = cv2.dilate(person_mask, kernel20, iterations=1)
                    yellow_mask = cv2.dilate(person_mask, kernel40, iterations=1)
                    yellow_ring = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(red_mask))

                    # Contours from masks (only the visible part affects the contour, which is what you want)
                    person_contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    person_contour = person_contours[0] if person_contours else None
                    red_contour = red_contours[0] if red_contours else None
                    yellow_contour = yellow_contours[0] if yellow_contours else None
            # ---- END NEW ----


        # --- 10s buffer: no HUD rendering ---
        if (time.time() - self._task3_enter_time) < 10.0:
            return image

        # --- BOTTLE ---
        bottle_center = None
        bottle_box = None
        if bottle_info:
            bx, by, bw, bh = bottle_info['box']
            bottle_center = (bx, by)
            bottle_box = (bx, by, bw, bh)

        # --- TARGET ZONE (left of person bottom) ---
        target_center = None
        target_w = None
        target_h = None
        if cup_info and bottle_info:
            if person_contour is not None:
                left_x = int(np.min(person_contour[:, 0, 0]))
                bottom_y = int(np.max(person_contour[:, 0, 1]))
            else:
                cx, cy, cw, ch = cup_info['box']
                left_x = cx - cw // 2
                bottom_y = cy + ch // 2

            extra_left = int((cup_info['box'][2]) * 2.0)
            zone_x = min(w - 1, max(0, left_x - extra_left))
            zone_y = min(h - 1, max(0, bottom_y))
            target_center = (zone_x, zone_y)
            target_w = int(bottle_box[2] * 0.7)
            target_h = int(bottle_box[2] * 0.5)

        # --- PATH PLANNING (visual) ---
        path_points = []
        hit_red = False
        hit_yellow = False
        bottle_mask = None
        if bottle_center and target_center:
            p1 = np.array(bottle_center, dtype=np.float32)
            p2 = np.array(target_center, dtype=np.float32)
            mid = (p1 + p2) / 2.0
            vec = p2 - p1
            norm = np.linalg.norm(vec)
            if norm < 1:
                vec = np.array([1.0, 0.0], dtype=np.float32)
                norm = 1.0
            vec /= norm
            normal = np.array([-vec[1], vec[0]], dtype=np.float32)

            if cup_info:
                pc = np.array(cup_info['center'], dtype=np.float32)
                if np.dot(normal, pc - mid) > 0:
                    normal *= -1.0

            offset = max(80.0, norm * 0.35)
            ctrl = mid + normal * offset
            path_points = self._get_bezier_curve_points_custom(p1, p2, ctrl, num_points=60)

            # If path intersects yellow/red, push further away
            def _intersects(mask, points):
                if mask is None:
                    return False
                for pt in points:
                    x = int(max(0, min(w - 1, pt[0])))
                    y = int(max(0, min(h - 1, pt[1])))
                    if mask[y, x] > 0:
                        return True
                return False

            if _intersects(yellow_ring if yellow_ring is not None else yellow_mask, path_points):
                ctrl = mid + normal * (offset * 1.6)
                path_points = self._get_bezier_curve_points_custom(p1, p2, ctrl, num_points=60)

            # Build bottle mask for danger detection
            bottle_mask = np.zeros((h, w), dtype=np.uint8)
            if bottle_info:
                det = bottle_info.get('det', {})
                if det.get('polygon'):
                    bpts = np.array(det['polygon'], dtype=np.float32)
                    bpts[:, 0] *= w
                    bpts[:, 1] *= h
                    bpts = bpts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(bottle_mask, [bpts], 255)
                else:
                    bx, by, bw, bh = bottle_info['box']
                    cv2.rectangle(bottle_mask, (bx-bw//2, by-bh//2), (bx+bw//2, by+bh//2), 255, -1)

            if bottle_mask is not None and red_mask is not None:
                hit_red = cv2.countNonZero(cv2.bitwise_and(bottle_mask, red_mask)) > 0
            if bottle_mask is not None and (yellow_ring is not None or yellow_mask is not None):
                ymask = yellow_ring if yellow_ring is not None else yellow_mask
                hit_yellow = cv2.countNonZero(cv2.bitwise_and(bottle_mask, ymask)) > 0

        # --- SCAN PHASE (no HUD except scan) ---
        if not self._task3_scan_done:
            if self._task3_scan_start is None:
                self._task3_scan_start = time.time()
            scan_elapsed = time.time() - self._task3_scan_start
            if scan_elapsed < 3.0:
                phase_text = "老人扫描"
                (tw, th), _ = cv2.getTextSize(phase_text, self.FONT, 1.4, 3)
                text_x = w // 2 - tw // 2
                text_y = int(h * 0.12)
                self._put_text(image, phase_text, (text_x, text_y), 1.4, (255, 120, 0), 3)

                # Blue up/down scan line
                period = 2.0
                t = (time.time() % period) / period
                if t <= 0.5:
                    scan_y = int((t * 2.0) * h)
                else:
                    scan_y = int((1.0 - (t - 0.5) * 2.0) * h)
                cv2.line(image, (0, scan_y), (w, scan_y), (255, 120, 0), 4, cv2.LINE_AA)
                return image
            else:
                self._task3_scan_done = True

        # --- ALERT LOGIC ---
        if hit_yellow or hit_red:
            dim_overlay = image.copy()
            cv2.rectangle(dim_overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(dim_overlay, 0.6, image, 0.4, 0, image)

        # Draw person fills (red 0-20%, yellow 20-40%)
        if red_mask is not None or yellow_ring is not None:
            overlay = image.copy()
            if red_mask is not None:
                overlay[red_mask > 0] = (0, 0, 255)
            if yellow_ring is not None:
                overlay[yellow_ring > 0] = (0, 255, 255)
            cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

        # Overlay patient PNG image
        if person_png_overlay is not None and person_png_bbox is not None:
            min_x, min_y, max_x, max_y = person_png_bbox
            overlay_img = person_png_overlay
            if overlay_img.shape[2] == 4:
                alpha = overlay_img[:, :, 3] / 255.0
                for c in range(3):
                    image[min_y:max_y, min_x:max_x, c] = (
                        alpha * overlay_img[:, :, c] + (1 - alpha) * image[min_y:max_y, min_x:max_x, c]
                    )

        # Draw person contours
        if yellow_contour is not None:
            cv2.polylines(image, [yellow_contour], True, (0, 255, 255), 2, cv2.LINE_AA)
        if red_contour is not None:
            cv2.polylines(image, [red_contour], True, (0, 0, 255), 2, cv2.LINE_AA)
        if person_contour is not None:
            cv2.polylines(image, [person_contour], True, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw bottle (contour + translucent fill)
        if bottle_info:
            det = bottle_info.get('det', {})
            if det.get('polygon'):
                overlay = image.copy()
                self._draw_segmentation_fill_custom(overlay, det, w, h, (0, 255, 255))
                cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
                pts = np.array(det['polygon'], dtype=np.float32)
                pts[:, 0] *= w
                pts[:, 1] *= h
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

        # Red warning text
        if hit_red or hit_yellow:
            warn_text = "进入安全警戒区"
            (tw, th), _ = cv2.getTextSize(warn_text, self.FONT, 1.2, 3)
            text_x = w // 2 - tw // 2
            text_y = int(h * 0.12)
            self._put_text(image, warn_text, (text_x, text_y), 1.2, (0, 0, 255), 3)
            return image

        # Draw target zone + path if safe
        if target_center and target_w and target_h:
            target_overlay = image.copy()
            cv2.ellipse(target_overlay, target_center, (target_w//2, target_h//2), 0, 0, 360, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.addWeighted(target_overlay, 0.25, image, 0.75, 0, image)

        if bottle_center and target_center:
            # Completion check
            dist = np.linalg.norm(np.array(bottle_center) - np.array(target_center))
            in_zone = dist < (target_w * 0.6) if target_w else False

            # Trail update
            if self._task3_last_point is None or np.linalg.norm(np.array(bottle_center) - np.array(self._task3_last_point)) > 2:
                self._task3_trail.append(bottle_center)
                self._task3_last_point = bottle_center
                if len(self._task3_trail) > self._task3_trail_max:
                    self._task3_trail.pop(0)

            if len(self._task3_trail) >= 2:
                trail_overlay = image.copy()
                pts = np.array(self._task3_trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(trail_overlay, [pts], False, (255, 200, 120), 3, cv2.LINE_AA)
                cv2.addWeighted(trail_overlay, 0.35, image, 0.65, 0, image)

            if len(path_points) >= 2:
                for i in range(len(path_points) - 1):
                    cv2.line(image,
                             tuple(path_points[i].astype(int)),
                             tuple(path_points[i + 1].astype(int)),
                             (255, 255, 255), 2, cv2.LINE_AA)

            if in_zone:
                done_text = "完成"
                (tw, th), _ = cv2.getTextSize(done_text, self.FONT, 1.2, 3)
                text_x = w // 2 - tw // 2
                text_y = int(h * 0.18)
                self._put_text(image, done_text, (text_x, text_y), 1.2, (0, 255, 0), 3)

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
        debug_text = f"调试：检测到 {len(detections)} 个目标"
        if bottle_info: debug_text += " [瓶子]"
        if book_info: debug_text += " [书本]"
        self._put_text(image, debug_text, (10, 60), 0.7, (0, 0, 255), 2)
        
        # 2. If no bottle, show waiting status
        if not bottle_info:
             self._task1_trail = []
             self._task1_last_point = None
             self._task1_phase_start = None
             self._task1_has_target = False
             self._task1_confirmed = False
             if not book_info:
                 self._put_text(image, "等待瓶子和书本...", (w//2-100, h//2), 0.7, (255, 255, 255), 2)
             else:
                 self._put_text(image, "等待瓶子...", (w//2-100, h//2), 0.7, (255, 255, 255), 2)
             return image

        # We have a bottle.
        
        # Calculate Target (if book exists)
        target_center = None
        is_success = False
        
        if book_info:
            bx, by, bw, bh = book_info['px_box']
            # Book is already drawn above in step 1

            book_center_y = by
            book_left_x = bx - bw // 2
            
            # Offset: 5cm to the LEFT
            # We estimate 5cm based on book width assuming book is ~20cm? 
            # Let's say 5cm is 1/4 of book width for now.
            offset_px = int(bw * 0.4) 
            
            target_x = book_left_x - offset_px
            target_y = book_center_y  # Center height
            target_center = (target_x, target_y)
            
            # Check Success
            bottle_bottom = bottle_info['bottom']
            dist = np.linalg.norm(np.array(bottle_bottom) - np.array(target_center))
            
            # Threshold: Half bottle width or roughly 50px
            threshold = bottle_info['px_box'][2] * 0.8
            if dist < threshold:
                is_success = True
        else:
            target_center = None

        # --- Fake visual analysis phase handling ---
        has_target = target_center is not None
        if not has_target:
            self._task1_phase_start = None
            self._task1_has_target = False
            self._task1_confirmed = False
        else:
            if not self._task1_confirmed and not self._task1_has_target:
                self._task1_phase_start = time.time()
            self._task1_has_target = True

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
            self._task1_trail = []
            self._task1_last_point = None
            self._task1_phase_start = None
            self._task1_has_target = False
            self._task1_confirmed = False
            return image

        # Not Success: Guide Mode

        # 2. Draw Target & Path (if book present)
        if target_center:
            tx, ty = target_center
            target_radius = int(bottle_info['px_box'][2] * 0.6) # Slightly larger than bottle radius

            # Fake analysis timeline (run once)
            if not self._task1_confirmed:
                elapsed = 0.0 if self._task1_phase_start is None else (time.time() - self._task1_phase_start)

                # Timing: 10s delay -> phase1 3s -> phase2 2s -> confirm
                delay_t = 10.0
                phase1_t = 3.0
                phase2_t = 2.0

                phase_text = None

                if elapsed < delay_t:
                    phase_text = None
                elif elapsed < delay_t + phase1_t:
                    phase_text = "正在扫描桌面"
                    # full-screen back-and-forth blue sweep line
                    sweep_color = (255, 120, 0)  # Blue (BGR)
                    period = 4.0
                    t = (time.time() % period) / period
                    if t <= 0.5:
                        scan_y = int((t * 2.0) * h)
                    else:
                        scan_y = int((1.0 - (t - 0.5) * 2.0) * h)
                    cv2.line(image, (0, scan_y), (w, scan_y), sweep_color, 4, cv2.LINE_AA)
                elif elapsed < delay_t + phase1_t + phase2_t:
                    phase_text = "正在分析可放置区域"
                else:
                    phase_text = "推荐位置已确认"
                    self._task1_confirmed = True

                # Upper-center text (larger, blue)
                if phase_text:
                    (tw, th), _ = cv2.getTextSize(phase_text, self.FONT, 1.8, 4)
                    text_x = w // 2 - tw // 2
                    text_y = int(h * 0.12)
                    self._put_text(image, phase_text, (text_x, text_y), 1.8, (255, 120, 0), 4)

            # Only show target/contours/trajectory after confirmation
            if self._task1_confirmed:
                # 1. Draw Bottle (Contour + Yellow Fill)
                overlay = image.copy()
                if self._draw_segmentation_fill_custom(overlay, bottle_info['det'], w, h, (0, 255, 255)):
                    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

                if bottle_info['polygon']:
                    pts = np.array(bottle_info['polygon'], dtype=np.float32)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    pts = pts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

                # Book outline
                if book_info and book_info['polygon']:
                    pts = np.array(book_info['polygon'], dtype=np.float32)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    pts = pts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

                # Grasp point
                bcx, bcy, bcw, bch = bottle_info['px_box']
                grasp_pt = (bcx, bcy)
                cv2.circle(image, grasp_pt, 5, (255, 255, 255), -1)
                cv2.circle(image, grasp_pt, 9, (0, 255, 255), 2)

                # Ellipse breathing
                breathe = 0.25 + 0.1 * (np.sin(time.time() * 2.0) * 0.5 + 0.5)
                target_overlay = image.copy()
                cv2.ellipse(target_overlay, (tx, ty), (target_radius, target_radius // 3),
                            0, 0, 360, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.addWeighted(target_overlay, breathe, image, 1 - breathe, 0, image)
            
            # (Ellipse rendering moved into analysis phases)
            
            # 3. Path (Bottle -> Target)
            if self._task1_confirmed:
                bcx, bcy, bcw, bch = bottle_info['px_box']
                p1 = (bcx, bcy)
                p2 = target_center

                # Update trail
                if self._task1_last_point is None or np.linalg.norm(np.array(p1) - np.array(self._task1_last_point)) > 2:
                    self._task1_trail.append(p1)
                    self._task1_last_point = p1
                    if len(self._task1_trail) > self._task1_trail_max:
                        self._task1_trail.pop(0)

                # Draw past trajectory (light blue, semi-transparent)
                if len(self._task1_trail) >= 2:
                    trail_overlay = image.copy()
                    pts = np.array(self._task1_trail, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(trail_overlay, [pts], False, (255, 200, 120), 3, cv2.LINE_AA)
                    cv2.addWeighted(trail_overlay, 0.35, image, 0.65, 0, image)

                # Draw planned trajectory (solid white curve)
                self._draw_solid_curve(image, p1, p2, (255, 255, 255))

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

    def _get_bezier_curve_points(self, p1, p2, num_points=60):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        dist = np.linalg.norm(p2 - p1)
        if dist < 1:
            return []

        mid = (p1 + p2) / 2.0
        arch_height = max(40.0, dist * 0.35)
        ctrl = mid + np.array([0.0, -arch_height], dtype=np.float32)

        curve_points = []
        for t in np.linspace(0, 1, num_points):
            pt = (1 - t) ** 2 * p1 + 2 * (1 - t) * t * ctrl + t ** 2 * p2
            curve_points.append(pt)
        return curve_points

    def _get_bezier_curve_points_custom(self, p1, p2, ctrl, num_points=60):
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        ctrl = np.array(ctrl, dtype=np.float32)
        dist = np.linalg.norm(p2 - p1)
        if dist < 1:
            return []

        curve_points = []
        for t in np.linspace(0, 1, num_points):
            pt = (1 - t) ** 2 * p1 + 2 * (1 - t) * t * ctrl + t ** 2 * p2
            curve_points.append(pt)
        return curve_points

    def _draw_solid_curve(self, img, p1, p2, color):
        """Draw a smooth solid curve between two points."""
        curve_points = self._get_bezier_curve_points(p1, p2, num_points=60)
        if len(curve_points) < 2:
            return

        for i in range(len(curve_points) - 1):
            cv2.line(img,
                     tuple(curve_points[i].astype(int)),
                     tuple(curve_points[i + 1].astype(int)),
                     color, 2, cv2.LINE_AA)

        # Arrow head at end (tangent)
        if len(curve_points) >= 5:
            end_vec = curve_points[-1] - curve_points[-5]
            angle = np.arctan2(end_vec[1], end_vec[0])
            arrow_len = 16
            arrow_width = np.pi / 6
            p_tip = curve_points[-1]
            p_left = p_tip - arrow_len * np.array([np.cos(angle - arrow_width), np.sin(angle - arrow_width)])
            p_right = p_tip - arrow_len * np.array([np.cos(angle + arrow_width), np.sin(angle + arrow_width)])
            cv2.line(img, tuple(p_tip.astype(int)), tuple(p_left.astype(int)), color, 2, cv2.LINE_AA)
            cv2.line(img, tuple(p_tip.astype(int)), tuple(p_right.astype(int)), color, 2, cv2.LINE_AA)

    def _make_patient_profile_contour(self, cx, cy, pw, ph):
        """Create a simple side-view upper-body silhouette contour.

        The contour is centered at (cx, cy) and scaled to the detected person size.
        """
        # Use image-based template if available
        if self._task3_profile_template is not None:
            profile = self._task3_profile_template
        else:
            # Normalize profile points (side-view upper body) in local coords
            # x,y in [-1,1] with origin at center
            profile = np.array([
                [0.35, -0.55],   # head top
                [0.55, -0.40],   # forehead
                [0.60, -0.20],   # nose
                [0.52, -0.05],   # mouth
                [0.50,  0.10],   # chin
                [0.40,  0.25],   # neck front
                [0.15,  0.30],   # chest front
                [-0.10, 0.30],   # chest mid
                [-0.35, 0.22],   # shoulder back
                [-0.55, 0.10],   # upper back
                [-0.60, -0.10],  # back mid
                [-0.50, -0.30],  # back upper
                [-0.20, -0.45],  # back to head
            ], dtype=np.float32)

        # Scale to detection size (upper-body roughly 75% of height)
        sx = pw * 0.55
        sy = ph * 0.75

        pts = profile.copy()
        pts[:, 0] = pts[:, 0] * sx + cx
        pts[:, 1] = pts[:, 1] * sy + cy
        return pts

    def _load_patient_profile_template(self):
        """Load patient silhouette template from assets and normalize to [-1,1]."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        asset_path = os.path.join(repo_root, "..", "assets", "flipped_transparent.png")
        asset_path = os.path.normpath(asset_path)

        if not os.path.exists(asset_path):
            return None

        img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.shape[2] >= 4:
            alpha = img[:, :, 3]
            mask = (alpha > 10).astype(np.uint8) * 255
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = (gray > 10).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        contour = contour.reshape((-1, 2)).astype(np.float32)

        # Normalize contour to center and scale into [-1,1]
        min_xy = contour.min(axis=0)
        max_xy = contour.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        size = np.maximum(max_xy - min_xy, 1.0)
        norm = (contour - center) / (size / 2.0)

        # Clamp to [-1, 1]
        norm = np.clip(norm, -1.0, 1.0)
        return norm

    def _load_patient_profile_image(self):
        """Load patient profile PNG with alpha channel for overlay."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        asset_path = os.path.join(repo_root, "..", "assets", "flipped_transparent.png")
        asset_path = os.path.normpath(asset_path)
        if not os.path.exists(asset_path):
            return None
        img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        return img

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
                

