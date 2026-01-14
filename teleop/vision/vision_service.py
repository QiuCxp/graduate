import sys
import time
import os
import argparse
import json

print("[VisionService] Initializing...", flush=True)

# Try importing ultralytics to verify the environment
try:
    print("[VisionService] Importing ultralytics...", flush=True)
    from ultralytics import YOLO
    print("[VisionService] Importing numpy/cv2/zmq...", flush=True)
    import numpy as np
    import cv2
    import zmq
except ImportError as e:
    print(f"[VisionService] Error: Failed to import dependencies. Make sure you are running in the 'xr_vision' environment.")
    print(f"[VisionService] Detail: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Independent Vision Service for XR Teleoperation")
    parser.add_argument("--model", type=str, default="yolov8l-seg.pt", help="Path to YOLO model (auto-downloaded if not found)")
    parser.add_argument("--ip", type=str, default="192.168.123.164", help="IP address of the image server")
    parser.add_argument("--port", type=int, default=55555, help="ZMQ port of the head camera")
    parser.add_argument("--pub_port", type=int, default=55556, help="ZMQ port to publish detections")
    parser.add_argument("--debug", action="store_true", help="Show debug window")
    args = parser.parse_args()

    print(f"[VisionService] Starting Visual Detection Service...", flush=True)
    print(f"[VisionService] Loading model: {args.model}", flush=True)

    try:
        # Load model - this will download yolov8n.pt on first run
        model = YOLO(args.model) 
        print(f"[VisionService] Model loaded successfully.", flush=True)
    except Exception as e:
        print(f"[VisionService] Critical Error loading model: {e}")
        sys.exit(1)
    
    # Initialize ZMQ Subscriber (Camera Input)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, '') 
    socket.setsockopt(zmq.CONFLATE, 1) # Keep only latest frame
    
    # Initialize ZMQ Publisher (Detection Output)
    pub_socket = context.socket(zmq.PUB)
    pub_addr = f"tcp://*:{args.pub_port}"
    pub_socket.bind(pub_addr)
    
    connect_str = f"tcp://{args.ip}:{args.port}"
    print(f"[VisionService] Connecting to Camera at {connect_str}...", flush=True)
    try:
        socket.connect(connect_str)
    except zmq.ZMQError as e:
        print(f"[VisionService] Connect error: {e}", flush=True)

    print(f"[VisionService] Publishing detections to {pub_addr}", flush=True)
    print("[VisionService] Vision loop started. Waiting for images...", flush=True)
    
    try:
        while True:
            try:
                # Receive JPEG image from ZMQ
                jpg_buffer = socket.recv()
                
                # Decode JPEG
                np_arr = np.frombuffer(jpg_buffer, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("[VisionService] Warning: Received empty frame or decode failed.")
                    continue

                # Run Inference
                # conf=0.25 is standard YOLO default. 0.5 was too high for household objects in varying light.
                results = model.predict(frame, verbose=False, conf=0.25, iou=0.45)
                
                # Process Results
                det_count = len(results[0].boxes)
                detection_list = []
                
                if det_count > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] I SEE {det_count} OBJECTS!")
                    
                    # Re-loop with index to get matching masks
                    for i, box in enumerate(results[0].boxes):
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        # Round box coordinates to 3 decimal places
                        xywhn = np.round(box.xywhn[0].cpu().numpy(), 3).tolist()

                        poly = []
                        if results[0].masks is not None:
                             # normalized segments [N, 2]
                             if len(results[0].masks.xyn) > i:
                                 # Downsample polygon: Take every 10th point
                                 # A 3090 GPU is fast, but JSON serialization/drawing on weak CPU is slow.
                                 # Reducing 500 points -> 50 points makes 0 visual difference but 10x faster.
                                 raw_poly = results[0].masks.xyn[i]
                                 if len(raw_poly) > 0:
                                     step = max(1, len(raw_poly) // 50) # Target approx 50 points max
                                     # Round to 3 decimal places to drastically reduce JSON size (text length)
                                     poly = np.round(raw_poly[::step], 3).tolist()

                        detection_list.append({
                            "label": cls_name,
                            "conf": conf,
                            "box": xywhn,
                            "polygon": poly
                        })

                        # Log interesting objects
                        if cls_name in ['cup', 'bottle', 'remote', 'mouse', 'keyboard', 'cell phone']:
                             print(f"   -> FOUND TARGET: {cls_name} ({conf:.2f})")
                
                # Publish detections (as JSON) - always publish even if empty
                pub_socket.send_string(json.dumps(detection_list))

                # Optional: Show debug window locally
                if args.debug:
                    annotated_frame = results[0].plot()
                    cv2.imshow("Vision Service Debug", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            except zmq.ZMQError as e:
                # print(f"[VisionService] ZMQ Error (waiting for cam): {e}")
                time.sleep(0.5)
            except Exception as e:
                print(f"[VisionService] Error: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("[VisionService] Stopping...")
    finally:
        socket.close()
        pub_socket.close()
        context.term()
        if args.debug:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
