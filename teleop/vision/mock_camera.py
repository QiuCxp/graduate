import zmq
import time
import cv2
import os

def main():
    port = 55555
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    addr = f"tcp://*:{port}"
    print(f"[MockCamera] Starting Fake Camera Server on port {port}...")
    try:
        socket.bind(addr)
    except zmq.ZMQError as e:
        print(f"[MockCamera] Error binding port {port}: {e}")
        print("[MockCamera] Is another service using this port? Try running on a different machine or kill the process.")
        return

    # Load a sample image (bus.jpg)
    img_path = "bus.jpg"
    if not os.path.exists(img_path):
        print(f"[MockCamera] Error: {img_path} not found. Running wget to get it.")
        os.system("wget https://ultralytics.com/images/bus.jpg")
    
    img = cv2.imread(img_path)
    if img is None:
        print("[MockCamera] Failed to load image.")
        return

    print(f"[MockCamera] Streaming {img_path} ({img.shape}) at 10Hz...")

    try:
        while True:
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                continue
            
            # Send raw bytes
            socket.send(buffer.tobytes())
            time.sleep(0.1) # 10 FPS
    except KeyboardInterrupt:
        print("[MockCamera] Stopping...")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
