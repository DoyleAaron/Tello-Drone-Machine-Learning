# This logic was obtained from ChatGPT and modified to work with the Tello drone.
# The code captures video from the Tello drone, processes the frames, and uses a pre-trained Keras model to classify letters.
# The code also displays the classification results on the video stream.

import socket
import time
import cv2
import numpy as np
import tensorflow as tf
import av
import threading

# -------------------------------
# Start Tello video stream
# -------------------------------
def start_video_stream():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tello_address = ('192.168.10.1', 8889)

    sock.sendto(b'command', tello_address)
    time.sleep(1)
    sock.sendto(b'streamon', tello_address)
    time.sleep(1)

start_video_stream()

# -------------------------------
# Load Keras model and labels
# -------------------------------
model = tf.keras.models.load_model("LetterRecognition/converted_keras/keras_model.h5")

labels = []
with open("LetterRecognition/converted_keras/labels.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            index, label = parts
            index = int(index)
            while len(labels) <= index:
                labels.append(None)
            labels[index] = label

# -------------------------------
# Frame grabber thread
# -------------------------------
class FrameGrabber:
    def __init__(self, stream_url):
        self.container = av.open(stream_url, options={"fflags": "nobuffer"})
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        for packet in self.container.demux(self.stream):
            if not self.running:
                break
            for frame in packet.decode():
                img = np.array(frame.to_image())
                with self.lock:
                    self.frame = img

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.container.close()

# -------------------------------
# Centre crop utility
# -------------------------------
def centre_crop(img):
    h, w, _ = img.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return img[top:top+min_dim, left:left+min_dim]

# -------------------------------
# Main loop
# -------------------------------
grabber = FrameGrabber("udp://@0.0.0.0:11111")

print("[INFO] Running real-time classification. Press 'q' to quit.")

try:
    while True:
        frame = grabber.read()
        if frame is None:
            continue  # No frame yet

        cropped = centre_crop(frame)
        resized = cv2.resize(cropped, (224, 224))
        input_tensor = tf.convert_to_tensor([resized], dtype=tf.float32) / 255.0

        predictions = model(input_tensor)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_label = labels[predicted_class]

        # Display result
        if confidence > 0.6:
            label_text = f"{predicted_label} ({confidence:.2f})"
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        top3 = np.argsort(predictions[0])[-3:][::-1]
        print("Top predictions:")
        for i in range(3):
            idx = top3[i]
            print(f"  {labels[idx]} ({predictions[0][idx]:.2f})")

        cv2.imshow("Tello Letter Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Exiting...")

finally:
    grabber.stop()
    cv2.destroyAllWindows()
