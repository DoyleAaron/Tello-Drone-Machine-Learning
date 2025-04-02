import socket
import time
import cv2
import numpy as np
import tensorflow as tf
import av

# -------------------------------
# Step 1: Start the Tello video stream
# -------------------------------
def start_video_stream():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tello_address = ('192.168.10.1', 8889)

    sock.sendto(b'command', tello_address)   # Enter SDK mode
    time.sleep(1)
    sock.sendto(b'streamon', tello_address)  # Start video stream
    time.sleep(1)

start_video_stream()

# -------------------------------
# Step 2: Load your trained Keras model
# -------------------------------
model = tf.keras.models.load_model("LetterRecognition/model.keras")

# ⚠️ Replace this with the actual order of class names from training
# (e.g. saved from train_ds.class_names or manually matching folder order)
labels = list("AbcdeFGhIJKLmonoPQRStuvWxyZ")

# -------------------------------
# Step 3: Centre crop utility (preserves aspect ratio)
# -------------------------------
def centre_crop(img):
    h, w, _ = img.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return img[top:top+min_dim, left:left+min_dim]

# -------------------------------
# Step 4: Open Tello video stream
# -------------------------------
container = av.open('udp://@0.0.0.0:11111')

# -------------------------------
# Step 5: Frame-by-frame prediction
# -------------------------------
for frame in container.decode(video=0):
    img = np.array(frame.to_image())

    # Preprocess: crop, resize, normalise
    cropped = centre_crop(img)
    resized_img = cv2.resize(cropped, (64, 64))
    input_tensor = tf.convert_to_tensor([resized_img], dtype=tf.float32) / 255.0

    # Predict
    predictions = model(input_tensor)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    predicted_label = labels[predicted_class]

    # Draw only if confidence is reasonably high
    if confidence > 0.6:
        label_text = f"{predicted_label} ({confidence:.2f})"
        cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    # Optional: show top 3 predictions in terminal
    top3 = np.argsort(predictions[0])[-3:][::-1]
    print("Top predictions:")
    for i in range(3):
        idx = top3[i]
        print(f"  {labels[idx]} ({predictions[0][idx]:.2f})")

    # Display annotated frame
    cv2.imshow("Tello Letter Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Step 6: Cleanup
# -------------------------------
cv2.destroyAllWindows()
