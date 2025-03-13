from djitellopy import Tello
import time

sleepy = 5

# Initialize Tello
tello = Tello()

# Connect to Tello
try:
    tello.connect()
    print(f"Connected to Tello. Battery: {tello.get_battery()}%")
except Exception as e:
    print(f"Failed to connect to Tello: {e}")
    exit(1)

# Ensure SDK mode is properly activated
try:
    tello.send_command_without_return("command")
    time.sleep(sleepy)  # Wait for SDK mode
except Exception as e:
    print(f"Failed to enter SDK mode: {e}")
    tello.end()
    exit(1)

# Take off and wait before moving
try:
    tello.takeoff()
    time.sleep(sleepy)
except Exception as e:
    print(f"Takeoff failed: {e}")
    tello.end()
    exit(1)

# Move forward by 50 cm
try:
    tello.move_forward(50)
    time.sleep(sleepy)

    # Rotate 180 degrees
    tello.rotate_clockwise(180)
    time.sleep(sleepy)

    # Move forward by 50 cm again
    tello.move_forward(50)
    time.sleep(sleepy)

except Exception as e:
    print(f"Movement failed: {e}")
    tello.emergency()

# Attempt to land with error handling
try:
    tello.land()
except Exception as e:
    print(f"Landing failed: {e}. Trying emergency stop.")
    tello.emergency()

# Disconnect
tello.end()
