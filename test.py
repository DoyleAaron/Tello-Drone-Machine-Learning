from djitellopy import Tello
import time

# Initialize Tello
tello = Tello()
tello.connect()

# Print battery percentage
print(f"Battery: {tello.get_battery()}%")

# Ensure SDK mode is properly activated
tello.send_command_without_return("command")
time.sleep(1)  # Wait for SDK mode

# Take off and wait before moving
tello.takeoff()
time.sleep(1)

# Move forward by 50 cm
try:
    tello.move_forward(50)
    time.sleep(1)

    # Rotate 180 degrees
    tello.rotate_clockwise(180)
    time.sleep(1)

    # Move forward by 50 cm again
    tello.move_forward(50)
    time.sleep(1)

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
