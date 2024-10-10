import cv2
import numpy as np  # Use standard abbreviation for numpy

# Initializing function for the calling of the trackbar
def hello(x):
    # Only for reference
    print("")

# Initialize the camera
cap = cv2.VideoCapture(0)
bars = cv2.namedWindow("bars")

# Create trackbars for controlling HSV ranges
cv2.createTrackbar("upper_hue", "bars", 110, 180, hello)
cv2.createTrackbar("upper_saturation", "bars", 255, 255, hello)
cv2.createTrackbar("upper_value", "bars", 255, 255, hello)
cv2.createTrackbar("lower_hue", "bars", 68, 180, hello)
cv2.createTrackbar("lower_saturation", "bars", 55, 255, hello)
cv2.createTrackbar("lower_value", "bars", 54, 255, hello)

# Capturing the initial frame for background creation
while True:
    cv2.waitKey(1000)
    ret, init_frame = cap.read()
    if ret:
        break

# Start capturing frames for actual processing
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to HSV color space
    inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Getting the HSV values from trackbars
    upper_hue = cv2.getTrackbarPos("upper_hue", "bars")
    upper_saturation = cv2.getTrackbarPos("upper_saturation", "bars")
    upper_value = cv2.getTrackbarPos("upper_value", "bars")
    lower_hue = cv2.getTrackbarPos("lower_hue", "bars")
    lower_saturation = cv2.getTrackbarPos("lower_saturation", "bars")
    lower_value = cv2.getTrackbarPos("lower_value", "bars")
    
    # Define the kernel for dilation (to remove impurities)
    kernel = np.ones((3, 3), np.uint8)
    
    # Define the upper and lower HSV ranges for the cloak
    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
    
    # Create a mask based on the HSV range
    mask = cv2.inRange(inspect, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 3)
    
    # Create the inverse mask
    mask_inv = 255 - mask
    
    # Dilate the mask to smooth the edges
    mask = cv2.dilate(mask, kernel, 5)
    
    # Split the current frame into blue, green, red channels
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]
    
    # Apply bitwise AND to remove the masked-out area from the current frame
    b = cv2.bitwise_and(mask_inv, b)
    g = cv2.bitwise_and(mask_inv, g)
    r = cv2.bitwise_and(mask_inv, r)
    
    # Merge the channels back together
    frame_inv = cv2.merge((b, g, r))
    
    # Now apply the mask on the initial background (init_frame)
    b = init_frame[:, :, 0]
    g = init_frame[:, :, 1]
    r = init_frame[:, :, 2]
    
    # Use the mask to get the "cloak" area from the initial frame
    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    
    # Merge the channels back together to form the cloak area
    blanket_area = cv2.merge((b, g, r))
    
    # Combine the current frame with the blanket area using bitwise OR
    final = cv2.bitwise_or(frame_inv, blanket_area)
    
    # Show the output windows
    cv2.imshow("Harry's Cloak", final)
    cv2.imshow("Original", frame)
    
    # Wait for 'q' to quit
    if cv2.waitKey(3) == ord('q'):
        break

# Release the camera and close all windows
cv2.destroyAllWindows()
cap.release()
