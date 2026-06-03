import cv2
import numpy as np

### WARNING: The weights and biases are completly random ###

weights1 = np.array([
    [4.8,  4.8,  4.8,  4.8],   # Circle?
    [-3.9, -3.9, -3.9, -3.9]  # 1:1 width to height?
])
bias1 = np.array([0.9, 0.9, 0.9, 0.9])
weights2 = np.array([[2.0], [2.0], [2.0], [2.0]])
bias2 = np.array([-1.0])

def sigmoid(x):
    return 1/(1+np.exp(-x))
def ai_predict(circularity, aspect_ratio):
    inputs = np.array([circularity, aspect_ratio])
    
    # Layer 1
    hidden = sigmoid(np.dot(inputs, weights1) + bias1)
    
    # Layer 2
    output = sigmoid(np.dot(hidden, weights2) + bias2)
    return output[0]
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Definition of yellow
lower_yellow = np.array([15, 80, 70])
upper_yellow = np.array([35, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #Gets rid of holes and is more realistic
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    
    #Solves lighting problems
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    #Make yellow into mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #Finds all sides of the blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500000>area> 1000: # Needs to be bigger than 5x10 and no bigger than 5000x100
            
            # w:h ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # is it circly?
            ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            circularity = area / circle_area
            
            #returns a percent
            confidence = ai_predict(circularity, aspect_ratio)
            
            # Good enough?
            if confidence > 0.8 and circularity > 0.6:
                # Point at it
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 3)
                text = f"Ball: {confidence*100:.1f}%"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
    cv2.imshow("FTC High-Speed Vision", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()