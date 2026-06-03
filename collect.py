import cv2
import numpy as np

cap = cv2.VideoCapture(0)
lower_yellow = np.array([15, 80, 70])
upper_yellow = np.array([35, 255, 255])

with open("training_data.txt", "w") as f:
    print("Data Collector Started! Press 'b' for Ball, 'j' for Junk, 'q' to Quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            biggest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest_contour)
            
            if area > 500:
                x, y, w, h = cv2.boundingRect(biggest_contour)
                aspect_ratio = float(w) / h
                
                ((cx, cy), radius) = cv2.minEnclosingCircle(biggest_contour)
                circularity = area / (np.pi * (radius ** 2))
                
                # Draw crosshair on the object we are sampling
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                
                # Listen for keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('b'): # Record Ball
                    f.write(f"{circularity},{aspect_ratio},1\n")
                    print(f"Recorded BALL: Circ={circularity:.2f}, Aspect={aspect_ratio:.2f}")
                elif key == ord('j'): # Record Junk
                    f.write(f"{circularity},{aspect_ratio},0\n")
                    print(f"Recorded JUNK: Circ={circularity:.2f}, Aspect={aspect_ratio:.2f}")
                elif key == ord('q'):
                    break
                    
        cv2.imshow("Data Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()