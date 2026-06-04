import cv2
import numpy as np
import main

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Local Verification Playground active. Press 'q' to close window.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    contours, annotated_frame, telemetry_packet = main.runPipeline(frame, None)
    
    locked_count = int(telemetry_packet[0])
    if locked_count == 0:
        cv2.putText(annotated_frame, "SEARCHING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(annotated_frame, f"TRACKED TARGETS: {locked_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        
    cv2.imshow("FTC Local Playground Verification", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()