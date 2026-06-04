import cv2
import numpy as np

weights1 = np.array([
    [4.8,  4.8,  4.8,  4.8],   # Circle?
    [-3.9, -3.9, -3.9, -3.9]  # 1:1 width to height?
])
bias1 = np.array([0.9, 0.9, 0.9, 0.9])
weights2 = np.array([[2.0], [2.0], [2.0], [2.0]])
bias2 = np.array([-1.0])

class BallTracker:
    _id_count = 0
    
    def __init__(self, cx, cy, radius, confidence, x, y, max_lost_frames=5, required_consecutive_hits=5):
        self.id = BallTracker._id_count
        BallTracker._id_count += 1
        
        # Current data
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.confidence = confidence
        self.x = x
        self.y = y
        
        # Set up tracker
        self.consecutive_hits = 1
        self.lost_frame_counter = 0
        self.max_lost_frames = max_lost_frames
        self.required_consecutive_hits = required_consecutive_hits
        self.is_locked = False
        self.updated_this_frame = True

    def update_match(self, cx, cy, radius, confidence, x, y):
        self.lost_frame_counter = 0
        self.consecutive_hits += 1
        self.updated_this_frame = True
        
        # Update
        self.cx, self.cy, self.radius, self.confidence = cx, cy, radius, confidence
        self.x, self.y = x, y
        
        if self.consecutive_hits >= self.required_consecutive_hits:
            self.is_locked = True

    def update_miss(self):
        self.consecutive_hits = 0
        self.lost_frame_counter += 1
        self.updated_this_frame = False
        
        if self.lost_frame_counter >= self.max_lost_frames:
            self.is_locked = False
            self.confidence = 0.0

def sigmoid(x):
    return 1/(1+np.exp(-x))
def ai_predict(circularity, aspect_ratio):
    inputs = np.array([circularity, aspect_ratio])
    
    # Layer 1
    hidden = sigmoid(np.dot(inputs, weights1) + bias1)
    
    # Layer 2
    output = sigmoid(np.dot(hidden, weights2) + bias2)
    return output[0]
active_trackers = []

def runPipeline(image, ll_robot_inputs):
    global active_trackers
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Yellow Threshold
    lower_yellow = np.array([15, 80, 70])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ll_outputs = [0.0] * 17
    detected_blobs = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if 50000 > area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
            
            # It is probably there
            inferred_circle_area = np.pi * (radius ** 2)
            raw_circularity = area / inferred_circle_area
            if 0.40 <= raw_circularity <= 0.82 and 0.70 <= aspect_ratio <= 1.40:
                ai_input_circularity = 0.95
                is_occluded = True
            else:
                ai_input_circularity = raw_circularity
                is_occluded = False
            
            # Ball?
            confidence = ai_predict(ai_input_circularity, aspect_ratio)
            
            if confidence >= 0.88:
                display_conf = confidence if not is_occluded else confidence * 0.98
                detected_blobs.append((cx, cy, radius, display_conf, x, y))

    for tracker in active_trackers:
        tracker.updated_this_frame = False

    for blob in detected_blobs:
        bcx, bcy, bradius, bconf, bx, by = blob
        
        closest_tracker = None
        min_distance = 70.0 
        
        for tracker in active_trackers:
            if tracker.updated_this_frame:
                continue
                
            distance = np.sqrt((bcx - tracker.cx)**2 + (bcy - tracker.cy)**2)
            if distance < min_distance:
                min_distance = distance
                closest_tracker = tracker
                
        if not closest_tracker==None:
            closest_tracker.update_match(bcx, bcy, bradius, bconf, bx, by)
        else:
            new_tracker = BallTracker(bcx, bcy, bradius, bconf, bx, by)
            active_trackers.append(new_tracker)
    for tracker in active_trackers:
        if not tracker.updated_this_frame:
            tracker.update_miss()
    balls_locked = 0
    for tracker in active_trackers:
        if tracker.is_locked and balls_locked < 8:
            idx = 1 + (balls_locked * 2)
            
            ll_outputs[idx] = float(tracker.cx)
            ll_outputs[idx + 1] = float(tracker.cy)
            balls_locked += 1
            
            cv2.circle(image, (int(tracker.cx), int(tracker.cy)), int(tracker.radius), (0, 255, 0), 2)
            cv2.circle(image, (int(tracker.cx), int(tracker.cy)), 3, (0, 0, 255), -1)
            
            text = f"ID:{tracker.id} ({tracker.confidence*100:.0f}%)"
            cv2.putText(image, text, (tracker.x, tracker.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    active_trackers = [t for t in active_trackers if t.lost_frame_counter < t.max_lost_frames]
    
    return contours, image, ll_outputs