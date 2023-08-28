import cv2
import numpy as np
import os

# Video and save paths
video_path = r"C:\Users\pc_al\Desktop\New folder\K.mp4"
save_path = r"C:\Users\pc_al\Desktop\New folder"
class_id = 9

# Load video
video = cv2.VideoCapture(video_path)

# Get video dimensions
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global enter
    if event == cv2.EVENT_RBUTTONDOWN:
        enter = True

# Create window and set mouse callback
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

# Create tracker object
tracker = cv2.TrackerCSRT_create()

# Lists to store data
labels = []
frames = []
checker = []
enter = False

# Read first frame and define ROI
ret, frame = video.read()
bbox = cv2.selectROI(frame, True)
tracker.init(frame, bbox)
frames.append(frame.copy())
labels.append(list(bbox))
checker.append(frame.copy())

# Loop through video frames and track object
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    checker.append(frame.copy())
    diff = cv2.absdiff(checker[1], checker[0])
    
    mean = np.mean(diff)
    std = np.std(diff)
            
    if ((mean > 20) and (std > 20)) or enter:
        tracker = cv2.TrackerCSRT_create()
        bbox = cv2.selectROI(frame, True)
        frames.append(frame.copy())
        labels.append(list(bbox))
        tracker.init(frame, bbox)
        ret, frame = video.read()
        enter = False
        if not ret:
            break
    
    success, bbox = tracker.update(frame)
    if success:
        frames.append(frame.copy())
        labels.append(list(bbox))
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
    else:
        cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imshow('Frame', frame)
                        
    del checker[0]
                        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Normalize and save labels
try_labels = labels.copy()

for i in range(len(try_labels)):
    for j in range(len(try_labels[i])):
        try_labels[i][j] = max(0, min(200, try_labels[i][j]))

normalized_labels = []

for label in try_labels:
    (x, y, w, h) = label
    x_center = x + w / 2
    y_center = y + h / 2
    normalized_x_center = x_center / width
    normalized_y_center = y_center / height
    normalized_w = w / width
    normalized_h = h / height
    normalized_labels.append([class_id, normalized_x_center, normalized_y_center, normalized_w, normalized_h])

# Create directory structure if it doesn't exist
data_path = os.path.join(save_path, "data")

if not os.path.exists(data_path):
    os.makedirs(os.path.join(data_path, "train", "images"))
    os.makedirs(os.path.join(data_path, "train", "labels"))
    os.makedirs(os.path.join(data_path, "valid", "images"))
    os.makedirs(os.path.join(data_path, "valid", "labels"))
else:
    print("Directory already exists.")

print(len(normalized_labels))
print(len(frames))

val_images = frames[:30]
val_labels = normalized_labels[:30]
train_images = frames[30:]
train_labels = normalized_labels[30:]

for i in range(len(val_images)):
    cv2.imwrite(os.path.join(data_path, "valid", "images", f"{class_id}({i}).jpg"), val_images[i])
    with open(os.path.join(data_path, "valid", "labels", f"{class_id}({i}).txt"), "w") as file:
        file.write(' '.join(map(str, val_labels[i])))

for i in range(len(train_images)):
    cv2.imwrite(os.path.join(data_path, "train", "images", f"{class_id}({i+30}).jpg"), train_images[i])
    with open(os.path.join(data_path, "train", "labels", f"{class_id}({i+30}).txt"), "w") as file:
        file.write(' '.join(map(str, train_labels[i+30])))
