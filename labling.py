import cv2
import numpy as np
import os

# Load video and create a tracker object
video = cv2.VideoCapture(r"C:\Users\pc_al\Desktop\New folder\K.mp4")
save_path=r"C:\Users\pc_al\Desktop\New folder"
class_=9

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

global enter
enter=False
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        global enter
        enter=True

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

#if you wnat to change track algorithm
tracker = cv2.TrackerCSRT_create()

labels=[]
Frames=[]
# Read first frame and define region of interest (ROI)
ret, frame = video.read()


bbox = cv2.selectROI(frame, True)
tracker.init(frame, bbox)

Frames.append(frame.copy())
labels.append(list(bbox))

checker=[frame.copy()]

# Loop through video frames and track object
while True:
    print(enter)
    ret, frame = video.read()
    if not ret:
        break

    checker.append(frame.copy())
        
    #To see if there is a difference between the current frame and the previous one
    diff=cv2.absdiff(checker[1], checker[0])

    
    mean = np.mean(diff)
    std = np.std(diff)
            
    print("mean of The difference between the two frames :",mean)
    print("std of The difference between the two frames :",std)
    print('=======')
            
    #Determine the value that determines if there is a difference or not
    if ((mean >20) and (std > 20)) or (enter==True):
        #if you wnat to change track algorithm
        tracker = cv2.TrackerCSRT_create()
        bbox = cv2.selectROI(frame, True)
        
        Frames.append(frame.copy())
        labels.append(list(bbox))
        
        tracker.init(frame, bbox)
        
        ret, frame = video.read()
        
        enter=False
        if not ret:
            break
        
         
    # Update tracker and get new bounding box coordinates
    success, bbox = tracker.update(frame)
    if success:
        Frames.append(frame.copy())
        labels.append(list(bbox))
        # Draw bounding box on frame
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w,y+h), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
    else:
        cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # Display frame
        cv2.imshow('Frame', frame)
                        
                        
    del checker[0]
                        
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close window
video.release()
cv2.destroyAllWindows()

TRYlabel=labels.copy()

for i in range(len(TRYlabel)):
    for j in range(len(TRYlabel[i])):
        if TRYlabel[i][j] <0:
            TRYlabel[i][j]=0
            
        elif TRYlabel[i][j]>200:
            TRYlabel[i][j]=200
            
normalized_label=[]
for i in TRYlabel:
    (x,y,w,h)=i
    x_center = x + w/2
    y_center = y + h/2
    
    normalized_x_centerr=x_center/width
    normalized_y_centerr=y_center/height
    
    normalized_w = w / width
    normalized_h = h / height
    
    normalized_label.append([class_,normalized_x_centerr,normalized_y_centerr,normalized_w,normalized_h])


if not os.path.exists("data"):
    # Create new folder
    os.mkdir(fr"{save_path}\data")
    
    os.mkdir(fr"{save_path}\data\train")
    os.mkdir(fr"{save_path}\data\train\images")
    os.mkdir(fr"{save_path}\data\train\labels")
    
    os.mkdir(fr"{save_path}\data\valid")
    os.mkdir(fr"{save_path}\data\valid\images")
    os.mkdir(fr"{save_path}\data\valid\labels")
else:
    print("file is exist")
    
print(len(normalized_label))
print(len(Frames))

val_image=Frames[:30]
val_label=normalized_label[:30]

train_image=Frames[30:]
train_image=normalized_label[30:]


for i in range(len(val_image)):
    cv2.imwrite(fr"{save_path}\data\valid\images\{class_}({i}).jpg", Frames[i])
    
    file = open(fr"{save_path}\data\valid\labels\{class_}({i}).txt", "w")
    file.write(' '.join(map(str, normalized_label[i])))
    file.close()

for i in range(len(train_image)):
    cv2.imwrite(fr"{save_path}\data\train\images\{class_}({i+30}).jpg", Frames[i+30])
    
    file = open(fr"{save_path}\data\train\labels\{class_}({i+30}).txt", "w")
    file.write(' '.join(map(str, normalized_label[i+30])))
    file.close()

    
    
    
    





