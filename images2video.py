import cv2
import numpy as np
import os



# Load video and create a tracker object
video = cv2.VideoCapture(r"C:\Users\pc_al\Desktop\New folder\sighn_V1.mp4")
save_path=r"C:\Users\pc_al\Desktop\New folder"
class_=0



width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)



#if you wnat to change track algorithm
tracker = cv2.TrackerMedianFlow_create()

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
    if (mean >15) and (std > 15):
        #if you wnat to change track algorithm
        tracker = cv2.TrackerMedianFlow_create()
        bbox = cv2.selectROI(frame, True)
        
        Frames.append(frame.copy())
        labels.append(list(bbox))
        
        tracker.init(frame, bbox)
        
        ret, frame = video.read()
        if not ret:
            break
        

                
                
                
                
    # Update tracker and get new bounding box coordinates
    success, bbox = tracker.update(frame)
    if success:
        Frames.append(frame.copy())
        labels.append(list(bbox))
        # Draw bounding box on frame
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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




for i in range(len(labels)):
    for j in range(len(labels[i])):
        if labels[i][j] <0:
            labels[i][j]=0
            
        elif labels[i][j]>200:
            labels[i][j]=200
            

            
        
        

normalized_label=[]
for i in labels:
    x,y,w,h=i
    
    xmin=x/width
    ymin=y/height
    xmax=(x+w)/width
    ymax=(y+h)/height
    
    normalized_label.append([class_,xmin,ymin,xmax,ymax])
    

print(normalized_label[0])





if not os.path.exists("data"):
    # Create new folder
    os.mkdir(f"{save_path}\data")
    os.mkdir(f"{save_path}\data\image")
    os.mkdir(f"{save_path}\data\label")
else:
    print("Folder already exists.")



for i in range(len(normalized_label)):
    cv2.imwrite(f"{save_path}\data\image\({i}).jpg", Frames[i])
    
    file = open(f"{save_path}\data\label\({i}).txt", "w")
    file.write(' '.join(map(str, normalized_label[i])))
    file.close()



        
      
