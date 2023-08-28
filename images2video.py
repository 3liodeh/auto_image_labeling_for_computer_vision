import cv2
import os

class Image2Video:
    def __init__(self,ImageFileDirection,videoPath_Name="output.mp4",CODEC='XVID',fps=10):
        
        codec = cv2.VideoWriter_fourcc(*CODEC)

        img_names = os.listdir(ImageFileDirection)
        
        frame_size = cv2.imread(os.path.join(ImageFileDirection, img_names[0])).shape[:2]
        
        video_writer = cv2.VideoWriter(videoPath_Name, codec, fps, frame_size)


        for i in range(len(img_names)):
            #print(f"({i}).{img_names[0].split('.')[-1]}")
            img_path =f"{ImageFileDirection}\ ({i}).{img_names[0].split('.')[-1]}"
            img = cv2.imread(img_path)
            #print(img_path)
            video_writer.write(img)
    
        video_writer.release()
        cv2.destroyAllWindows()
        
Image2Video(r"C:\Users\pc_al\Desktop\New folder (3)\asl_alphabet_train\asl_alphabet_train\A",)
