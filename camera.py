import cv2
from ultralytics import YOLO

model=YOLO("C:/Users/Home/Desktop/AXZ/Code/Class project/py/ML/Ml_advance/Cnn/best.pt")

cap=cv2.VideoCapture(0)

while True:
    itr,frame=cap.read()
    results=model.predict(source=frame,)
    final_frame=results[0].plot()
    cv2.imshow("narges",final_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break