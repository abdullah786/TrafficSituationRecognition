import numpy as np
import cv2
import pickle
import time
import PySimpleGUI as sg            # Uncomment 1 to run on that framework
import imutils
import pathlib
#############################################

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'No Traffic'
    elif classNo == 1: return 'Medium Traffic'
    elif classNo == 2: return 'Heavy Traffic'
    elif classNo == 3: return 'Accident'



starting_time = time.time()
frame_id = 0
# Import Trained Model for Our Project Traffic Prediciton
# pickle_in=open("model_trained_new.p","rb")  ## rb = READ BYTE
pickle_in=open("model.p","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)

# Load Yolo Pre Trainined Model for Object Detection in the frame
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

sg.theme('Dark Brown')

# define the window layout
layout = [[sg.Text('Select Video From File System', size=(25, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Submit(), sg.Cancel()],
    [sg.Text('Class',font=("Helvetica", 10), size=(10, 1)), sg.Text('CLASS', size=(15, 1), key='_CLASS_')],
    [sg.Text('Probability',font=("Helvetica", 10), size=(10, 1)), sg.Text('PROBABILITY', size=(15, 1), key='PROBABILITY')],
    [sg.Image(filename='',  key='_IMAGE_')],
    [sg.Image(filename='', key='_IMAGE2_')]
          ]
# Number of frames to capture
num_frames = 3;



# create the window and show it without the plot
window = sg.Window('Traffic Situation Machine Learning Project', layout, location=(400,400))
event, values = window.Read()

if event =='Submit':
    path = values[0]
    if not path:
        path = 0;
    if pathlib.Path(values[0]).suffix == '.mp4' or path == 0:
        # SETUP THE VIDEO CAMERA
        cap = cv2.VideoCapture(path)
        while True:
            event, values = window.Read(timeout=0, timeout_key='timeout')
            #event, values = window.Read()
            # READ IMAGE
            for i in range(0, num_frames):
                success, imgOrignal = cap.read()
            frame = imutils.resize(imgOrignal, width=500, height=500)

            frame_id += 1
            # Process Frame to Detect and Predict Objects
            imgOrignal = frame
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)
            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 0.75, color, 3)

            # PROCESS IMAGE
            img = np.asarray(imgOrignal)
            img = cv2.resize(img, (32, 32))
            img = preprocessing(img)
            # cv2.imshow("Processed Image", img)
            img = img.reshape(1, 32, 32, 1)
           # cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
           # cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            # PREDICT IMAGE
            predictions = model.predict(img)
            classIndex = model.predict_classes(img)
            probabilityValue =np.amax(predictions)
            #if probabilityValue > threshold:
                #print(getCalssName(classIndex))
                #cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                #cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time

            # cv2.imshow("Result", imgOrignal)
            imgbytes = cv2.imencode('.png', imgOrignal)[1].tobytes()  # Convert the image to PNG Bytes

            window.FindElement('_IMAGE_').Update(data=imgbytes, size=(1080, 400))  # Change the Image Element to show the new image
            window.FindElement('_CLASS_').update(str(getCalssName(classIndex)))
            window.FindElement('PROBABILITY').update(str(round(probabilityValue*100,2) )+"%")
            if event == 'Cancel':
                raise SystemExit()

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        else:
            sg.popup("Cancel", "No filename supplied")
            raise SystemExit()
else:
    sg.popup("Cancel", "No filename supplied")
    raise SystemExit("Cancelling: no filename supplied or Wrong file supplied")