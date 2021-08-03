import cv2
import numpy as np

cap = cv2.VideoCapture(0)

width = 320
height = 320

confThreshold = 0.5
nmsThreshold = 0.2

# empty list
class_names = []

# Colour Randomiser
#colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# for reading all the datasets from the coco.names file into the array
with open("coco.names", 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# configration and weights file location
model_config_file = "yolo-config\\yolov3-tiny.cfg"
model_weight = "yolo-weights\\yolov3-tiny.weights"

# darknet files
net = cv2.dnn.readNetFromDarknet(model_config_file, model_weight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# function for finding objects


def find(outputs, img):

    # the following loop is for finding confidence level
    hT, wT, cT = frame.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # the following loop is for bounding boxes and text

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(frame, f'{class_names[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)


while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerName = net.getLayerNames()
    # print(layerName)

    outputnames = [layerName[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputnames)

    output = net.forward(outputnames)
    # print(type(output[0]).shape)
    # print(type(output[1]).shape)
    # print(type(output[2]).shape)

    find(output, frame)

    cv2.imshow("Webcam feed", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
