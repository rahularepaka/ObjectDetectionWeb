import cv2
import streamlit as st
import numpy as np
import pandas as pd
import time

from streamlit_webrtc import (
    VideoTransformerBase
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


st.set_page_config(page_title="Object Detection", page_icon="🤖")


st.title("Lite Object Detection WebApp")
st.subheader(
    "Using COCO Dataset, YOLO-Tiny v3.0 Weight and Configuration files")

option = st.selectbox(
    'Please Select the Configuration file', ("yolov3-tiny.cfg",))
option = st.selectbox('Please Select the Weight file',
                      ("yolov3-tiny.weights",))


with st.spinner('Wait for the Weights and Configuration files to load'):
    time.sleep(3)
st.success('Done!')


threshold1 = 0.5
nmsThreshold = 0.2

class_names = []

# for reading all the datasets from the coco.names file into the array
with open("coco.names", 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# configration and weights file location
model_config_file = "/app/objectdetectionweb/yolo-config/yolov3-tiny.cfg"
model_weight = "/app/objectdetectionweb/yolo-weights/yolov3-tiny.weights"

# darknet files
net = cv2.dnn.readNetFromDarknet(model_config_file, model_weight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def find(outputs, img):

    # the following loop is for finding confidence level
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold1:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold1, nmsThreshold)

    # the following loop is for bounding boxes and text

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(img, f'{class_names[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)


class VideoTransformer(VideoTransformerBase):

    def __init__(self):
        self.threshold1 = 0.5

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        blob = cv2.dnn.blobFromImage(
            img, 1/255, (320, 320), [0, 0, 0], 1, crop=False)

        net.setInput(blob)

        layerName = net.getLayerNames()

        outputnames = [layerName[i[0]-1]
                       for i in net.getUnconnectedOutLayers()]
        # print(outputnames)

        output = net.forward(outputnames)

        find(output, img)

        return img


cap = webrtc_streamer(
    key="example", video_transformer_factory=VideoTransformer)

st.error('Please allow access to camera and microphone inorder for this to work')

st.warning('The object detection model might varies from machine to machine')


st.subheader("List of COCO dataset")
st.text("Total number of dataset are 80")
df = pd.read_excel("dataset.xlsx")

st.write(df)

st.subheader("How does it work ?")
st.text("Here is visualization of the algorithm")

st.image("/app/objectdetectionweb/Media/pic1.png", caption="YOLO Object Detection of 'Dog', 'Bicycle, 'Car'", width=None, use_column_width=None,
         clamp=False, channels='RGB', output_format='auto')

st.image("/app/objectdetectionweb/Media/pic2.png", caption="Algorithm", width=None, use_column_width=None,
         clamp=False, channels='RGB', output_format='auto')


st.subheader("About this App")

st.markdown("""
This app displays only data from COCO dataset downloaded from https://pjreddie.com/darknet/yolo/
and the configuration files and weights can be changed from the source code by downloading them from the above website.

You can see how this works in the [see the code](https://github.com/rahularepaka/ObjectDetectionWeb).

""")

with st.expander("Source Code"):

    code = '''
    
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)

    width = 320
    height = 320
    confThreshold = 0.5
    nmsThreshold = 0.2

    # empty list
    class_names = []

    # for reading all the datasets from the coco.names file into the array
    with open("coco.names", 'rt') as f:
        class_names = f.read().rstrip().split()
        
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
        outputnames = [layerName[i[0]-1] for i in net.getUnconnectedOutLayers()]
        output = net.forward(outputnames)
        find(output, frame)
        
        cv2.imshow("Webcam feed", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    '''

    st.code(code, language='python')


with st.expander("License"):

    st.markdown("""
                
MIT License

Copyright (c) 2021 Rahul Arepaka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
                """
                )

st.subheader("Author")
st.markdown(
    '''
    I am Rahul Arepaka, II year CompSci student at Ecole School of Engineering, Mahindra University
    '''
    '''
    Linkedin Profile : https://www.linkedin.com/in/rahul-arepaka/
    '''
    '''
    Github account : https://github.com/rahularepaka
    '''
)
st.info("Feel free to edit with the source code and enjoy coding")
