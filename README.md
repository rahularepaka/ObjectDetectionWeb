# YOLOv3 Object Detection 📸
Object Detection using coco.names dataset , weights and configuration files of real time object detection algorithm YOLOv3

This is WebApp of my previous repo [here](https://github.com/rahularepaka/ObjectDetectionYOLO)

To change the weights and configurations file , you may do so by changing the file directory of the same.

# Requirements 🏫
```
- pip install opencv-python
- pip install numpy
- pip install streamlit
- pip install streamlit-webrtc
- pip install time
- pip install pandas
```
# COCO Dataset 🍫

| S.No | Sports         | Living   | Things      | Vehicles  | Safety        | Food     | Dining       | Electronics |
|------|----------------|----------|-------------|-----------|---------------|----------|--------------|-------------|
| 1    | frisbee        | bird     | bench       | bicycle   | traffic light | banana   | bottle       | tvmonitor   |
| 2    | skis           | cat      | backpack    | car       | fire hydrant  | apple    | wine glass   | laptop      |
| 3    | snowboard      | dog      | umbrella    | motorbike | stop sign     | sandwich | cup          | mouse       |
| 4    | sports ball    | horse    | handbag     | aeroplane | parking meter | orange   | fork         | remote      |
| 5    | kite           | sheep    | tie         | bus       | ~Nil~         | broccoli | knife        | keyboard    |
| 6    | baseball bat   | cow      | suitcase    | train     | ~Nil~         | carrot   | spoon        | cell phone  |
| 7    | baseball glove | elephant | chair       | truck     | ~Nil~         | hot dog  | bowl         | ~Nil~       |
| 8    | skateboard     | bear     | sofa        | boat      | ~Nil~         | pizza    | microwave    | ~Nil~       |
| 9    | surfboard      | zebra    | pottedplant | ~Nil~     | ~Nil~         | donut    | oven         | ~Nil~       |
| 10   | tennis racket  | giraffe  | bed         | ~Nil~     | ~Nil~         | cake     | toaster      | ~Nil~       |
| 11   | ~Nil~          | person   | diningtable | ~Nil~     | ~Nil~         | ~Nil~    | sink         | ~Nil~       |
| 12   | ~Nil~          | ~Nil~    | toilet      | ~Nil~     | ~Nil~         | ~Nil~    | refrigerator | ~Nil~       |
| 13   | ~Nil~          | ~Nil~    | book        | ~Nil~     | ~Nil~         | ~Nil~    | ~Nil~        | ~Nil~       |
| 14   | ~Nil~          | ~Nil~    | clock       | ~Nil~     | ~Nil~         | ~Nil~    | ~Nil~        | ~Nil~       |
| 15   | ~Nil~          | ~Nil~    | vase        | ~Nil~     | ~Nil~         | ~Nil~    | ~Nil~        | ~Nil~       |
| 16   | ~Nil~          | ~Nil~    | scissors    | ~Nil~     | ~Nil~         | ~Nil~    | ~Nil~        | ~Nil~       |
| 17   | ~Nil~          | ~Nil~    | teddy bear  | ~Nil~     | ~Nil~         | ~Nil~    | ~Nil~        | ~Nil~       |
| 18   | ~Nil~          | ~Nil~    | hair drier  | ~Nil~     | ~Nil~         | ~Nil~    | ~Nil~        | ~Nil~       |
| 19   | ~Nil~          | ~Nil~    | toothbrush  | ~Nil~     | ~Nil~         | ~Nil~    | ~Nil~        | ~Nil~       |

# Usage 👥
```
# Paramaters which can be tuned to your requirements
confThreshold = 0.5
nmsThreshold = 0.2

# for reading all the datasets from the coco.names file into the array
with open("coco.names", 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
    
# configration and weights file location
model_config_file = "yolo-config\\yolov3-tiny.cfg"
model_weight = "yolo-weights\\yolov3-tiny.weights"

```

# How to Run this Program 🏃‍♂️
```
streamlit run main.py
```

# Reference 🧾
You can read more about [YOLO](https://pjreddie.com/darknet/yolo/)
and more about [streamlit](https://streamlit.io/)


#About Me 😉
I am Rahul Arepaka, II year CompSci student at Ecole School of Engineering, Mahindra University
```
Feel free to edit with the source code and enjoy coding
```

# Contact 📞
You may reach me using 

- [Mail](mailto:rahul20ucse156@mahindrauniversity.edu.in) 📧
- [Linkedin](https://www.linkedin.com/in/rahul-arepaka/) 😇
