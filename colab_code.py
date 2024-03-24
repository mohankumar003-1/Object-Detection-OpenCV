import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow
def predict_from_image(image_path, modelConfiguration, modelWeights, classesFile, whT, confThreshold, nmsThreshold):

  img = cv.imread(image_path)
  classNames = []
  with open(classesFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')
  net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
  net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
  blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
  net.setInput(blob)
  layersNames = net.getLayerNames()
  outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
  outputs = net.forward(outputNames)
  findObjects(outputs, img, classNames, confThreshold, nmsThreshold)

  return img

def findObjects(outputs, img, classNames, confThreshold, nmsThreshold):


  hT, wT, cT = img.shape
  bbox = []
  classIds = []
  confs = []
  for output in outputs:
      for det in output:
          scores = det[5:]
          classId = np.argmax(scores)
          confidence = scores[classId]
          if confidence > confThreshold:
              w, h = int(det[2]*wT) , int(det[3]*hT)
              x, y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
              bbox.append([x, y, w, h])
              classIds.append(classId)
              confs.append(float(confidence))

  indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

  for i in indices:
      box = bbox[i]
      x, y, w, h = box[0], box[1], box[2], box[3]

      cv.rectangle(img, (x, y), (x+w,y+h), (255, 255 , 0), 2)
      cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (120, 1, 255), 2)


modelConfiguration = r'/content/yolov3.cfg'
modelWeights = r'/content/drive/MyDrive/yolov3.weights'
classesFile = r"/content/coco.names"
whT = 320
confThreshold =0.5
nmsThreshold= 0.2

import os
folder_path = '/content/images/'
for file_name in os.listdir(folder_path):
        if(file_name != '.ipynb_checkpoints'):
          image_path = folder_path +file_name
        print(image_path) 
        predicted_image = predict_from_image(image_path, modelConfiguration, modelWeights, classesFile, whT, confThreshold, nmsThreshold)
        cv2_imshow(predicted_image)
        print()

cv.waitKey(0)