import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import pafy
from cv2 import cv2


#functions
def addLineToCSV(file, line, isAppend):
    if isAppend:
        with open(file, 'a', newline='') as doc:
            writer = csv.writer(doc)
            writer.writerow(line)
    else:
        with open(file, 'w', newline='') as doc:
            writer = csv.writer(doc)
            writer.writerow(line)


# arguments
myParser = argparse.ArgumentParser(
    description=
    "Python file that uses OpenCV and detects the objects with given arguments"
)
myParser.add_argument(
    "--video",
    type=str,
    help="Video path, optional. If not given then code uses webcam.")
myParser.add_argument("--youtube",
                      type=str,
                      help="Youtube video key. i.e _yDHcLUmXYk")
myParser.add_argument("--confidence",
                      type=float,
                      help="confidence, optional, default=0.5",
                      default=0.5)
myParser.add_argument("--threshold",
                      type=float,
                      help="threshold, optional, default=0.3",
                      default=0.3)
myParser.add_argument("--file",
                      type=str,
                      help="Output file that contains time and object counts")

requiredArgs = myParser.add_argument_group("Required arguments")
requiredArgs.add_argument("--weight",
                          type=str,
                          help="YoloV3 weight path",
                          required=True)
requiredArgs.add_argument("--cfg",
                          type=str,
                          help="YoloV3 config path",
                          default="darknet/cfg/yolov3.cfg",
                          required=True)
requiredArgs.add_argument("--labels", type=str, help="Object names path")

argsList = myParser.parse_args()

# input arguments (starts with i)
iLabels = argsList.labels
videopath = argsList.video
iWeight = argsList.weight
iConfig = argsList.cfg
iConfidence = argsList.confidence
iThreshold = argsList.threshold
iYoutubeVideoKey = argsList.youtube
iFile = argsList.file

# if you have more than one webcam, 
# and if this does not work, you may want to change to webcam index you want
webcamIndex = 0 

labellist = open(iLabels).read().replace('\\n', '\n').split()

objText = ''
textLine = ['second', 'frame']  # first line of the csv file

# prepare counts
countdict = {}
for i in labellist:
    countdict[i] = 0
    textLine.append(i)
addLineToCSV(iFile, textLine, False)  # first line of the csv file, column names

labels = list(countdict)

# yolov3
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# cv2
net = cv2.dnn.readNetFromDarknet(iConfig, iWeight)

# layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# YOUTUBE
if iYoutubeVideoKey:
    yturl = 'http://www.youtube.com/watch?v=' + iYoutubeVideoKey
    video = pafy.new(yturl)
    best = video.getbest(preftype="any")

# if video path argument full then video, else webcam
if videopath:
    vid = cv2.VideoCapture(videopath)
elif iYoutubeVideoKey:
    vid = cv2.VideoCapture(best.url)
else:
    vid = cv2.VideoCapture(webcamIndex)

vid.set(cv2.CAP_PROP_BUFFERSIZE, 4)

# FPS - TEXT COUNT WRITE
# used to calculate the frame rate so we can save txt each second
# we got ceil for fps because it needs to be divided without remainder
fps = math.ceil(vid.get(cv2.CAP_PROP_FPS))
framecount = 0
vidsecond = 0

framespeed = 30  # used to stop the frame

while True:
    _, frame = vid.read()
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame,
                                 1 / 255.0, (320, 320),
                                 swapRB=True,
                                 crop=False)
    net.setInput(blob)

    start = time.time()
    outs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classids = []

    # text file writing per frame
    framecount += 1
    if framecount > fps:
        framecount = 0
        vidsecond += 1
    textLine = [vidsecond, framecount]

    for out in outs:
        for detection in out:
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # Consider only the predictions that are above a certain confidence level
            if confidence > iConfidence:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, iConfidence, iThreshold)

    # reset counts for each frame and recalculate
    for key in countdict:
        countdict[key] = 0

    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])

            # Prepare object counts
            objname = labels[classids[i]]
            countdict[objname] += 1

            # box text
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2)

        # Prepare object count text
        objText = ''
        for key in countdict:
            if countdict[key] > 0:
                objText += '\n' + str(key) + ': ' + str(countdict[key])

    else:
        objText = ''

    # add all labels' (obj types') count numbers
    for key in countdict:
        textLine.append(countdict[key])

    # Put object counts vertically
    y0, dy = 20, 30
    for i, line in enumerate(objText.split('\n')):
        fname = line.split(':')[0].lower()
        if fname:
            fid = labels.index(fname)
            fcolor = [int(c) for c in colors[fid]]
        else:
            fcolor = [0, 255, 0]
        y = y0 + i * dy
        cv2.putText(frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, fcolor,
                    2)

    cv2.imshow('webcam', frame)

    pressedKey = cv2.waitKey(framespeed) & 0xFF
    if pressedKey == ord('q'):
        break
    elif pressedKey == 32:
        # added for if I want to stop frame, press space bar stops the frame, pressing again continues
        if framespeed == 0:
            framespeed = 30
        else:
            framespeed = 0

    # at the end, append the line to the csv file which includes time second, frame numbers, labels' count numbers
    addLineToCSV(iFile, textLine, True)

vid.release()
cv2.destroyAllWindows()