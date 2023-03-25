# Import packages
from tflite_runtime.interpreter import Interpreter
import os
import cv2
import numpy as np
import time
from datetime import datetime
from centroidtracker import CentroidTracker
from flask import Flask, render_template
import multiprocessing as mp
import sqlite3

# Connect to the database
conn = sqlite3.connect('data.db')

# Create a cursor object
cur = conn.cursor()

# # Create a table
# cur.execute('''CREATE TABLE IF NOT EXISTS data_table (
#                ID INTEGER PRIMARY KEY,
#                Left INTEGER,
#                Right INTEGER,
#                Time TEXT,
#                Person TEXT);''')

cur.execute('''CREATE TABLE IF NOT EXISTS data_table (
               ID INTEGER PRIMARY KEY,
               Date TEXT,
               Time TEXT,
               Person TEXT);''')
app = Flask(__name__, static_folder='/')

def get_db():

    conn2 = sqlite3.connect('data.db')
    conn2.row_factory = sqlite3.Row
    return conn2

@app.route("/")
def index():
    conn2 = get_db()
    cur2 = conn2.cursor()
    # cur2.execute('BEGIN')
    cur2.execute('SELECT * FROM data_table')

    # Read data from the table into a Pandas DataFrame
    rows = cur2.fetchall()
    conn2.close()
    return render_template('index.html', rows=rows)

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
objects ={}
old_objects={}
curr_ID = 0
captured_image = np.array([])
is_captured = False

# compare the co-ordinates for dictionaries of interest
def DictDiff(dict1, dict2):
   dict3 = {**dict1}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = np.subtract(dict2[key], dict1[key])
   return dict3

import re
def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

# MODEL_NAME = args.modeldir
MODEL_NAME = "all_models/"
# GRAPH_NAME = args.graph
GRAPH_NAME = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
#LABELMAP_NAME = args.labels
LABELMAP_NAME = "coco_labels.txt"
# min_conf_threshold = float(args.threshold)
min_conf_threshold = 0.4
imW, imH = 800, 480

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

labels = load_labels(PATH_TO_LABELS)

def main():

    global labels, ct, objects, old_objects, curr_ID, captured_image, is_captured, conn
    
    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Newly added co-ord stuff
    leftcount = 0
    rightcount = 0 
    obsFrames = 0
    cap = cv2.VideoCapture("testing/output_v3.mp4")
    # cap = cv2.VideoCapture(0)
    ret = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    ret = cap.set(3,imW)
    ret = cap.set(4,imH)

    captured_image = False
    while True:
   
        saved_image_path = 'person_' + str(curr_ID) + '.png'
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # On the next loop set the value of these objects as old for comparison
        old_objects.update(objects)

        # Grab frame from camera
        hasFrame, frame1 = cap.read()

        # Acquire frame and resize to input shape expected by model [1xHxWx3]
        frame = frame1.copy()
        # recolor the image (reopenCV uses BGR instead of RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        #rects variable
        rects =[]

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if object_name == 'person':
            
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    box = np.array([xmin,ymin,xmax,ymax])

                    rects.append(box.astype("int"))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    if is_captured:
                        # Save the image
                        roi = frame[ymin:ymin+ymax, xmin:xmin+xmax]

                        saved_image_path =  'person_' + str(curr_ID) + '.png'
                        captured_image = cv2.imwrite(saved_image_path, roi)

                        print( "captured_image:{} ".format( saved_image_path ) )
                        is_captured = False


                    # Draw label
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        #update the centroid for the objects
        objects = ct.update(rects)
        # calculate the difference between this and the previous frame
        x = DictDiff(objects,old_objects)
	    # loop over the tracked objects
        for (objectID, centroid) in objects.items():

		    # draw both the ID of the object and the centroid of the
		    # object on the output frame
            textID = "ID {}".format(objectID)
        
            curr_ID = objectID

            cv2.putText(frame, textID, (centroid[0] - 2, centroid[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,'Right: {0}'.format(rightcount),(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),2,cv2.LINE_AA)
        cv2.putText(frame,'Left: {0}'.format(leftcount),(30,120),cv2.FONT_HERSHEY_SIMPLEX,1,(255,120,60),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        #count number of frames for direction calculation
        obsFrames = obsFrames + 1

        #see what the difference in centroids is after every x frames to determine direction of movement
        #and tally up total number of objects that travelled left or right
        if obsFrames % 30 == 0:
            d = {}
            for k,v in x.items():
                if v[0] > 3: 
                    d[k] =  "Left"
                    leftcount = leftcount + 1
                    is_captured = True
                elif v[0]< -3:
                    d[k] =  "Right"
                    rightcount = rightcount + 1
                    is_captured = True
                else: 
                    d[k] = "Stationary"
                    is_captured = True
            if bool(d):
                print(d, time.ctime()) # prints the direction of travel (if any) and timestamp
                cur.execute("INSERT OR IGNORE INTO data_table (ID, Date, Time, Person) VALUES (?, ?, ?, ?)", (curr_ID, datetime.now().date(), datetime.now().strftime("%H:%M:%S"), saved_image_path))
                conn.commit()
        
        # Press 'q' to quit and give the total tally
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    conn.close()

def run_flask():
    app.run(host='0.0.0.0', port=8080, threaded=False, debug=False) # Run FLASK

if __name__ == '__main__':
    p1 = mp.Process(target=run_flask)
    p2 = mp.Process(target=main)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
