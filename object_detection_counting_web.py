# Import packages
from tflite_runtime.interpreter import Interpreter
import os
import cv2
import numpy as np
import time
from datetime import datetime
from centroidtracker import CentroidTracker
from flask import Flask, render_template, Response, request
import threading
import sqlite3
import utils as cm
from PIL import Image

# Connect to the database
conn = sqlite3.connect('data.db', check_same_thread=False)
# Create a cursor object
cur = conn.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS data_table (
               ID INTEGER PRIMARY KEY,
               Date TEXT,
               Time TEXT,
               Person TEXT);''')

app = Flask(__name__, static_folder='/')

def get_db():

    conn2 = sqlite3.connect('data.db', check_same_thread=False)
    conn2.row_factory = sqlite3.Row
    return conn2

@app.route("/")
def index():
    conn2 = get_db()
    cur2 = conn2.cursor()
    # cur2.execute('BEGIN')
    cur2.execute('SELECT * FROM data_table')

    rows = cur2.fetchall()
    conn2.close()

    return render_template('index2.html', rows=rows)

@app.route('/video_feed')
def video_feed():
    #global cap
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


cap = cv2.VideoCapture("testing/output_v3.mp4")
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
objects ={}
old_objects={}
curr_ID = 0
captured_image = np.array([])
is_captured = False
# frame_rate_calc = 0
# leftcount = 0
# rightcount = 0

def show_streaming(frame, labels, boxes, classes, scores, min_conf_threshold):
    global x, objects, old_objects, ct, is_captured, curr_ID, captured_image, frame_rate_calc, leftcount, rightcount
    rects = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold

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
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
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
                frame = cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                frame = cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

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

        frame = cv2.putText(frame, textID, (centroid[0] - 2, centroid[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame = cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Draw framerate in corner of frame
    frame = cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,'Right: {0}'.format(rightcount),(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),2,cv2.LINE_AA)
    frame = cv2.putText(frame,'Left: {0}'.format(leftcount),(30,120),cv2.FONT_HERSHEY_SIMPLEX,1,(255,120,60),2,cv2.LINE_AA)

    return frame




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

# labels = load_labels(PATH_TO_LABELS)

# Newly added co-ord stuff
leftcount = 0
rightcount = 0
obsFrames = 0
def main():

    global labels, ct, objects, old_objects, curr_ID, captured_image, is_captured, conn, leftcount, rightcount, obsFrames

    interpreter, labels =cm.load_model(MODEL_NAME,PATH_TO_CKPT,PATH_TO_LABELS)

    # Load the Tensorflow Lite model.
    # interpreter = Interpreter(model_path=PATH_TO_CKPT)

    # interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # cap = cv2.VideoCapture("testing/output_v3.mp4")
    # cap = cv2.VideoCapture(0)
    # ret = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # ret = cap.set(3,imW)
    # ret = cap.set(4,imH)
    
    # Create a new window
    window_name = "Person Detector"
    cv2.namedWindow(window_name)

    # Set the window's property to full screen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

        cv2_im = frame1

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)


        # recolor the image (reopenCV uses BGR instead of RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        cm.set_input(interpreter, pil_im)

        # Perform the actual detection by running the model with the image as input
        # interpreter.set_tensor(input_details[0]['index'],input_data)
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
                    cv2_im = cv2.rectangle(frame1, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    if is_captured:
                        # Save the image
                        roi = frame1[ymin:ymin+ymax, xmin:xmin+xmax]

                        saved_image_path =  'person_' + str(curr_ID) + '.png'
                        captured_image = cv2.imwrite(saved_image_path, roi)

                        print( "captured_image:{} ".format( saved_image_path ) )
                        is_captured = False


                    # Draw label
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2_im = cv2.rectangle(frame1, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2_im = cv2.putText(frame1, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

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

                cv2.putText(frame1, textID, (centroid[0] - 2, centroid[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Draw framerate in corner of frame
        # cv2_im = cv2.putText(frame1,'FPS: {0:.2f}'.format(frame_rate_calc),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2_im = cv2.putText(frame1,'Right: {0}'.format(rightcount),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(140,110,150),2,cv2.LINE_AA)
        cv2_im = cv2.putText(frame1,'Left: {0}'.format(leftcount),(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(140,110,150),2,cv2.LINE_AA)

        cv2_im = cv2.putText(frame1, '{0} - {1}'.format(datetime.now().date(), datetime.now().strftime("%H:%M:%S")),(200,470),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow(window_name, cv2_im)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        #count number of frames for direction calculation
        obsFrames = obsFrames + 1

        # cv2_im = show_streaming(cv2_im, labels, boxes, classes, scores, 0.6)

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

        ret, jpeg = cv2.imencode('.jpg', cv2_im)
        pic = jpeg.tobytes()

        #Flask streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + pic + b'\r\n\r\n')


    cv2.destroyAllWindows()
    cap.release()
    conn.close()

def run_server():

    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False) # Run FLASK

if __name__ == '__main__':

    server_thread = threading.Thread(target=run_server)
    app_thread = threading.Thread(target=main)
    
    server_thread.start()
    app_thread.start()

    # wait until thread 1 is finished
    server_thread.join()
    # wait until thread 2 is finished
    app_thread.join()
