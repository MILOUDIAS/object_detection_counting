# Import packages
from tflite_runtime.interpreter import Interpreter
import os
import cv2
import numpy as np
# import time
from datetime import datetime
from centroidtracker import CentroidTracker
from flask import Flask, render_template, Response
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
    cur2.execute('SELECT * FROM data_table')

    rows = cur2.fetchall()
    conn2.close()

    return render_template('index2.html', rows=rows)

@app.route('/video_feed')
def video_feed():
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# compare the co-ordinates for dictionaries of interest
def DictDiff(dict1, dict2):
   dict3 = {**dict1}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = np.subtract(dict2[key], dict1[key])
   return dict3

MODEL_NAME = "all_models/"
GRAPH_NAME = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
LABELMAP_NAME = "coco_labels.txt"
min_conf_threshold = 0.5
imW, imH = 800, 480

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)


def main():

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("testing/recording_test_1.mp4")
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    objects ={}
    old_objects={}
    curr_ID = 0
    captured_image = np.array([])
    is_captured = False

    # Newly added co-ord stuff
    leftcount = 0
    rightcount = 0
    obsFrames = 0
    
    roi_pos_to_left = 0.2
    roi_pos_to_right = 0.8
    is_Left = False
    is_Right = False
    
    interpreter, labels =cm.load_model(MODEL_NAME,PATH_TO_CKPT,PATH_TO_LABELS)


    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    # Create a new window
    window_name = "Person Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


    while True:
   
        # On the next loop set the value of these objects as old for comparison
        old_objects.update(objects)

        # Grab frame from camera
        _, frame1 = cap.read()

        cv2_im = frame1

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        cm.set_input(interpreter, pil_im)

        # Perform the actual detection by running the model with the image as input
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        saved_image_path = 'person_' + str(curr_ID) + '.png'
        
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

                        cur.execute("INSERT OR IGNORE INTO data_table (ID, Date, Time, Person) VALUES (?, ?, ?, ?)", (curr_ID, datetime.now().date(), datetime.now().strftime("%H:%M:%S"), saved_image_path))
                        conn.commit()
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

                if is_Left and centroid[0] < roi_pos_to_left*imW:
                        leftcount += 1
                        is_Left = False

                if is_Right and centroid[0] > roi_pos_to_right*imW:
                        rightcount += 1
                        is_Right = False
		        # draw both the ID of the object and the centroid of the
		        # object on the output frame
                textID = "ID {}".format(objectID)

                curr_ID = objectID

                cv2.putText(frame1, textID, (centroid[0] - 2, centroid[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2_im = cv2.putText(frame1,'Right: {0}'.format(rightcount),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(250,80,80),2,cv2.LINE_AA)
        cv2_im = cv2.putText(frame1,'Left: {0}'.format(leftcount),(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(250,80,80),2,cv2.LINE_AA)
        cv2_im = cv2.putText(frame1,'Detected: {0}'.format(len(objects)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(250,110,100),2,cv2.LINE_AA)
        cv2_im = cv2.putText(frame1, '{0} - {1}'.format(datetime.now().date(), datetime.now().strftime("%H:%M:%S")),(200,470),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        #count number of frames for direction calculation
        obsFrames = obsFrames + 1
        
        #see what the difference in centroids is after every x frames to determine direction of movement
        #and tally up total number of objects that travelled left or right
        if obsFrames % 24 == 0:  # obs = 18 and diff = 10 is good enough

            for k,v in x.items():
                if v[0] > 3: 
                    is_Left = True
                    is_captured = True
                elif v[0] < -3:
                    is_Right = True
                    is_captured = True        
        # Set the window's property to full screen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow(window_name, cv2_im)
        
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
