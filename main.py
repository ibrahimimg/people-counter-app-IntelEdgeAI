"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
#import datetime
import socket
import json
import cv2

#import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

#time_for_file_name=datetime.datetime.now().replace(microsecond=0).isoformat()

def get_args():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image, video file or use CAM for webcam (id 0)")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    args = parser.parse_args()
    return args

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client("People Counter")
    client.connect(
        host=MQTT_HOST,
        port=MQTT_PORT,
        keepalive=MQTT_KEEPALIVE_INTERVAL
    )
    return client

def draw_bounding_boxes(frame, result, prob_th, width, height):
    #Draw bounding boxes onto the frame.
    for box in result[0][0]:
        confidence = box[2]
        if confidence > prob_th:
            xmin,xmax = map(lambda b : int(b*width), [box[3],box[5]])
            ymin,ymax = map(lambda b : int(b*height), [box[4],box[6]])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0) , 2)
    return frame

def infer_on_stream(args,client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold=args.prob_threshold
    # Set request id
    req_id=0

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    input=args.input
    is_one_image=False
    #check for live camera feed
    if input.lower()=='cam':
        input=0   
    image_formats=[".png",".jpg",".bmp",".jpeg"]
    for i in range(len(image_formats)):
        if input==args.input:
            if input.endswith(image_formats[i]):
                is_one_image=True
                break
        else:
            input=0
    # Get and open video capture
    capture = cv2.VideoCapture(input)
    capture.open(input)
    if not capture.isOpened():
        print("ERROR! Unable to open input source")
        exit(1)

    # Grab the shape of the input 
    width = int(capture.get(3))
    height = int(capture.get(4))
    
    # Set global variables for people counting
    current_count = 0
    time_start = 0
    duration = 0
    #previous_duration=0
    total_count = 0
    total_count4text = 0
    previous_count = 0
    omitted_results = 0
    ### TODO: Loop until stream is over ###
    while capture.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = capture.read()[:]
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        _width=net_input_shape[3]
        _height=net_input_shape[2]
        p_frame = cv2.resize(frame, (_width, _height))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(net_input_shape[0], net_input_shape[1],_height,_width)
        inference_start=time.time()
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(req_id,p_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait(req_id) == 0:
            inference_time=(time.time()-inference_start)*1000
            inference_time=round(inference_time,2)
            if is_one_image==False:
                ### write some info onto frame 
                # Uncomment the following codes to see the stats in video output as well
                '''
                people_in_message = "people in frame : "+str(current_count)
                cv2.putText(frame, people_in_message, (10, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 200, 0), 1, cv2.LINE_AA, False)
                total_count_message = "total people counted : "+str(total_count4text)
                cv2.putText(frame, total_count_message, (10, 35),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 200, 150), 1, cv2.LINE_AA, False)
                '''
                frame_message = "omitted results : "+str(omitted_results)
                cv2.putText(frame, frame_message, (10, 55),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA, False)
        
            inference_time_message = "inference time : "+str(inference_time)+" ms"
            cv2.putText(frame, inference_time_message, (10, 420),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA, False)
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(req_id)
            ### TODO: Extract any desired stats from the results ###
            out_frame = draw_bounding_boxes(frame, result, prob_threshold, width, height)
            
            ### TODO: Calculate current_count, total_count and duration
            ### TODO: send relevant information on current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            current_count=0
            for r in result[0][0]:
                confidence = r[2]
                if confidence > prob_threshold:
                    current_count+=1

            ## on state change e.g if new person enter
            if current_count > previous_count:
                #store current time for calculating duration
                time_start=time.time()
                total_count += current_count - previous_count
            
            ## on state change e.g: if person left
            if current_count < previous_count:
                #calcute the time a person spent
                duration = time.time()-time_start
                #convert duration, from float to integer
                duration = int(duration)
                ## to avoid counting person more than one
                ## person detected should be there for atleast 2sec
                if duration>=2:
                    total_count = total_count
                else:
                    # substract previous count from total_count
                    # and count it as omitted frame
                    total_count=total_count-previous_count
                    omitted_results=omitted_results+1
                
                # Publish messages to the MQTT server, topic:person, key:total
                client.publish(topic="person", payload=json.dumps({"total" : total_count}))
                total_count4text=total_count
                if duration>=2:
                    # Publish messages to the MQTT server, topic:duration, key:duration [when person left]
                    client.publish(topic="person/duration", payload=json.dumps({"duration" : duration}))
                
            # Publish message to the MQTT server, topic: person, key:count
            client.publish(topic="person", payload=json.dumps({"count" : current_count}))
            previous_count = current_count
            # Break if escape key pressed
            if key_pressed == 27:
                break
            # save current frame if s key pressed
            if key_pressed == ord('s'):
                cv2.imwrite('output_frame.png',frame)

        #cv2.imshow("People Counter By Ibrahim",frame)
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if is_one_image==True:
            ### write the number of people in the image
            people_in_message = "people in the image : "+str(current_count)
            cv2.putText(frame, people_in_message, (10, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 100), 1, cv2.LINE_AA, False)
            cv2.imwrite("output_image.jpg",frame)

    # Release the capture and destroy any OpenCV windows
    capture.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = get_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args,client)

if __name__ == '__main__':
    main()
