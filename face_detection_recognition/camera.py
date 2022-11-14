''' main recognition program & camera classes '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle
import cv2
import imutils
import numpy as np
import facenet
import detect_face
from PIL import Image
import tensorflow.compat.v1 as tf
import requests

from base_camera import BaseCamera
from base_camera2 import BaseCamera2
from base_camera3 import BaseCamera3

# node-red url for timestamp data collecting
url = "https://nodered-odt.kku.ac.th/showup"

# loading mpdel for face recognition
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"

# variable for acuracy
setprobability = 0.95 # percent prediction of face recognition
size = 720 # size of video for resizing

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face default = 30
        threshold = [0.7,0.8,0.8]  # three steps's threshold
        factor = 0.709  # scale factor default = 0.709
        margin = 44
        batch_size =100 #1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile,encoding='latin1')

        class Camera(BaseCamera):
            ''' 1st camera '''
            camera_01 = 'rtsp://prawee:1q2w3e4r@10.88.97.100:554/cam/realmonitor?channel=6&subtype=0'
            # camera_01 = 0
            def __init__(self):
                if os.environ.get('OPENCV_CAMERA_SOURCE'):
                    Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
                super(Camera, self).__init__()

            @staticmethod
            def set_video_source(source):
                Camera.camera_01 = source

            @staticmethod
            def frames():
                camera = cv2.VideoCapture(Camera.camera_01)
                if not camera.isOpened():
                    raise RuntimeError('Could not start camera.')
                
                while True:
                    ret, frame = camera.read()
                    frame = imutils.resize(frame, height=size)
                    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame with spesific aspect (optional) 
                    timer =time.time()

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)

                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    faceNum = bounding_boxes.shape[0]
                    
                    if faceNum > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]
                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        for i in range(faceNum):
                            emb_array = np.zeros((1, embedding_size))
                            xmin = int(det[i][0])
                            ymin = int(det[i][1])
                            xmax = int(det[i][2])
                            ymax = int(det[i][3])

                            try:
                                # inner exception
                                if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                    print('Face is very close!')
                                    continue

                                cropped.append(frame[ymin:ymax, xmin:xmax,:])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                        interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                                # Draw box around each face
                                if best_class_probabilities>setprobability:
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  
                                    for H_i in HumanNames:
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names1 = HumanNames[best_class_indices[0]]
                                            datacam1 = {'facename':result_names1,'locate':'recording_room'}
                                            postcam1 = requests.post(url, json = datacam1)
                                            print("Predictions : [ name: {} , accuracy: {:.3f} ] recording room".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                            # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,0), -1)
                                            # cv2.putText(frame, result_names+': {:.2f}%'.format(best_class_probabilities[0]*100), (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            #            1, (0, 0, 0), thickness=1, lineType=1)
                                            cv2.putText(frame, result_names1, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)
                                            
                                else :
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                    # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 0, 255), -1)
                                    cv2.putText(frame, "Unknown", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (255, 255, 255), thickness=1, lineType=1)
                            except:   
                                print("error")
                            
                    endtimer = time.time()
                    fps = 1/(endtimer-timer)

                    # Text
                    cv2.putText(frame, "fps: {:.2f}".format(fps), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, "fps: {:.2f}".format(fps), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, "Recording room camera", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, "Recording room camera", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.imshow('Cam 1 Record Room', frame)
                    
                    yield cv2.imencode('.jpg', frame)[1].tobytes()
                
                    key = cv2.waitKey(1)
                    if key== 113: # "q"
                        print("Closing Program")
                        time.sleep(2)
                        break
                            
                camera.release()
                cv2.destroyAllWindows()

        class Camera2(BaseCamera2):
            ''' 2rd camera '''
            camera_02 = 'rtsp://prawee:1q2w3e4r@10.88.97.100:554/cam/realmonitor?channel=4&subtype=0'

            def __init__(self):
                if os.environ.get('OPENCV_CAMERA_SOURCE'):
                    Camera2.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
                super(Camera2, self).__init__()

            @staticmethod
            def set_video_source(source):
                Camera2.camera_02 = source

            @staticmethod
            def frames():
                camera = cv2.VideoCapture(Camera2.camera_02)
                if not camera.isOpened():
                    raise RuntimeError('Could not start camera.')
                
                while True:
                    ret, frame = camera.read()
                    frame = imutils.resize(frame, height=size)
                    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame with spesific aspect (optional) 
                    timer =time.time()

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)

                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    faceNum = bounding_boxes.shape[0]
                    
                    if faceNum > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]
                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        for i in range(faceNum):
                            emb_array = np.zeros((1, embedding_size))
                            xmin = int(det[i][0])
                            ymin = int(det[i][1])
                            xmax = int(det[i][2])
                            ymax = int(det[i][3])

                            try:
                                # inner exception
                                if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                    print('Face is very close!')
                                    continue

                                cropped.append(frame[ymin:ymax, xmin:xmax,:])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                        interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                                # Draw box around each face
                                if best_class_probabilities>setprobability:
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  
                                    for H_i in HumanNames:
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names1 = HumanNames[best_class_indices[0]]
                                            datacam1 = {'facename':result_names1,'locate':'Iot_room'}
                                            postcam1 = requests.post(url, json = datacam1)
                                            print("Predictions : [ name: {} , accuracy: {:.3f} ] IoT room".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                            # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,0), -1)
                                            # cv2.putText(frame, result_names+': {:.2f}%'.format(best_class_probabilities[0]*100), (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            #            1, (0, 0, 0), thickness=1, lineType=1)
                                            cv2.putText(frame, result_names1, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)
                                            
                                else :
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                    # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 0, 255), -1)
                                    cv2.putText(frame, "Unknown", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (255, 255, 255), thickness=1, lineType=1)
                            except:   
                                print("error")
                            
                    endtimer = time.time()
                    fps = 1/(endtimer-timer)

                    # Text
                    cv2.putText(frame, "fps: {:.2f}".format(fps), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, "fps: {:.2f}".format(fps), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, "IoT room camera", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, "IoT room camera", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.imshow('Cam 2 IoT room', frame)
                    
                    yield cv2.imencode('.jpg', frame)[1].tobytes()
        
                    key = cv2.waitKey(1)
                    if key== 113: # "q"
                        print("Closing Program")
                        time.sleep(2)
                        break
                            
                camera.release()
                cv2.destroyAllWindows()

        class Camera3(BaseCamera3):
            ''' 3rd camera '''
            camera_03 = 'rtsp://prawee:1q2w3e4r@10.88.97.100:554/cam/realmonitor?channel=8&subtype=0'

            def __init__(self):
                if os.environ.get('OPENCV_CAMERA_SOURCE'):
                    Camera3.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
                super(Camera3, self).__init__()

            @staticmethod
            def set_video_source(source):
                Camera3.camera_03 = source

            @staticmethod
            def frames():
                camera = cv2.VideoCapture(Camera3.camera_03)
                if not camera.isOpened():
                    raise RuntimeError('Could not start camera.')
                
                while True:
                    ret, frame = camera.read()
                    frame = imutils.resize(frame, height=size)
                    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame with spesific aspect (optional) 
                    timer =time.time()

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)

                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    faceNum = bounding_boxes.shape[0]
                    
                    if faceNum > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]
                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        for i in range(faceNum):
                            emb_array = np.zeros((1, embedding_size))
                            xmin = int(det[i][0])
                            ymin = int(det[i][1])
                            xmax = int(det[i][2])
                            ymax = int(det[i][3])

                            try:
                                # inner exception
                                if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                    print('Face is very close!')
                                    continue

                                cropped.append(frame[ymin:ymax, xmin:xmax,:])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                        interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                                # Draw box around each face
                                if best_class_probabilities>setprobability:
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  
                                    for H_i in HumanNames:
                                        if HumanNames[best_class_indices[0]] == H_i:
                                            result_names1 = HumanNames[best_class_indices[0]]
                                            datacam1 = {'facename':result_names1,'locate':'back_stair'}
                                            postcam1 = requests.post(url, json = datacam1)
                                            print("Predictions : [ name: {} , accuracy: {:.3f} ] Back Stair".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                            # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,0), -1)
                                            # cv2.putText(frame, result_names+': {:.2f}%'.format(best_class_probabilities[0]*100), (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            #            1, (0, 0, 0), thickness=1, lineType=1)
                                            cv2.putText(frame, result_names1, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)
                                            
                                else :
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                    # cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 0, 255), -1)
                                    cv2.putText(frame, "Unknown", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (255, 255, 255), thickness=1, lineType=1)
                            except:   
                                print("error")
                            
                    endtimer = time.time()
                    fps = 1/(endtimer-timer)

                    # Text
                    cv2.putText(frame, "fps: {:.2f}".format(fps), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, "fps: {:.2f}".format(fps), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, "Back Stair camera", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, "Back Stair camera", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.imshow('Cam 3 Back Stair', frame)
                    
                    yield cv2.imencode('.jpg', frame)[1].tobytes()
        
                    key = cv2.waitKey(1)
                    if key== 113: # "q"
                        print("Closing Program")
                        time.sleep(2)
                        break
                            
                camera.release()
                cv2.destroyAllWindows()

