from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import imutils
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
import requests
url = "https://nodered-odt.kku.ac.th/showup"

# url กล้อง IP camera
video = 'rtsp://prawee:1q2w3e4r@10.88.97.100:554/cam/realmonitor?channel=6&subtype=0'

# กล้องทดสอบ เป็น webcam บนเครื่อง 0 1 ตามแต่ว่ามีกี่ตัว
webcam = 0

# โหลดโมเดลที่ได้เทรนไว้
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'

# กำหนดค่าความแม่นยำที่ต้องการสำหรับคาดเดาใบหน้าและกำหนดขนาดของ frame ภาพ 
setprobability = 0.80
size = 480

with tf.Graph().as_default():

    # ขั้นกำหนดการทำงานของ GPU
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():

        # การตั้งค่าพารามิเตอร์
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # ขนาดใบหน้าที่เล็กที่สุดหน่วยพิกเซล
        threshold = [0.7,0.8,0.8]  # threshold ในการเลือก bounding box p-net r-net และ o-net
        factor = 0.709  # scale factor 
        margin = 44 # ระยะขอบ
        batch_size =100 #1000 ขนาดของ Batch หรือจำนวนข้อมูลที่จะให้ Optimiser คำนวนในหนึ่งครั้ง
        image_size = 182
        input_image_size = 160
        HumanNames = [] # กำหนดตัวแปร list เก็บชื่อบุคคล

        # โหลดโมเดลมา และการเตรียมข้อมูลที่ใช้เปรียบเทียบ
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        # อ่านข้อมูลจาก classifier.pkl เพื่อดึง model และ class ที่แบ่งไว้มาใช้
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile,encoding='latin1')

            # ดึงข้อมูลรายชื่อบุคคลที่เรียงลำดับไว้ใน classifier
            HumanNames = class_names[ :-2]

        # ประกาศใช้งานฟังก์ชันอ่านวีดีโอของ OpenCV
        video_capture = cv2.VideoCapture(webcam)
        print('Start Recognition')
        
        while True:

            # อ่านภาพจากวีดีโอมาใช้ทีละเฟรม
            ret, frame = video_capture.read()
            frame = imutils.resize(frame, height=size)
            timer =time.time()

            if frame.ndim == 2: # ndim จาก numpy ใช้ดูจำนวนมิติใน array
                frame = facenet.to_rgb(frame) # แปลงเป็น RGB

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

                        # เมื่อความเหมือนผ่านเงื่อนไข วาดกรอบรอบใบหน้าพร้อมเขียนชื่อกำกับ
                        if best_class_probabilities > setprobability:

                            # เขียนกรอบสี่เหลี่ยม
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) 

                            # loop ให้ค้นหาจาก list รายชื่อที่บันทึกไว้
                            for H_i in HumanNames:

                                # เลือกชื่อในรายการที่ตรงกับใบหน้าที่หาได้
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]

                                    # บันทึกข้อมูลการติดตาม ผ่าน http post method ส่งไปเป็น JSON
                                    datacam = {'facename':result_names,'locate':'recording_room'}
                                    postcam = requests.post(url, json = datacam)
                                    
                                    # เขียนข้อมูลลงไปที่ตำแหน่งเดียวกับกรอบใบหน้า
                                    percent = best_class_probabilities[0]*100
                                    print("Predictions : [ name: {} , accuracy: {:.3f} ] recording room".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                    cv2.putText(frame,"{} : {:.2f}%".format(result_names,percent), (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=1)
                                    
                        else :

                            # กรณีความเหมือนไม่ถึงเกณฆ์
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                            cv2.putText(frame, "Unknown", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=1)
                    except:   
                        print("error")

            # ใช้ดู framerate
            endtimer = time.time()
            fps = 1/(endtimer-timer)

            # เขียนข้อความลงบนภาพ
            cv2.putText(frame, "fps: {:.2f}".format(fps), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, "fps: {:.2f}".format(fps), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # แสดงผลในหน้าต่างใหม่
            cv2.imshow('Test Camera', frame)
            
            # การหยุด while loop
            key = cv2.waitKey(1)
            if key== 113: # กด "q" เพื่อปิด
                print("Closing Program")
                time.sleep(2)
                break

        # การจบการทำงานของ OpenCV ทั้งการเชื่อต่อกล้องและการปิดหน้าต่างทั้งหมด
        video_capture.release()
        cv2.destroyAllWindows()