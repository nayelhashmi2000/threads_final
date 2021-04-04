from flask import Flask, request, jsonify
from cdetect import V5
import cv2
import os
import torch
import glob
import os
from PIL import Image
import json
import sqlite3
import base64
from datetime import date
import PIL
import time
import shutil
from ctypes import *
import os.path
from os import path
import numpy as np
t1 = time.time()


app = Flask(__name__)


def load_model():
    global PATH_SAVED_MODEL, yoloTiny, IMAGE_SIZE
    # PATH to saved and exported tensorflow model
    PATH_SAVED_MODEL = os.path.join(os.getcwd(), 'best.pt')
    IMAGE_SIZE = 640
    yoloTiny = V5(PATH_SAVED_MODEL)


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def insertBLOB(Id, photo):
    try:
        sqliteConnection = sqlite3.connect('threads.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
        #sqliteConnection.execute("""CREATE TABLE IF NOT EXISTS cones(id INTEGER PRIMARY KEY, roi BINARY UNIQUE)""")
        max_id = cursor.execute("""SELECT count(id) FROM cones""")
        sqlite_insert_blob_query = f""" INSERT INTO cones
                                  (roi) VALUES (?)"""
        """if(path.exists(photo) == False):
            data_tuple = ("Rejected",)
            cursor.execute(sqlite_insert_blob_query, data_tuple)
            sqliteConnection.commit()
            print("Rejected file inserted successfully as a BLOB into a table")
            cursor.close()"""

        roi = convertToBinaryData(photo)
        # Convert data into tuple format
        data_tuple = (roi,)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")

"""def notcropped(Id, photo):
    try:
        sqliteConnection = sqlite3.connect('threads.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")"""




def coords(list):
    li2 = ['class', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    x3 = list[1]
    list.append(x3)
    y3 = list[4]
    list.append(y3)
    x4 = list[3]
    list.append(x4)
    y4 = list[2]
    list.append(y4)
    # print(list)
    thresh = list[5]
    del list[5]
    list.append(thresh)
    # print(list)
    dictionary = dict(zip(li2, list))
    return dictionary, x3, y4, x4, y3  # coordinates are same in bbox, foe eg: x3=x1


def saveimg(image_roi):
    cv2.imwrite(f"img.jpg", image_roi)



def roi(name_img, name_file):
    path_file = "runs/detect/exp/images/"
    for img in glob.glob(path_file + name_img):
        # with open(path + "labels/" + name_file, 'r') as file:
        file_path = f"runs/detect/exp/labels/{total_images}" + '.txt' 
        bool_path = path.exists(file_path)
        print(bool_path)
        print(total_images)
        #print(name_img)
        if(bool_path == True):
            with open(f"{file_path}", 'r') as file:
                data = file.read().replace("\n",'')
                #print(data)
                li = data.split(" ")
                #print(li)
                final_dictionary, xmin, ymin, xmax ,ymax = coords(li)
                xmn = int(xmin)
                ymn = int(ymin)
                xmx = int(xmax)
                ymx = int(ymax)
                img = cv2.imread(img)
                # cv2.imshow("Full Image",img)
                #cv2.waitKey(0)
                print(f"cropped image details: {final_dictionary}\n")
                cropped_image = img[ymn:ymx , xmn:xmx]
                #print("hello")
                #cv2.imshow("ROI", cropped_image)
                print(cropped_image)
                cv2.imwrite(f"runs/detect/exp/cropped/{image_name}" + '.jpg', cropped_image)
                # status = "ACCEPTED"
                return cropped_image 
                #cv2.waitKey(0)
        else:
            # status = 'REJECTED'
            cropped_image = np.zeros((100,100,3), dtype=np.uint8)
            return cropped_image 

load_model()

""" create a database connection to a SQLite database """
conn = None
try:
    conn = sqlite3.connect(r"threads.db")
    print(sqlite3.version)
except Error as e:
    print(e)
finally:
    # create_connection(r"threads.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cones(id INTEGER PRIMARY KEY AUTOINCREMENT, roi BINARY UNIQUE)""")
    if conn:
        conn.close()

sqliteConnection = sqlite3.connect('threads.db')
cursor = sqliteConnection.cursor()
image_name = cursor.execute("""SELECT count(id) FROM cones""").fetchall()[0][0]
cursor.close()
os.environ['path'] += ';.\\camera'
libKsj = WinDLL("camera\KSJApi64.dll")
total_images = len(os.listdir("runs/detect/exp/images"))
#accepted_images = cursor.execute("""SELECT count(id) FROM cones""").fetchall()[0][0]
#rejected_images = int(image_name) - int(accepted_images)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        global image_name
        global total_images
        # global rejected_images
        # global accepted_images
        #img = cv2.imread('index.jpeg', 0)

        # libKsj.KSJ_Init()
        # nWidth = c_int()
        # nHeight = c_int()
        # nBitCount = c_int()

        # libKsj.KSJ_CaptureGetSizeEx(
        #     0, byref(nWidth), byref(nHeight), byref(nBitCount))
        # nbufferSize = nWidth.value * nHeight.value * nBitCount.value / 8
        # pRawData = create_string_buffer(int(nbufferSize))
        # retValue = libKsj.KSJ_CaptureRgbData(0, pRawData)
        # side1 = np.fromstring(pRawData, np.uint8).reshape(
        #     nHeight.value, nWidth.value
        libKsj.KSJ_Init()
        nWidth = c_int()
        nHeight = c_int()
        nBitCount = c_int()
        libKsj.KSJ_CaptureGetSizeEx(
            0, byref(nWidth), byref(nHeight), byref(nBitCount))
        nbufferSize = nWidth.value * nHeight.value * nBitCount.value / 8

        pRawData = create_string_buffer(int(nbufferSize))
        retValue = libKsj.KSJ_CaptureRgbData(0, pRawData)
        # img = np.frombuffer(pRawData, np.uint8).reshape(nHeight.value, nWidth.value, int(nBitCount.value/8))
        img = np.fromstring(pRawData, np.uint8).reshape(
            nHeight.value, nWidth.value, int(nBitCount.value/8))
        # , int(nBitCount.value/8))
        # side1 = cv2.flip(side1, 0)
        # side1 = cv2.resize(side1, (640, 640))

        print(img)
        # scale_percent = 30  # percent of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)
        #img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (640, 640))

        img = cv2.flip(img, 0)

        a = yoloTiny.detect(img, total_images)
        #a_list  = embeddings.tolist(b)
        #print(b.flatten().tolist())
        cv2.imwrite(f"runs/detect/exp/images/{total_images}.jpg", a)
        # full_image, cropped_image = roi(f"{image_name}.jpg", "exp.txt")
        # insertBLOB(image_name, f"runs/detect/exp/cropped/{image_name}.jpg")
        # print(image_name)
        # with open(f"runs/detect/exp/cropped/{image_name}.jpg", "rb") as img_file:
        #     my_string = base64.b64encode(img_file.read())
        # with open(f"runs/detect/exp/images/{image_name}.jpg", "rb") as image1:
        #     my_string1 = base64.b64encode(image1.read())
        # image_name += 1

        # return json.dumps({"image": my_string.decode('utf-8'), "image1": my_string1.decode('utf-8')})
        cropped_image = roi(f"{total_images}.jpg","exp.txt")
        
        if(cropped_image.any() == True):
            insertBLOB(image_name,f"runs/detect/exp/cropped/{image_name}.jpg")
            print(image_name)
            status = "ACCEPTED"
            print(status)
            with open(f"runs/detect/exp/cropped/{image_name}.jpg","rb") as img_file:
                my_string = base64.b64encode(img_file.read())
            with open(f"runs/detect/exp/images/{image_name}.jpg","rb") as img1_file:
                my_string1 = base64.b64encode(img1_file.read())
            image_name +=1
            #accepted_images +=1 
            summary = f'Accepted cones: {image_name}'
            return json.dumps({"image": my_string.decode('utf-8'), "image1": my_string1.decode('utf-8'), 'Status': status, 'Summary': summary})
            # return jsonify({'Status': status})
        else:
            status = "REJECTED"
            #insertBLOB(image_name, f"runs/detect/exp/cropped/{image_name}.jpg")
            with open(f"runs/detect/exp/images/{total_images}.jpg","rb") as img1_file:
                my_string1 = base64.b64encode(img1_file.read())
            #image_name +=1
            print(status)
            rejected_images = int(total_images) - int(image_name)
            #rejected_images +=1
            summary = f'rejected cones = {rejected_images}'
            return json.dumps({"image1": my_string1.decode('utf-8'), 'Status': status, 'Summary': summary}) 
            # return jsonify({'Status': status})

