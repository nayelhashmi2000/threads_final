from cdetect import V5
import cv2
import os
import cv2
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
t1 = time.time()

from flask import Flask,request,jsonify

app = Flask(__name__)

image_name = 0
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return "hello"