import cv2
import torch
import glob
import os
from PIL import Image  
import PIL
import time
t1 = time.time()
def coords(list):
    li2 = ['class','x1','y1','x2','y2','x3','y3','x4','y4']
    x3 = list[1] 
    list.append(x3)
    y3 = list[4]
    list.append(y3)
    x4 = list[3]
    list.append(x4)
    y4 = list[2]
    list.append(y4)
    #print(list)
    thresh = list[5]
    del list[5]
    list.append(thresh)
    #print(list)
    dictionary = dict(zip(li2, list))
    return dictionary, x3, y4, x4, y3 #coordinates are same in bbox, foe eg: x3=x1

def saveimg(image_roi):
    cv2.imwrite(f"img.jpg", image_roi)


def roi(name_img, name_file):
    path = "runs\\detect\\exp\\"
    for img in glob.glob(path + name_img):
        with open(path + "labels\\" + name_file, 'r') as file:
            data = file.read().replace("\n",'')
            #print(data)
            li = data.split(" ")
            #print(li)
            final_dictionary, xmin, ymin, xmax , ymax = coords(li)
            xmn = int(xmin)
            ymn = int(ymin)
            xmx = int(xmax)
            ymx = int(ymax)
            img = cv2.imread(img)
            cv2.imshow("Full Image",img)
            cv2.waitKey(0)
            print(f"cropped image details: {final_dictionary}\n")
            cropped_image = img[ymn:ymx , xmn:xmx]
            cv2.imshow("ROI", cropped_image)
            cv2.imwrite(f"imgs_cropped//{name_img}", cropped_image)
            cv2.waitKey(0)
        
path = "runs\\detect\\exp"
img_list = os.listdir(path)
img_list.remove('labels')
file_list = os.listdir(path + '\\labels')
#print(file_list)
for i in range(len(img_list)):
    roi(img_list[i], file_list[i])

t2 = time.time()
tt = t2 - t1
print(tt)
        
    #dictions = dict(zip(li2,li))
    #print(dictions)


"""coordinates = roi(li)
print(coordinates)

for img in glob.glob("runs\\detect\\exp3\\0000_20210203_130756_0496_0095_bmp.rf.b14327744576df21c1e8c6bbe5aca092.jpg"):
img = cv2.imread(img)
cropped_image = img[294:383 , 130:205]
cv2.imshow("image" ,cropped_image)
cv2.waitKey(0)"""

        


