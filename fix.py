import numpy as np 
import glob
import math
import cv2 as cv
import os
import pandas as pd
import pickle
import urllib.request
import concurrent.futures
import socket

model = pickle.load(open('model_TA.pkl', 'rb'))
port = 5000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '192.168.243.221'
s.connect((host, port))
def derajat0 (img):
    max = np.max(img)
    imgtmp = np.zeros([max+1, max+1])
    for i in range(len(img)):
        for j in range(len(img[i])-1):
            imgtmp[img[i,j],img[i,j+1]]+=1
    transpos = np.transpose(imgtmp)
    data = imgtmp+transpos
    tmp = 0
    for i in range(len(data)):
        for j in range(len(data)):
            tmp+=data[i,j]
    for i in range(len(data)):
        for j in range(len(data)):
            data[i,j]/=tmp
    return data

def derajat45 (img):
    max = np.max(img)
    imgtmp = np.zeros([max+1, max+1])
    for i in range(len(img)-1):
        for j in range(len(img[i])-1):
            imgtmp[img[i+1,j],img[i,j+1]] += 1
  
    transpos = np.transpose(imgtmp)
    data = imgtmp+transpos

    tmp = 0
    for i in range(len(data)):
        for j in range(len(data)):
            tmp+=data[i,j] 
    for i in range(len(data)):
        for j in range(len(data)):
            data[i,j]/=tmp
    return data

def derajat90 (img):
    max = np.max(img)
    imgtmp = np.zeros([max+1, max+1])
    for i in range(len(img)-1):
        for j in range(len(img[i])):
            imgtmp[img[i+1,j],img[i,j]] += 1
  
    transpos = np.transpose(imgtmp)
    data = imgtmp+transpos

    tmp = 0   
    for i in range(len(data)):
        for j in range(len(data)):
            tmp+=data[i,j] 
    for i in range(len(data)):
         for j in range(len(data)):
            data[i,j]/=tmp
    return data

def derajat135 (img):
    max = np.max(img)
    imgtmp = np.zeros([max+1, max+1])
    for i in range(len(img)-1):
        for j in range(len(img[i])-1):
            imgtmp[img[i,j],img[i+1,j+1]] += 1
  
    transpos = np.transpose(imgtmp)
    data = imgtmp+transpos

    tmp = 0
    for i in range(len(data)):
        for j in range(len(data)):
            tmp+=data[i,j] 
    for i in range(len(data)):
        for j in range(len(data)):
            data[i,j]/=tmp
    return data

def contras(data):
    contras = 0
    for i in range(len(data)):
        for j in range(len(data)):
            contras+= data[i,j]*pow(i-j,2)
    return contras

def entropy(data):
    entro = 0
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i,j]>0.0:
                entro+= -(data[i,j]*math.log(data[i,j]))
    return entro

def homogentitas(data):
    homogen = 0
    for i in range(len(data)):
        for j in range(len(data)):
            homogen+=data[i,j]*(1+(pow(i-j,2)))
    return homogen

def energi(data):
    ener=0
    for i in range(len(data)):
        for j in range(len(data)):
            ener+=data[i,j]**2
    return ener

#<------------------Stream Menggunakan ESP32cam---------------------->

url = 'http://192.168.192.221/cam-hi.jpg'
im=None
def run1():
    cv.namedWindow("live transmission", cv.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv.imdecode(imgnp,-1)
 
        cv.imshow('live transmission',im)
        key=cv.waitKey(5)
        if key==ord('q'):
            break
    cv.destroyAllWindows()           

def run2():
    cv.namedWindow("detection", cv.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv.imdecode(imgnp,-1)
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray,(128,128))
        coba = []
        coba.append(gray)
        hasilku = []
        for i in range(len(coba)):
            dat = []
            dat.append(derajat0(coba[i]))
            dat.append(derajat45(coba[i]))
            dat.append(derajat90(coba[i]))
            dat.append(derajat135(coba[i]))
            hasilku.append(dat)
        data0energi=[]
        data0 =[]
        x=['0','45','90','135']
        data45=[]
        data90=[]
        data135=[]
        hasilnya=[]
        for j in range(len(hasilku)):
            da = []
            da.append(im[j])
            for i in hasilku[j]:
                dx = energi(i)
                da.append(dx)

                dh = homogentitas(i)
                da.append(dh)
        
                den = entropy(i)
                da.append(den)

                dco = contras(i)
                da.append(dco)
            hasilnya.append(da)
        namatabel = ['file','energy_0','homogenity_0', 'entrophy_0','contras_0'
                    ,'energy_45','homogenity_45', 'entrophy_45','contras_45'
                    ,'energy_90','homogenity_90', 'entrophy_90','contras_90'
                    ,'energy_135','homogenity_135', 'entrophy_135','contras_135']
        df = pd.DataFrame(hasilnya, columns=namatabel)
        dat= df[['energy_0','homogenity_0', 'entrophy_0','contras_0',
                'energy_45','homogenity_45', 'entrophy_45','contras_45','energy_90','homogenity_90', 'entrophy_90'
                ,'contras_90','energy_135','homogenity_135', 'entrophy_135','contras_135']].to_numpy()
        pred = model.predict(dat)
        cv.putText(im, "Object: {}".format(pred), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(pred)
        cv.imshow('detection',im)

        key=cv.waitKey(5)
        if key==ord('q'):
            break
            
    cv.destroyAllWindows()

if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
            f1= executer.submit(run1)
            f2= executer.submit(run2)


# while True:
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     gray = cv.resize(gray,(128,128))
#     coba = []
#     coba.append(gray)
#     hasilku = []
#     for i in range(len(coba)):
#         dat = []
#         dat.append(derajat0(coba[i]))
#         dat.append(derajat45(coba[i]))
#         dat.append(derajat90(coba[i]))
#         dat.append(derajat135(coba[i]))
#         hasilku.append(dat)
#     data0energi=[]
#     data0 =[]
#     x=['0','45','90','135']
#     data45=[]
#     data90=[]
#     data135=[]
#     hasilnya=[]

#     for j in range(len(hasilku)):
#         da = []
#         da.append(frame[j])
#         for i in hasilku[j]:
#             dx = energi(i)
#             da.append(dx)

#             dh = homogentitas(i)
#             da.append(dh)
        
#             den = entropy(i)
#             da.append(den)

#             dco = contras(i)
#             da.append(dco)
#         hasilnya.append(da)

#     namatabel = ['file','energy_0','homogenity_0', 'entrophy_0','contras_0'
#                 ,'energy_45','homogenity_45', 'entrophy_45','contras_45'
#                 ,'energy_90','homogenity_90', 'entrophy_90','contras_90'
#                 ,'energy_135','homogenity_135', 'entrophy_135','contras_135']
#     df = pd.DataFrame(hasilnya, columns=namatabel)
#     dat= df[['energy_0','homogenity_0', 'entrophy_0','contras_0',
#                 'energy_45','homogenity_45', 'entrophy_45','contras_45','energy_90','homogenity_90', 'entrophy_90'
#                 ,'contras_90','energy_135','homogenity_135', 'entrophy_135','contras_135']].to_numpy()
#     pred = model.predict(dat)
#     print(pred)
#     cv.putText(frame, "Object: {}".format(pred), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv.imshow('Object Classification', frame)

#     # Exit loop if 'q' key is pressed
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows