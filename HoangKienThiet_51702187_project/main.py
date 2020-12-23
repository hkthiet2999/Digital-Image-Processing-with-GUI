import PySimpleGUI as sg
import cv2
import numpy as np
import math
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import numpy
from PIL import Image
from scipy.ndimage import gaussian_filter

# function
def equalizationHistogram(gray):
    w, h = gray.shape
    r = [0]*256
    # step 1 compute r
    for x in range(w):
        for y in range(h):
            r[gray[x, y]] += 1
    # step 2 compute cdf
    cdf = [0.0]*256
    cdf = r
    for i in range(1, 256):
        cdf[i] += cdf[i-1]
    pcdf = [cdf[i]/(w*h) for i in range(256)]
    r_new = [0]*256
    for i in range(256):
        # r_new[i] = math.floor(pcdf[i]*255)
        r_new[i] = math.floor(pcdf[i]*255)
    output = np.zeros((w,h), np.uint8)

    for x in range(w):
        for y in range(h):
            output[x,y] = r_new[gray[x,y]]
    return output, pcdf

def histSpec(gray1, gray2):
    out1, pcdf1 = equalizationHistogram(gray1)
    out2, pcdf2 = equalizationHistogram(gray2)
    w, h = gray1.shape
    r_new = [0]*256
    # step 2:
    for i in range(256):
        di_min = abs(pcdf1[i] - pcdf2[0])
        k = 0
        for j in range(256):
            di_k = abs(pcdf1[i] - pcdf2[j])
            if di_min > di_k:
                k = j
                di_min = di_k
        r_new[i] = k
    
    out = np.zeros((w,h), np.uint8)
    for x in range(w):
        for y in range(h):
            out[x,y] = r_new[gray1[x,y]]
    return out

def img_not(bw):
    w, h = bw.shape
    b_not = np.zeros((w,h),np.uint8)
    for i in range(w):
        for j in range(h):
            b_not[i,j] = 255.0 - bw[i,j]
    return b_not

def img_and(bw1,bw2):
    w, h = bw1.shape
    b_and = np.zeros((w,h),np.uint8)
    for i in range(w):
        for j in range(h):
            if bw1[i,j] and bw2[i,j]:
                b_and[i,j] = bw1[i,j]
            else:
                b_and[i,j] = 0.0
    return b_and

def img_or(bw1,bw2):
    w, h = bw1.shape
    b_or = np.zeros((w,h),np.uint8)
    for i in range(w):
        for j in range(h):
            if bw1[i,j] or bw2[i,j]:
                b_or[i,j] = 0
            else:
                b_or[i,j] = bw2[i,j]
    return b_or

def STA(gray):
    T = gray.mean()
    w, h = gray.shape
    while True:
        R1 = R2 = []
        for i in range(w):
            for j in range(h):
                if gray[i, j] > T:
                    R2.append(gray[i, j])
                else:
                    R1.append(gray[i, j])
        m1 = np.mean(R1)
        m2 = np.mean(R2)
        new_T = (m1+m2)/2
        if abs(new_T - T) < 0.1:
            break
        T = new_T
    
    b = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            if gray[i, j] > T:
                b[i,j] = 1
            else:
                b[i,j] = 0
    return b   
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v
  
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

# layout 
layout = [[sg.Text('Xử lý ảnh')],
        [sg.Text('1. Biến đổi không gian màu')],
        [sg.Button('RGB to HSV'), sg.Button('RGB to YCbCr'), sg.Button('RGB to YIQ'), sg.Button('RGB to Gray')],
        [sg.Button('HSV to RGB'), sg.Button('YCbCr to RGB'), sg.Button('YIQ to RGB')],
        [sg.Text('2. Biến đổi hình học')],
        [sg.Button('Rotation'), sg.Button('Translation'), sg.Button('Zoom-in'), sg.Button('Zoom-out'), sg.Button('Deformation'), sg.Button('Perspective transformation')],
        [sg.Text('3. Các phép toán trên Histogram')],
        [sg.Button('Histogram Equalization'),sg.Button('Histogram Matching'),sg.Button('Histogram Comparisons')],
        [sg.Text('4. Xử lý hình thái')],
        [sg.Button('The logical operations'), sg.Button('Morphological operations'), sg.Button('Sharper algorithm')],
        [sg.Text('5. Làm mịn ảnh dựa trên các bộ lọc')],
        [sg.Button('Mean filter'), sg.Button('Median filter'), sg.Button('Min filter'), sg.Button('Max filter'), sg.Button('Midpoint filter'), sg.Button('Gaussian filter')],
        [sg.Text('6. Phát hiện cạnh')],
        [sg.Button('Sobel operator'), sg.Button('Prewitt operator'), sg.Button('Scharr operator'), sg.Button('Compass operator'), sg.Button('Thresholding and Sobel filter')],
        [sg.Text('7. Phân đoạn ảnh')],
        [sg.Button('The global thresholding'), sg.Button('The double thresholding for region growing')],
        [sg.Text('8. Thoát')],
        [sg.Button('Exit')]]
# rotation
layout_rts = [[sg.Text('Choose Degree')],
           [sg.Button('45 Degree - Left'), sg.Button('45 Degree - Right')], 
           [sg.Button('120 Degree - Left'), sg.Button('120 Degree - Right')], 
           [sg.Button('180 Degree'), sg.Button('270 Degree')]]
# rgb to hsv
layout_r2h= [[sg.Text('Convert RGB = (45,215,0) to HSV: ' + str(rgb2hsv(45, 215, 0)))]]
# hsv to rgb
layout_h2r= [[sg.Text('Convert HSV = (07.44186046511628, 1.0, 0.8431372549019608) to RGB: ' + str(hsv2rgb(07.44186046511628, 1.0, 0.8431372549019608)))]]
# method
layout_method = [[sg.Text('Choose method')],
           [sg.Button('Method 1'),sg.Button('Method 2'), sg.Button('Method 3')]]
# or not and xor
layout_ONAX = [[sg.Text('Choose logical')],
           [sg.Button('OR'), sg.Button('NOT')], 
           [sg.Button('AND'), sg.Button('XOR')]]

# Morphological operations
layout_mplg = [[sg.Text('Choose morphological')],
           [sg.Button('Erosion'), sg.Button('Dilate'), sg.Button('Boundary Extraction'), sg.Button('Morphological Gradient')], 
           [sg.Button('Closing'), sg.Button('Opening'), sg.Button('Top-Hat'), sg.Button('Black-Hat')]]
#----- main windows
window = sg.Window('DIP - 51702187', layout)
# switch case
while True:
    event, values = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
    # Rotation
    elif event == 'Rotation':
        window = sg.Window('Choose Degree', layout_rts)
    elif event == '45 Degree - Left':
        img = cv2.imread('ex2.png')
        w, h, c = img.shape
        center = ( w/2, h/2)
        angle = 45
        scale = 1
        M_r = cv2.getRotationMatrix2D(center, angle, scale)
        img_r = cv2.warpAffine(img, M_r,(h,w))
        result = np.hstack((img,img_r))
        cv2.imshow('Rotation - 45 Degree - Right',result)
        cv2.waitKey(0)
    elif event == '45 Degree - Right':
        img = cv2.imread('ex2.png')
        w, h, c = img.shape
        center = ( w/2, h/2)
        angle = -45
        scale = 1
        M_r = cv2.getRotationMatrix2D(center, angle, scale)
        img_r = cv2.warpAffine(img, M_r,(h,w))
        result = np.hstack((img,img_r))
        cv2.imshow('Rotation - 45 Degree - Left',result)
        cv2.waitKey(0)
    # 120
    elif event == '120 Degree - Left':
        img = cv2.imread('ex2.png')
        w, h, c = img.shape
        center = ( w/2, h/2)
        angle = 120
        scale = 1
        M_r = cv2.getRotationMatrix2D(center, angle, scale)
        img_r = cv2.warpAffine(img, M_r,(h,w))
        result = np.hstack((img,img_r))
        cv2.imshow('Rotation - 120 Degree - Right',result)
        cv2.waitKey(0)
    elif event == '120 Degree - Right':
        img = cv2.imread('ex2.png')
        w, h, c = img.shape
        center = ( w/2, h/2)
        angle = -120
        scale = 1
        M_r = cv2.getRotationMatrix2D(center, angle, scale)
        img_r = cv2.warpAffine(img, M_r,(h,w))
        result = np.hstack((img,img_r))
        cv2.imshow('Rotation - 120 Degree - Left',result)
        cv2.waitKey(0)
    # 180 270
    elif event == '180 Degree':
        img = cv2.imread('ex2.png')
        w, h, c = img.shape
        center = ( w/2, h/2)
        angle = 180
        scale = 1
        M_r = cv2.getRotationMatrix2D(center, angle, scale)
        img_r = cv2.warpAffine(img, M_r,(h,w))
        result = np.hstack((img,img_r))
        cv2.imshow('Rotation - 120 Degree - Right',result)
        cv2.waitKey(0)
    elif event == '270 Degree':
        img = cv2.imread('ex2.png')
        w, h, c = img.shape
        center = ( w/2, h/2)
        angle = 270
        scale = 1
        M_r = cv2.getRotationMatrix2D(center, angle, scale)
        img_r = cv2.warpAffine(img, M_r,(h,w))
        result = np.hstack((img,img_r))
        cv2.imshow('Rotation - 120 Degree - Left',result)
        cv2.waitKey(0)

    elif event =='Translation':
        img = cv2.imread('ex2.png')
        w, h,  = img.shape[:2]
        tx = 50
        ty = 100
        M = np.float32([ [1,0,tx] , [0,1,ty] ])
        img_trans = cv2.warpAffine(img, M, (w,h))
        cv2.imshow('Input Image',img)
        cv2.imshow('Translation Image',img_trans)
        cv2.waitKey(0)

    elif event == 'Zoom-in':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        center = ( w/2, h/2)
        angle = 0
        scale = 2
        M_r = cv2.getRotationMatrix2D(center, angle, scale)
        img_zi = cv2.warpAffine(img, M_r,(h,w))
        result = np.hstack((img,img_zi))
        cv2.imshow('Zoom-in',result)
        cv2.waitKey(0)

    elif event == 'Zoom-out':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        center = ( w/2, h/2)
        angle = 0
        scale = 0.5 
        M_r = cv2.getRotationMatrix2D(center, angle, scale)
        img_zo = cv2.warpAffine(img, M_r,(h,w))
        result = np.hstack((img,img_zo))
        cv2.imshow('Zoom-out',result)
        cv2.waitKey(0)
    # 1
    # RGB to YCbCr
    elif event == 'RGB to YCbCr':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        YCrCb = np.zeros((w,h,3), np.uint8)
        YCrCb[:,:,0] = 0.299*img[:,:,2]+0.587*img[:,:,1]+0.144*img[:,:,0]
        YCrCb[:,:,1] = 0.492*(img[:,:,0] - (0.299*img[:,:,2]+0.587*img[:,:,1]+0.144*img[:,:,0]))
        YCrCb[:,:,2] = 0.877*(img[:,:,0] - (0.299*img[:,:,2]+0.587*img[:,:,1]+0.144*img[:,:,0]))
        res = np.hstack((img,YCrCb))
        cv2.imshow('RGB to YCbCr',res)
    elif event == 'YCbCr to RGB':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        YCrCb = np.zeros((w,h,3), np.uint8)
        YCrCb[:,:,0] = 0.299*img[:,:,2]+0.587*img[:,:,1]+0.144*img[:,:,0]
        YCrCb[:,:,1] = 0.492*(img[:,:,0] - (0.299*img[:,:,2]+0.587*img[:,:,1]+0.144*img[:,:,0]))
        YCrCb[:,:,2] = 0.877*(img[:,:,0] - (0.299*img[:,:,2]+0.587*img[:,:,1]+0.144*img[:,:,0]))
        result = np.hstack((YCrCb,img))
        cv2.imshow('RGB to YCbCr',result)
    # YIQ
    elif event == 'RGB to YIQ':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        YIQ = np.zeros((w,h,3), np.uint8)
        RBG = np.zeros((w,h,3), np.uint8)
        YIQ[:,:,0] = 0.299*img[:,:,2]+0.587*img[:,:,1]+0.144*img[:,:,0]
        YIQ[:,:,1] = 0.596*img[:,:,2]-0.274*img[:,:,1]-0.322*img[:,:,0]
        YIQ[:,:,2] = 0.211*img[:,:,2]-0.523*img[:,:,1]+0.311*img[:,:,0]
        RBG[:,:,0] = YIQ[:,:,0] + 0.956*YIQ[:,:,1] + 0.621*YIQ[:,:,2]
        RBG[:,:,0] = YIQ[:,:,0] - 0.272*YIQ[:,:,1] - 0.649*YIQ[:,:,2]
        RBG[:,:,0] = YIQ[:,:,0] - 1.106*YIQ[:,:,1] + 1.703*YIQ[:,:,2]
        result = np.hstack((img,YIQ))
        cv2.imshow('RGB to YIQ',result)
    elif event == 'YIQ to RGB':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        YIQ = np.zeros((w,h,3), np.uint8)
        RBG = np.zeros((w,h,3), np.uint8)
        YIQ[:,:,0] = 0.299*img[:,:,2]+0.587*img[:,:,1]+0.144*img[:,:,0]
        YIQ[:,:,1] = 0.596*img[:,:,2]-0.274*img[:,:,1]-0.322*img[:,:,0]
        YIQ[:,:,2] = 0.211*img[:,:,2]-0.523*img[:,:,1]+0.311*img[:,:,0]
        RBG[:,:,0] = YIQ[:,:,0] + 0.956*YIQ[:,:,1] + 0.621*YIQ[:,:,2]
        RBG[:,:,0] = YIQ[:,:,0] - 0.272*YIQ[:,:,1] - 0.649*YIQ[:,:,2]
        RBG[:,:,0] = YIQ[:,:,0] - 1.106*YIQ[:,:,1] + 1.703*YIQ[:,:,2]
        result = np.hstack((YIQ,img))
        cv2.imshow('RGB to YIQ',result)
    elif event == 'RGB to HSV':
        window = sg.Window('RGB to HSV result', layout_r2h)
    elif event == 'HSV to RGB':
        window = sg.Window('HSV to RGB result', layout_h2r)
    elif event == 'RGB to Gray':
        window = sg.Window('Choose method', layout_method)
    # method 123
    elif event == 'Method 1':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        gray = np.zeros((w,h), np.uint8)
        gray[:,:] = 0.299*img[:,:,2] + 0.587*img[:,:,1] +0.144*img[:,:,0]
        cv2.imshow('Input Image',img)
        cv2.imshow('Gray Image',gray)
        cv2.waitKey(0)
    elif event == 'Method 2':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        gray = np.zeros((w,h), np.uint8)
        gray[:,:] = 0.21*img[:,:,2] + 0.72*img[:,:,1] + 0.07*img[:,:,0]
        cv2.imshow('Input Image',img)
        cv2.imshow('Gray Image',gray)
        cv2.waitKey(0)
    elif event == 'Method 3':
        img = cv2.imread('ex2.png')
        w,h,c = img.shape
        gray = np.zeros((w,h), np.uint8)
        gray[:,:] = (img[:,:,2] + img[:,:,1] + img[:,:,0])/3
        cv2.imshow('Input Image',img)
        cv2.imshow('Gray Image',gray)
        cv2.waitKey(0)
    # Histogram Equalization
    elif event == 'Histogram Equalization':
        img1 = cv2.imread("test.bmp")
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        out1, pcdf1 = equalizationHistogram(gray1)
        result1 = np.hstack((gray1, out1))
        cv2.imshow("Histogram Equalization",result1)
        cv2.waitKey(0)
    # Histogram Matching
    elif event == 'Histogram Matching':
        img1 = cv2.imread("test.bmp")
        img2 = cv2.imread("test.png")
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        out = histSpec(gray1,gray2)
        result2 = np.hstack((gray1, out))
        cv2.imshow("Histogram Matching",result2)
        cv2.waitKey(0)
    # new windows AOXN
    elif event == 'The logical operations':
        window = sg.Window('Choose logical', layout_ONAX)
    # not
    elif event == 'NOT':
        img1 = cv2.imread('r.png')
        img2 = cv2.imread('c.png')
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret1, bw1 = cv2.threshold(gray1, 50, 200, cv2.THRESH_BINARY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret2, bw2 = cv2.threshold(gray2, 50, 200, cv2.THRESH_BINARY)
        #not
        mbw1 = img_not(bw1)
        inputIMG_RC = np.hstack((img1,img2))
        cv2.imshow('Two image input',inputIMG_RC)
        cv2.imshow('The logical NOT',mbw1)
        cv2.waitKey(0)
    # and
    elif event == 'AND':
        img1 = cv2.imread('r.png')
        img2 = cv2.imread('c.png')
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret1, bw1 = cv2.threshold(gray1, 50, 200, cv2.THRESH_BINARY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret2, bw2 = cv2.threshold(gray2, 50, 200, cv2.THRESH_BINARY)
        mbw2 = img_and(bw1,bw2)
        inputIMG_RC = np.hstack((img1,img2))
        cv2.imshow('Two image input',inputIMG_RC)
        cv2.imshow('The logical AND',mbw2)
        cv2.waitKey(0)
    # or
    elif event == 'OR':
        img1 = cv2.imread('r.png')
        img2 = cv2.imread('c.png')
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret1, bw1 = cv2.threshold(gray1, 50, 200, cv2.THRESH_BINARY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret2, bw2 = cv2.threshold(gray2, 50, 200, cv2.THRESH_BINARY)
        mbw3 = img_or(bw1,bw2)
        inputIMG_RC = np.hstack((img1,img2))
        cv2.imshow('Two image input',inputIMG_RC)
        cv2.imshow('The logical OR',mbw3)
        cv2.waitKey(0)
    # xor // em bi bug ko debug dc nen xai` build-in func nha co @@
    elif event == 'XOR':
        img1 = cv2.imread('r.png')
        img2 = cv2.imread('c.png')
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret1, bw1 = cv2.threshold(gray1, 50, 200, cv2.THRESH_BINARY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret2, bw2 = cv2.threshold(gray2, 50, 200, cv2.THRESH_BINARY)
        convert_bw4 = cv2.bitwise_xor(bw1,bw2, mask=None)
        inputIMG_RC = np.hstack((img1,img2))
        cv2.imshow('Two image input',inputIMG_RC)
        cv2.imshow('The logical OR',convert_bw4)
        cv2.waitKey(0)
    # Morphological operations
    elif event == 'Morphological operations':
        window = sg.Window('Choose morphological', layout_mplg)
    elif event == 'Erosion':
        img = cv2.imread('j.png',0)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 1)
        result = np.hstack((img,erosion))
        cv2.imshow('Erosion',result)
        cv2.waitKey(0)

    elif event == 'Dilate':
        img = cv2.imread('j.png',0)
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = 1)
        result = np.hstack((img,dilation))
        cv2.imshow('Dilate',result)
        cv2.waitKey(0)
    elif event == 'Boundary Extraction':
        I = cv2.imread('edge.jpg',0)
        ret,img = cv2.threshold(I,127,255,cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3),np.uint8)
        # in
        r=cv2.erode(img,kernel,iterations=1)
        e=img-r
        rs1 = np.hstack((I,e))
        cv2.imshow('Inner Boundary Extraction',rs1)
        # out
        d = cv2.dilate(img,kernel,iterations=1)
        e=d-img
        rs2 = np.hstack((I,e))
        cv2.imshow('Outer Boundary Extraction',rs2)
        cv2.waitKey(0)

    elif event == 'Morphological Gradient':
        img = cv2.imread('j.png',0)
        kernel = np.ones((5,5),np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        result = np.hstack((img,gradient))
        cv2.imshow('Morphological Gradient',result)
        cv2.waitKey(0)
    elif event == 'Closing':
        img = cv2.imread('midterm.png',0)
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        result = np.hstack((img,closing))
        cv2.imshow('Closing',result)
        cv2.waitKey(0)
    elif event == 'Opening':
        img = cv2.imread('midterm.png',0)
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        result = np.hstack((img,opening))
        cv2.imshow('Opening',result)
        cv2.waitKey(0)
    elif event == 'Top-Hat':
        img = cv2.imread('j.png',0)
        kernel = np.ones((5,5),np.uint8)
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        result = np.hstack((img,tophat))
        cv2.imshow('Top-Hat',result)
        cv2.waitKey(0)
    elif event == 'Black-Hat':
        img = cv2.imread('j.png',0)
        kernel = np.ones((5,5),np.uint8)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        result = np.hstack((img,blackhat))
        cv2.imshow('Black-Hat',result)
        cv2.waitKey(0)
    # Smoothing
    elif event == 'Median filter':
        fig = plt.figure('Median filter')
        plt.gray()  
        ax1 = fig.add_subplot(121)  
        ax2 = fig.add_subplot(122)  
        ascent = misc.ascent()
        result = ndimage.median_filter(ascent, size=20)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()
    elif event == 'Max filter':
        fig = plt.figure('Max filter')
        plt.gray()  
        ax1 = fig.add_subplot(121)  # left img
        ax2 = fig.add_subplot(122)  # right img
        ascent = misc.ascent()
        result = ndimage.maximum_filter(ascent, size=20)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()
    elif event == 'Min filter':
        fig = plt.figure('Min filter')
        plt.gray()  
        ax1 = fig.add_subplot(121)  
        ax2 = fig.add_subplot(122)  
        ascent = misc.ascent()
        result = ndimage.minimum_filter(ascent, size=20)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()
    elif event == 'Gaussian filter':
        a = np.arange(50, step=2).reshape((5,5))
        gaussian_filter(a, sigma=1)
        fig = plt.figure('Gaussian filter')
        plt.gray()  # show the filtered result in grayscale
        ax1 = fig.add_subplot(121)  # left side
        ax2 = fig.add_subplot(122)  # right side
        ascent = misc.ascent()
        result = gaussian_filter(ascent, sigma=5)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()

    elif event == 'Mean filter':
        fig = plt.figure('Mean filter')
        plt.gray() 
        ax1 = fig.add_subplot(121)  
        ax2 = fig.add_subplot(122)  
        ascent = misc.ascent()
        result = ndimage.uniform_filter(ascent, size=20)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()

    elif event == 'Midpoint filter':
        fig = plt.figure('Midpoint filter')
        plt.gray()  
        ax1 = fig.add_subplot(121)  
        ax2 = fig.add_subplot(122)  
        ascent = misc.ascent()
        result = ndimage.rank_filter(ascent, rank=42, size=20)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()
    # Edge detection 
    elif event == 'Sobel operator':
        fig = plt.figure('Sobel operator')
        plt.gray()  
        ax1 = fig.add_subplot(121)  
        ax2 = fig.add_subplot(122)  
        ascent = misc.ascent()
        result = ndimage.sobel(ascent)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()
    elif event == 'Prewitt operator':
        fig = plt.figure('Prewitt operator')
        plt.gray()  
        ax1 = fig.add_subplot(121)  
        ax2 = fig.add_subplot(122)  
        ascent = misc.ascent()
        result = ndimage.prewitt(ascent)
        ax1.imshow(ascent)
        ax2.imshow(result)
        plt.show()
    # The global thresholding
    elif event == 'The global thresholding':
        img = cv2.imread('lena.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b = STA(gray)
        cv2.imshow('The global thresholding',b)
        cv2.waitKey(0)
    # no content
    # The double thresholding for region growing
    elif event == 'The double thresholding for region growing':
        img = cv2.imread('nocontent.jpg')
        cv2.imshow('I am sorry, no content available',img)
        cv2.waitKey(0)
    # Thresholding and Sobel filter
    elif event == 'Thresholding and Sobel filter':
        img = cv2.imread('nocontent.jpg')
        cv2.imshow('I am sorry, no content available',img)
        cv2.waitKey(0)
    # Compass operator
    elif event == 'Compass operator':
        img = cv2.imread('nocontent.jpg')
        cv2.imshow('I am sorry, no content available',img)
        cv2.waitKey(0)
    # Scharr operator
    elif event == 'Scharr operator':
        img = cv2.imread('nocontent.jpg')
        cv2.imshow('I am sorry, no content available',img)
        cv2.waitKey(0)
    # Sharper algorithm
    elif event == 'Sharper algorithm':
        img = cv2.imread('nocontent.jpg')
        cv2.imshow('I am sorry, no content available',img)
        cv2.waitKey(0)
    # Histogram Comparisons
    elif event == 'Histogram Comparisons':
        img = cv2.imread('nocontent.jpg')
        cv2.imshow('I am sorry, no content available',img)
        cv2.waitKey(0)
    # Perspective transformation
    elif event == 'Perspective transformation':
        img = cv2.imread('nocontent.jpg')
        cv2.imshow('I am sorry, no content available',img)
        cv2.waitKey(0)
    # Deformation 
    elif event == 'Deformation':
        img = cv2.imread('nocontent.jpg')
        cv2.imshow('I am sorry, no content available',img)
        cv2.waitKey(0)
# end program
window.close()