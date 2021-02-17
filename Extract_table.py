# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 09:13:59 2021

@author: stevea

We get a Invoice Document and we need to return the table from the invoice in a excel table.
There is two types of tables invoice's :
    1) Table with horizontal and vertical lines.
    2) Table with horizontal lines only and space instead of lines for vertil lines
    

1) Apply the tesseract on the image and we get the place ( left,right,up,down) of each word .
    
2) Crop the image where there is the table. Between the words Description and Total.

3) If it's a table of type 2 , for each pixel in the width we verify if it's possible to draw a line 
    and this line is not between the left and the right of a word.
    If it's possible i draw the line on the table.

4) Get the structure table ( Threshold and find vertical and horizontal lines)
    
5) Get the coordinates of each box on the structure table.
    
6) For each box return the text that correspond to the coordinate box.
    
7) Put the table in Table excel
    
    
    

"""


import pytesseract
import pandas as pd
import numpy as np
import cv2
import more_itertools as mit


#################################################  FUNCTION  ##############################################

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
   
def detect_lines(Col_left , Col_right ,  col):
        if((col > Col_left) & (col < Col_right)):
            return 1
        else:
            return 0

def draw_horizontal_lines(img_line , height, width , col ):
    x1, y1 = col , 0
    x2, y2 = col , height
    line_thickness = 5
    cv2.line(img_line, (x1, y1), (x2, y2), (0, 255, 0), thickness = line_thickness)

def mean(a):
    return round((sum(a) / len(a)))

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return ( cnts , boundingBoxes )


def Crop_image_to_data(image_to_data , UP , DOWN , height):
        image_to_data = image_to_data.dropna()
        image_to_data['Row_up']      = image_to_data['top']
        image_to_data['Row_down']    = (image_to_data['top'] + image_to_data['height'] )
        image_to_data['Col_left']    = image_to_data['left']
        image_to_data['Col_right']   = (image_to_data['left'] + image_to_data['width'] )
        image_to_data = image_to_data.dropna(subset=['text'])
        Row_up   = image_to_data[image_to_data['text'].str.lower().str.contains(UP)].Row_up.iloc[0]
        Row_down = image_to_data[(image_to_data['text'].str.lower().str.contains(DOWN)) & (image_to_data['Row_up'] > Row_up + 0.05*height)].Row_up.iloc[0]
        image_to_data = image_to_data[(image_to_data['Row_up']>=Row_up-10) & (image_to_data['Row_up'] <= Row_down)-10 ]
        return image_to_data


def Crop_table(path,image_to_data,UP,DOWN):
    img = cv2.imread(path,0)
    height, width =  img.shape
    Row_up   = image_to_data[image_to_data['text'].str.lower().str.contains(UP)].Row_up.iloc[0]
    Row_down = image_to_data[(image_to_data['text'].str.lower().str.contains(DOWN)) & (image_to_data['Row_up'] > Row_up + 0.05*height)].Row_up.iloc[0]
    crop_img = img[Row_up - 50 :Row_down - 5, 0:width]
    height, width =  crop_img.shape
    return crop_img

 
def Add_lines(Table , image_to_data ):#, UP , DOWN):
    height, width =  Table.shape
    Vertical_lines = []
    cols =  np.arange(start = 0, stop = width , step = 1)
    image_to_data = image_to_data[image_to_data['text'].map(len) > 2]
    for col in cols:
        Help_line_sum = 0
        image_to_data['Help_line'] = 0
        image_to_data['Help_line'] = np.vectorize(detect_lines)(image_to_data['Col_left'],image_to_data['Col_right'],col)
        Help_line_sum = image_to_data['Help_line'].sum()
        if(Help_line_sum == 0 ):
            Vertical_lines.append(col)            
        # KEEP ONLY THE MIDDLE LINE :
    Vertical_lines = [list(group) for group in mit.consecutive_groups(Vertical_lines)]
    Vertical_lines = list(map(mean, Vertical_lines))
    print(Vertical_lines)  
       
    for col in Vertical_lines:
        draw_horizontal_lines(Table , height , width , int(col))
       
    return Table

   

def Table_structure(img):            
    # Otsu thresholding
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_OTSU)
    img_bin = 255-img_bin    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Extracting vertical lines :
    vertical_kernel = cv2.getStructuringElement( cv2.MORPH_RECT , (1, np.array(img).shape[1]//150))
    eroded_image = cv2.erode(img_bin, vertical_kernel, iterations=5)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=5)    
    # Extracting horizontal Lines
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//150, 1))
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=5)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)    
    #The horizontal and vertical lines are added
    vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
    thresh,vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255,cv2.THRESH_OTSU)
    #show(cv2.resize(vertical_horizontal_lines, (1500,800)))
    return vertical_horizontal_lines


def Get_boxes(vertical_horizontal_lines) :    
    contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    contours, boundingBoxes = sort_contours(contours, method ="top-to-bottom")    
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            box.append([x,y,w,h])
    row=[]
    column=[]
    j=0
    for i in range(len(box)):
        if(i==0):
            column.append(box[i])
            previous=box[i]
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]
                if(i==len(box)-1):
                    row.append(column)
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol    
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i]) ) if row[0]]
    center=np.array(center)
    center.sort()    
    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)
   
    return finalboxes , countcol , row


def Table_Tesseract(finalboxes,countcol,table,row):    
    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    #or_img = cv2.imread(path,0)
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    try:
                        y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                        if((x<=5) | (y<=5)):
                            finalimg = table[0:x+h, y:y+w]
                        else:
                            finalimg = table[x-5:x+h+5, y-5:y+w+5]
                        out = pytesseract.image_to_string(finalimg)#,config='--psm 11')
                    except:
                        out = ''
                       
                    if(len(out)== 0):
                        out = ''
                    print('text:',out)
                    #show(finalimg)
                outer.append(out)
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row),countcol))          
    return dataframe


def Detect_vertical_lines(img):            
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_OTSU)
    img_bin = 255-img_bin    
    vertical_kernel = cv2.getStructuringElement( cv2.MORPH_RECT , (1, np.array(img).shape[1]//150))
    eroded_image = cv2.erode(img_bin, vertical_kernel, iterations = 5)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=5)    
    edges = cv2.Canny(vertical_lines,0,255,apertureSize = 3)
    minLineLength = 200
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=150,lines=np.array([]), minLineLength=minLineLength,maxLineGap=2)
    try:
        if((lines == None)):
            return False
    except:
        ValueError
    if(len(lines) > 2):
        return True
    else:
        return False

###########################################################################################################

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#template2_2
#amazon_test
path = r'C:\Users\stevea\Desktop\Amazon Description\amazon_test.tif'
img = cv2.imread(path,0)
height, width  = img.shape
image_to_data = pytesseract.image_to_data(path,output_type='data.frame')#,config = '--psm 11')
UP = 'desc'
DOWN = 'total'



data  = Crop_image_to_data(image_to_data , UP , DOWN , height)      
table = Crop_table(path,data,UP,DOWN)
#show(cv2.resize(table, (1000,400)))


if(Detect_vertical_lines(table) == False):
    table = Add_lines(table,data)
   

#show(cv2.resize(table, (1000,400)))


vertical_horizontal_lines = Table_structure(table)
#show(cv2.resize(vertical_horizontal_lines, (1000,400)))
finalboxes , countcol , row = Get_boxes(vertical_horizontal_lines)
dataframe = Table_Tesseract(finalboxes,countcol,table,row)
data = dataframe.style.set_properties(align="left")
data.to_excel(r'C:\Users\stevea\Desktop\Amazon Description\TABLE.xlsx')




