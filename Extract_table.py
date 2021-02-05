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

###############################################  FUNCTIONS  ###############################################

# Show the image
def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
   
# Help function to detect if in specific col it's posible to draw a line
def detect_lines(Col_left , Col_right ,  col):
        if((col > Col_left) & (col < Col_right)):
            return 1
        else:
            return 0
        
# This function get image and draw vertical line at the col pixel
def draw_horizontal_lines(img_line , height, width , col ):
    x1, y1 = col , 0
    x2, y2 = col , height
    line_thickness = 15
    cv2.line(img_line, (x1, y1), (x2, y2), (0, 255, 0), thickness = line_thickness)

# Help function
def mean(a):
    return sum(a) / len(a)

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

########################################################################################################

path = r'C:\Users\stevea\Desktop\Amazon Description\amazon2.tif'

#Apply the Tesseract on the image :
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img = cv2.imread(path,0)
height, width  = img.shape
image_to_data = pytesseract.image_to_data(path,output_type='data.frame')#,config = '--psm 11')
image_to_data.columns

# Get the left right top and down of each word :
image_to_data['Row_up']      = image_to_data['top']
image_to_data['Row_down']    = (image_to_data['top'] + image_to_data['height'] )
image_to_data['Col_left']    = image_to_data['left']
image_to_data['Col_right']   = (image_to_data['left'] + image_to_data['width'] )


image_to_data = image_to_data.dropna(subset=['text'])

# Crop the table from the image:
Row_up_Description = image_to_data[image_to_data['text'].str.contains('Units')].Row_up.iloc[0]
Row_up_Total       = image_to_data[(image_to_data['text'].str.contains('Total')) & (image_to_data['Row_up'] > Row_up_Description + 0.1*height)].Row_up.iloc[0]
crop_img           = img[Row_up_Description - 20 :Row_up_Total - 5, 0:width]
img = crop_img

Vertical_lines = []
cols =  np.arange(start = 0, stop = width, step = 1)

image_to_data = image_to_data[(image_to_data['Row_up']>=Row_up_Description-10) & (image_to_data['Row_up']<=Row_up_Total)-10]
image_to_data = image_to_data[image_to_data['text'].map(len) > 1]


# Find the columns line where there is not line :
for col in cols:
    image_to_data['Help_line'] = np.vectorize(detect_lines)(image_to_data['Col_left'],image_to_data['Col_right'],col)
    Help_line_sum = image_to_data['Help_line'].sum()
    if(Help_line_sum == 0 ):
        Vertical_lines.append(col)


# If fin severall lines keep the middle lines :
Vertical_lines = [list(group) for group in mit.consecutive_groups(Vertical_lines)]
Vertical_lines = list(map(mean, Vertical_lines))

# Draw the lines:
for col in Vertical_lines:
    draw_horizontal_lines(img , height , width , int(col))
 
    
edges = cv2.Canny(img,0,255,apertureSize = 3)
minLineLength = 200  
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=150,lines=np.array([]), minLineLength=minLineLength,maxLineGap=2)
 

a,b,c = lines.shape
for i in range(a) :
    # print(lines[i][0][0])
    # print(lines[i][0][1])
    # print(lines[i][0][2])
    # print(lines[i][0][3])
    cv2.line( img , (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA )
   


###############################################################################################################################################


# Otsu thresholding
thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_OTSU)
img_bin = 255-img_bin

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Extracting vertical lines :
vertical_kernel = cv2.getStructuringElement( cv2.MORPH_RECT , (1, np.array(img).shape[1]//150))
eroded_image = cv2.erode(img_bin, vertical_kernel, iterations=5)
#show(eroded_image)
vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=5)
#show(vertical_lines)

#Extracting horizontal Lines
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//150, 1))
image_2 = cv2.erode(img_bin, hor_kernel, iterations=5)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)
#show(cv2.resize(horizontal_lines, (1500,800)))

#The horizontal and vertical lines are added
vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
#show(vertical_horizontal_lines)

thresh,vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255,cv2.THRESH_OTSU)
#show(cv2.resize(vertical_horizontal_lines, (1500,800)))

##
contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# Sort all the contours by top to bottom.
contours, boundingBoxes = sort_contours(contours, method ="top-to-bottom")

#Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
#Get mean of heights
mean = np.mean(heights)



#Create list box to store all boxes in  
box = []
# Get position (x,y), width and height for every contour and show the contour on image
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w<1000 and h<500):
        image = cv2.rectangle(vertical_horizontal_lines,(x,y),(x+w,y+h),(0,255,0),2)
        box.append([x,y,w,h])



#Creating two lists to define row and column in which cell is located
row=[]
column=[]
j=0
#Sorting the boxes to their respective row and column
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


#calculating maximum number of cells
countcol = 0
for i in range(len(row)):
    countcol = len(row[i])
    if countcol > countcol:
        countcol = countcol

#Retrieving the center of each column
center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i]) ) if row[0]]
center=np.array(center)
center.sort()

#Regarding the distance to the columns center, the boxes are arranged in respective order
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

image_to_data['help'] = 0
#from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
outer=[]
or_img = cv2.imread(path,0)
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner=''
        if(len(finalboxes[i][j])==0):
            outer.append(' ')
        else:
            for k in range(len(finalboxes[i][j])):
                y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                finalimg = or_img[x-30+Row_up_Description:x+h+Row_up_Description-15, y:y+w]
                #out = pytesseract.image_to_string(finalimg,config='--psm 11')
                c = 20
                out = image_to_data[(image_to_data['Row_up'] >= x-c+Row_up_Description  )&(image_to_data['Row_down'] <= x+h+Row_up_Description)&
                                    (image_to_data['Col_left'] >= y-c) &(image_to_data['Col_right'] <= y+w+c)]
                out['text'] =  out.groupby(['help'])['text'].transform(lambda x: ' '.join(x))
                try:
                    out = out['text'].iloc[0]
                except:
                    out = ''
                if(len(out)== 0):
                    out = ''
            outer.append(out)
           

#Creating a dataframe of the generated OCR list
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row),countcol))
data = dataframe.style.set_properties(align="left")
 


data.to_excel(r'C:\Users\stevea\Desktop\Amazon Description\RESULT_TABLE.xlsx')





