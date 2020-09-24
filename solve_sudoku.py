# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:26:14 2020

@author: Dell
"""

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

import imutils
import numpy as np
import cv2

def find_puzzle(image):
    #image = cv2.imread(r"H:\sudoku_solver(opencv,backtracking)\sudoku_puzzle.jpg")
            
            #def find_puzzle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 3)
    
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)   
    thresh = cv2.bitwise_not(thresh)
    
    
    #cv2.imshow("thresh", thresh)
    #cv2.waitKey(0)
    
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    grabbed_contours = imutils.grab_contours(contours)
    grabbed_contours_s = sorted(grabbed_contours, key =cv2.contourArea, reverse = True )
    
    puzzle_cnt = None
    
    approx = []
    for cnt in grabbed_contours_s:
        peri = cv2.arcLength(cnt,closed = True)
        poly_curve = cv2.approxPolyDP(cnt,epsilon = 0.02*peri,closed = True )
        approx.append(poly_curve)  #just for checking, not relevant
        if len(poly_curve)==4:
            puzzle_cnt = poly_curve
            break
    
      
        
    if puzzle_cnt is None:
        raise Exception("no contours found try updating the threshold and stuff")
    output = image.copy()
    cv2.drawContours(output,[puzzle_cnt],-1,(255,0,0),2)
    #v2.imshow("contour_highlighted", output)
    #cv2.waitKey(0)
    puzzle_cnt_reshaped = puzzle_cnt.reshape(4,2)
    
    puzzle = four_point_transform(output, puzzle_cnt_reshaped)
    warped = four_point_transform(gray, puzzle_cnt_reshaped)
    
    return(warped, puzzle)

'''cv2.imshow("warped", warped)
cv2.waitKey(0)'''

#TODO: find the digit up in that puzzle

def extract_digit(roi):
    #roi = warped[0:42,0:53]       # got by analysing the values in warped
    thresh = cv2.threshold(roi, 0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cleared = clear_border(thresh)
    
    contours = cv2.findContours(cleared.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grabbed_contours_roi = imutils.grab_contours(contours)
    
    if len(grabbed_contours_roi)==0:                #no need to find biggest contour as the roi is already given 
        return None
    
    big_c = max(grabbed_contours_roi, key = cv2.contourArea) 
    mask = np.zeros(thresh.shape, dtype = "uint8")
    cv2.drawContours(mask, [big_c], -1,255,2)
    '''cv2.imshow("mask", mask)   
    cv2.waitKey(0)''' 
    
    (w,h)= thresh.shape
    percent_filled = cv2.countNonZero(mask)/float(w*h)
    if percent_filled<0.03:
        return None
    
    digit = cv2.bitwise_and(thresh,thresh, mask = mask)
    return digit

from tensorflow.keras.models import load_model

model = load_model(r"H:\sudoku_solver(opencv,backtracking)\digit_classifier.h5") 

image = cv2.imread(r"H:\sudoku_solver(opencv,backtracking)\sudoku_puzzle.jpg")

image = imutils.resize(image,width = 600)

board = np.zeros((9,9), dtype = "int")

(warped,puzzle) = find_puzzle(image)

stride_x = warped.shape[0]//9
stride_y = warped.shape[1]//9

from tensorflow.keras.preprocessing.image import img_to_array

for y in range(0,9):
    for x in range(0,9):
        
        startx = x*stride_x            
        endx = x*stride_x+stride_x
        starty = y*stride_y
        endy = y*stride_y+stride_y
        
        cell = warped[startx:endx,starty:endy]
        
        digit = extract_digit(cell)
        
        if digit is not None :
            
            roi = cv2.resize(digit, (28,28))   # mnist digit classifier take an input of 28x28
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi , axis = 0)
            pred = model.predict(roi)
            pred_2 = model.predict(roi).argmax(axis=1)
            prediction = model.predict(roi).argmax(axis= 1)[0]
            board[x,y]= prediction
            
#sudoku solving part
board = board.tolist()   
'''image2 = cv2.imread(r"H:\sudoku_solver(opencv,backtracking)\sudoku_puzzle.jpg" )
cv2.imshow('d', image2)
cv2.waitKey(0)'''

def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row,col = find
    for i in range(1,10):
        if check_board_validity(bo,i,(row,col)) == True:
            bo[row][col]=i
            
        
            if solve(bo) ==True:  #recursion
             return True
                   
            bo[row][col] = 0
        
    return False    
               

#check_board_validity(board,5,(1,5))
def check_board_validity(bo, entry, pos):  # checking if the current no fits according to the rules of sudoku
    for i in range(len(bo[0])):
        #check row
        if bo[pos[0]][i]==entry and pos[1]!=i:
            return False
        
    #check column 
    for i in range(len(bo)):
        if bo[i][pos[1]]==entry and pos[0]!=i:
            return False 
    box_y = pos[1] //3                      # check the little 3x3 grid that you're in 
    box_x = pos[0] //3
    for i in range(box_x*3,box_x*3+3):
        for j in range(box_y*3,box_y*3+3):
            if bo[i][j]==entry and (i,j)!= pos:
                return False
    return True        
    
def print_board(bo):  #visualize the board (not important)
    for i in range(len(bo)):
        if i% 3==0 and i!=0:
            print("- - - - - - - - - - - - -")
        for j in range(len(bo[0])):
            if j%3==0 and j!=0:
                print(" | " , end= "")
            #if j==8:    
            print(bo[i][j] ,end = " ") 
        print("\n")    
            
       

def find_empty(bo):  # return the spot with a 0 
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j]==0:
                return (i,j)
    return None        

print_board(board)
print("------------------------------------------")
solve(board)
print_board(board)


            

            
        












    
    
    