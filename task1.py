"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
     
    # Computing Key points and Descriptors using SIFT
    kp,des=enrollment(characters)   
    
    # Computing detected characters and Coordinates
    Samples,Cordinates=detection(test_img)
    
    # Matching computed features in Enrollment to test Image
    results=recognition(characters,Samples,Cordinates,kp,des)
    
    return results
    #raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    
    # Using SIFT to create features for characters
    sift = cv2.SIFT_create()
    
    kp_all=[]   # Stores Key Points for all enrolled characters
    des_all=[]  # Stores Descriptors for all enrolled characters
        
    for tag, img in characters:
       ret,img = cv2.threshold(img,127,255,0)
       
       #Computing Key points and descriptors for each character
       kp,des=sift.detectAndCompute(img,None)
       kp_all.append(kp)
       des_all.append(des)
       
    return (kp_all,des_all)
    #raise NotImplementedError

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    
    # Converting Gray Image to Binary
    ret,thresh = cv2.threshold(test_img,127,255,0)
    
    # Storing a copy to draw bounding boxes
    detection=thresh.copy()    
    
    # Get rows and columns in test image
    height=thresh.shape[0]
    width=thresh.shape[1]   
    
    # Initialising the Labelled Image
    labelled_img=np.zeros((height,width))
    
    #Setting Current Label to 1
    label=1    
    
    """Sequential Two Pass Algorithm For Connected Component Labelling"""    
    
    # 1st pass: labelling the Foreground Pixels
    for i, row in enumerate(thresh):
        for j, pixel in enumerate(row):
            if pixel==255:
                continue
            elif pixel==0:
                north,west=i-1,j-1
                if thresh[north][j] and thresh[i][west] != 0:
                    labelled_img[i][j]=label
                    label+=1
                elif thresh[north][j]==0:
                    labelled_img[i][j]=labelled_img[north][j]
                elif thresh[i][west]==0:
                    labelled_img[i][j]=labelled_img[i][west]
                elif thresh[north][j]:
                    labelled_img[i][j]=labelled_img[i][west]
    
    # 2nd Pass: Merging Equivalent Regions
    for i, row in enumerate(thresh):
        for j, pixel in enumerate(row):
            if pixel==255:
                pass
            elif pixel==0:
                north,south,east,west=i-1,i+1,j+1,j-1
                """
                Finding pixels which are essentially 
                the same region but have different labels and 
                Replacing them with a single label
                """
                # Checks if north and west neighbours have a label and merges them 
                if labelled_img[north][j]!=0 and labelled_img[i][west]!=0:
                    m=min(labelled_img[north][j],labelled_img[i][west])
                    labelled_img[i][j]=m
                    if labelled_img[north][j]!=m:
                        x=labelled_img.copy()
                        x=x.reshape((height*width))
                        n=labelled_img[north][j]
                        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                        x[get_indexes(n,x)]=m
                        labelled_img=x.reshape((height,width))
                    if labelled_img[i][west]!=m:
                        x=labelled_img.copy()
                        x=x.reshape((height*width))
                        n=labelled_img[i][west]
                        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                        x[get_indexes(n,x)]=m
                        labelled_img=x.reshape((height,width))
                
                # Checks if south and east neighbours have a label and merges them
                if labelled_img[south][j]!=0 and labelled_img[i][east]!=0:
                    m=min(labelled_img[south][j],labelled_img[i][east],labelled_img[i][j])
                    if thresh[south][j]==0 and labelled_img[south][j]!=m:                        
                        x=labelled_img.copy()
                        x=x.reshape((height*width))
                        n=labelled_img[south][j]
                        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                        x[get_indexes(n,x)]=m
                        labelled_img=x.reshape((height,width))
                    if thresh[i][east]==0 and labelled_img[i][east]!=m:
                        x=labelled_img.copy()
                        x=x.reshape((height*width))
                        n=labelled_img[i][east]
                        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                        x[get_indexes(n,x)]=m
                        labelled_img=x.reshape((height,width))
                
                # Checks if only one of the neighbours have a label
                elif labelled_img[north][j]!=0 and labelled_img[i][west]==0:
                    labelled_img[i][j]=labelled_img[north][j]
                elif labelled_img[i][west]!=0 and labelled_img[north][j]==0:
                    labelled_img[i][j]=labelled_img[i][west]             
                
    #Assign sequential labels to objects in order of detection
    for i, lab in enumerate(np.unique(labelled_img)):        
        x=labelled_img.copy()
        x=x.reshape((height*width))
        n=lab
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        x[get_indexes(n,x)]=i
        labelled_img=x.reshape((height,width)) 
    
    # Remove background label            
    labels=np.unique(labelled_img)
    labels=np.delete(labels,0)
    
    # Find Coordinates for each detected label 
    Cord_x=[]
    Cord_y=[]
    for i in range(len(labels)+1):
        Cord_x.append([])
        Cord_y.append([])
    
    # Finding all occurences of a label in the labelled image
    for i in range(height):
        for j in range(width):
            val=labelled_img[i][j]
            if val in labels:
                k=int(val)                
                Cord_x[k].append(j)
                Cord_y[k].append(i)
    Cord_x[0]=[0,0]
    Cord_y[0]=[0,0]
    
    # Initialising list of detected characters and Coordinates 
    Samples=[]
    Cords=[]    
       
    """ 
    Computes Coordinates for each label and stores each label 
    as an image array in a list of samples
    """
    for i in range(len(labels)+1):        
        x_max=max(Cord_x[i])
        x_min=min(Cord_x[i])
        y_min=min(Cord_y[i])
        y_max=max(Cord_y[i]) 
        x=x_min
        y=y_min
        w=x_max-x_min+1
        h=y_max-y_min
        sample_i=thresh[y:y+h,x:x+w]
        Samples.append(np.asarray(sample_i))        
        Cords.append([x,y,w,h])
        detection = cv2.rectangle(detection, (x,y),(x+w,y+h), (0,255,0), 1)
    
    # Removes Background Label '0'
    Cords.pop(0)
    Samples.pop(0)
    
    # Add Padding to the detected images    
    for i in range(len(Samples)):
        img=Samples[i]
        top = int(0.2 * img.shape[0])  # shape[0] = rows
        bottom = top
        left = int(0.2 * img.shape[1])  # shape[1] = cols
        right = left
        borderType = cv2.BORDER_CONSTANT
        
        img=cv2.copyMakeBorder(img,top,bottom,left,right,borderType,None,(255,0,0))
        Samples[i]=img
        
    # Display Result of detection
    show_image(detection) 
      
    return(Samples,Cords)
    
    #raise NotImplementedError
    
def recognition(characters,Samples,Cordinates,kp,des):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    results=[]      
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()     
    
    for i in range(len(Samples)):
        # Get Coordinates for each sample
        Cord=Cordinates[i]
        
        # Duplicating each test sample to find features        
        test=Samples[i]
        
        
       
        # Using a FLANN based matcher 
        flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)                     
        
        # A lsit to keep track of all matches with each enrolled character
        match=[]
        
        # Initialise counter for each enrolled character
        j=0
        for tag, img in characters:
            # To count number of matches
            count=0
            
            # Scaling the enrolled character by 250% and resizing the sample image to match the size of the character
            upscale=250
            w=int(img.shape[1]*upscale/100)
            h=int(img.shape[0]*upscale/100)
            dim=(w,h)
            img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
            c=img.shape[0]
            r=img.shape[1]
            dim=(r,c)
            test=cv2.resize(test,dim,interpolation=cv2.INTER_AREA)
            
            # Compute key points and descriptors for each candidate character in the test image
            kp_test,des_test = sift.detectAndCompute(test,None)
            
            # Compute matches 
            try:
                matches = flann.knnMatch(des[j],des_test,k=2)
            except:
                pass
            
            # Find only good matches, so create a mask            
            matchesMask = [[0,0] for i in range(len(matches))]
            
            # Choosing only the good matches by setting threshold to 0.5
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.5*n.distance:
                    matchesMask[i]=[1,0]
                    # Increment count for each match found
                    count+=1               
            
            ## To Visualise the each match
            draw_params = dict(matchColor = (0,255,0),
                                singlePointColor = (255,0,0),
                                matchesMask = matchesMask,
                                flags = cv2.DrawMatchesFlags_DEFAULT)
            img3 = cv2.drawMatchesKnn(characters[j][1],kp[j],test,kp_test,matches,None,**draw_params)
            # show_image(img3)
            
            # Store quality of match with each enrolled character
            match.append(count)
            j+=1
        
        # Check match with which character is maximum and if no match return "UNKNOWN"
        if max(match)==0:
            string="UNKNOWN"               
        else:
            string=characters[match.index(max(match))][0]
        
        # Store recognised test character in the result            
        result={"bbox": Cord ,"name":string}        
        results.append(result)       
       
    return results
    #raise NotImplementedError 


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)
   
    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
