import cv2
import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np



THRESHOLD = 5

# define color for each type of bbox
color_type = {
    "text":  (0, 255,0) ,
     "formula":  (0, 0, 255) ,
     "illustration": (255, 0, 255)
}


def get_path(path : str) -> list:
    '''
    Get all path of files in folder
    Args:
        path: path to folder
    Returns:
        list of path
    '''
    return os.listdir(path)


def getinfo_annot(path : str) -> dict:
    '''
    Read json file
    Args:
        path: path to json file
    Returns:
        dict of json file
    
    '''
    
    with open(path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    return content

def calcenter(bbox : tuple) -> tuple:
    '''
    calculate center of bbox
    Args:
        bbox: tuple of bbox
    Returns:
        tuple of center of bbox
    '''
    # calculate center of bbox
    return (int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2))



def statics_height_bbox(bboxes : list) -> int:
    '''
    Statics height of bbox and calculate median height
    Args:
        bboxes: list of bbox
    Returns:
        median height of bbox
    '''
    # calculate height of bbox
    height = []
    for i in bboxes:
        height.append(i[0][3] - i[0][1])
    # sort and get median height
    height.sort()
    height = height[len(height) // 2]
    return height


def grouplines(bboxes : list, width_ : int) -> dict:
    '''
    Group lines by y center of bbox
    Args:
        bboxes: list of bbox
    Returns:
        dict of lines
    '''
    
    # get median height of bbox
    median_height = statics_height_bbox(bboxes)

    # group lines by y center of bbox   
    # fairs is dict of lines and metadata of bbox
    fairs = {}
    count = 1
    idx = 0
    prev = 0
    while True:
        
        '''
        Rule base, check type of bbox and distance between 2 center of bbox compared with median height of bbox
        Median height of bbox is calculated by statics_height_bbox function (that statics all of height bbox in image)
        That rule base cover 8 / 10 image, 2 / 10 image wrong because of line is tilted
        '''
        
        #check if type of bbox != formula
        
        if bboxes[idx][2] !=  'formula' :
            # check if type next bbox is text
            if bboxes[idx + 1][2] == 'text' :
                # check if distance between 2 center of bbox > median height of bbox / 4 then split line prev -> idx + 1
                if bboxes[idx + 1][1][1] - bboxes[idx][1][1] > median_height // 4  :
                    # group line by index of line
                    fairs[count] = bboxes[prev:idx + 1]
                    # update prev index 
                    prev = idx + 1
                    # count line split
                    count += 1
            # check if type next bbox is formula
            elif bboxes[idx + 1][2] == 'formula' :
                # check if distance between 2 center of bbox > median height of bbox / 3 then split line prev -> idx + 1
                if bboxes[idx + 1][1][1] - bboxes[idx][1][1] > median_height // 3  :
                    # group line by index of line
                    fairs[count] = bboxes[prev:idx + 1]
                    # update prev index 
                    prev = idx + 1
                    # count line split
                    count += 1
            # check if type next bbox is illustration
            elif bboxes[idx + 1][2] == 'illustration':
                # check if distance between 2 center of bbox > median height of bbox then split line prev -> idx + 1
                if bboxes[idx + 1][1][1] - bboxes[idx][1][1] > median_height   :
                    # group line by index of line
                    fairs[count] = bboxes[prev:idx + 1]
                    # update prev index 
                    prev = idx + 1
                    # count line split
                    count += 1
        # check if type of bbox == formula
        else: 
            # check if distance between 2 center of bbox > median height of bbox / 2 then split line prev -> idx + 1
            if bboxes[idx + 1][1][1] - bboxes[idx][1][1] > median_height //2  :
                # group line by index of line
                fairs[count] = bboxes[prev:idx + 1]
                # update prev index 
                prev = idx + 1
                # count line split
                count += 1
                
        idx += 1
        if idx == len(bboxes) - 1:
            fairs[count] = bboxes[prev:]
            break
    
    # sort by x center of bbox
    for itemkey in fairs:
        # sort by x
        fairs[itemkey].sort(key=lambda x: x[1][0])

    return fairs


def drawbboxes(anno : dict) -> None:
    
    '''
    Draw bboxes and put index of line and index of bbox for each bbox
    Args:
        anno: dict of json file
    Returns:
        None
    '''
    
    start = time.time()
    
    # read image with opencv
    prefix = './Images'
    image =  anno['images'][0]['file_name']
    
    img = cv2.imread(os.path.join(prefix, image))
    

    
    width_, height_ = anno['images'][0]['width'], anno['images'][0]['height']
    
 
    
    # get bbox, and calculate center of bbox for each bbox
    bboxes = [(i['bbox'], calcenter(i['bbox']), i['category'], i['value']) for i in anno['annotations']]
    # sort by y center of bbox
    bboxes.sort(key=lambda x: x[1][1])
    # call function grouplines to group lines by y center of bbox and sort by x center of bbox
    fairs = grouplines(bboxes, width_)
    
    
    # file_name_txt = './convert/ ' + image.split('.')[0] + '.txt'
    
    # write index of line and index of bbox for each bbox to txt file
    # with open(file_name_txt, 'w', encoding='utf-8') as file:
    #     for bbx in fairs:
    #         for ii, i in enumerate(fairs[bbx]):
    #             file.write(f'{i[3]}\t')
    #         file.write('\n')
    
    
    # draw bbox and put index of line and index of bbox for each bbox
    for bbx in fairs:
       
        for ii, i in enumerate(fairs[bbx]):
            # get position of bbox
            xmin = int(i[0][0])
            ymin = int(i[0][1])
            xmax = int(i[0][2])
            ymax = int(i[0][3])
            # put index of line and index of bbox for each bbox
            cv2.putText(img, f'{bbx , ii + 1}', (i[1][0], i[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
            #draw bbox
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_type[i[2]], 2)
    # save image if you want
    # cv2.imwrite('./convert/' + image, img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # save image
   
    
    print(round((time.time() - start), 2))
    


    # plt.imshow(img)
    # plt.show()
  
    
    cv2.imshow('image', img)     
    cv2.waitKey(0)
    cv2.destroyAllWindows()     

if __name__ == "__main__":
    
    # get all path of images and annotations
    # you must create images folder and annotations folder in same folder with solution.py
    images = ['./images/' + str(x) for x in  get_path('./images')]
    annotations = ['./annotations/' + str(x) for x in  get_path('./annotations')]
    
    # get info of json file
    info = [getinfo_annot(x) for x in annotations]
    
    # call function drawbboxes to draw bbox and put index of line and index of bbox for each bbox   
    
    for i in info:
        drawbboxes(i)
       