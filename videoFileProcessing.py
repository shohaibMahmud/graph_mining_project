import glob, os
import cv2
import numpy as np
import random
from utils import get_frames
from PIL import Image

os.chdir("C:\\shuvo\\graph_mining_project\\unprocessed_labels\\")
finalDir1 = "C:\\shuvo\\graph_mining_project\\train\\"
finalDir2 = "C:\\shuvo\\graph_mining_project\\test\\"
#file = open('C:\\shuvo\\graph_mining_project\\labels\\error_S_G3.csv')
videoDir = 'C:\\shuvo\\graph_mining_project\\videos\\'
totalFrame = 93266
DownsampleFactor = 2
listOfImgIndices = [n for n in range(int(totalFrame/DownsampleFactor))]
numOfFramesDone = 0
for fileName in glob.glob("*.csv"):
    file = open(fileName,'r')
    header = file.readline()
    for line in file:
        tmp = line[:-1].split(',')
        tmp2 = tmp[1].split('/')
        videoFileName = tmp2[-1]
        gesture = tmp2[-2]
        label = tmp[-1]
        videoFile = videoDir+gesture+'_sub\\'+videoFileName
        v_cap = cv2.VideoCapture(videoFile)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #totalFrame += v_len
        numOfFrames = int(v_len/2)
        train = np.random.choice([False, True], p=[0.1, 0.9])
        #line2write = ','.join(tmp)+','+str(numOfFrames)+'\n'
        dir = finalDir1
        if not train:
            dir = finalDir2
        imgData = get_frames(videoFile, numOfFrames)
        for n,frame in enumerate(imgData[0]):
            #print(frame.shape)
            img = Image.fromarray(frame, 'RGB')
            img = img.resize((240, 320), Image.ANTIALIAS)
            imgName = random.sample(listOfImgIndices, 1)[0]
            listOfImgIndices.remove(imgName)
            img.save(dir+'\\'+label+'\\'+str(imgName)+'.png')
        numOfFramesDone += numOfFrames
        print('--- '+str(((2*numOfFramesDone)/totalFrame)*100)+'% frames processed ---')
    file.close()
print(totalFrame)