import glob, os
import cv2
import numpy as np
import random
from utils import get_frames
from PIL import Image

os.chdir("C:\\shuvo\\graph_mining_project\\unprocessed_labels\\")
final_img_dir1 = "C:\\shuvo\\graph_mining_project\\train2\\img\\"
final_img_dir2 = "C:\\shuvo\\graph_mining_project\\test2\\img\\"
final_lbl_dir1 = "C:\\shuvo\\graph_mining_project\\train2\\label\\labels.csv"
final_lbl_dir2 = "C:\\shuvo\\graph_mining_project\\test2\\label\\labels.csv"
#file = open('C:\\shuvo\\graph_mining_project\\labels\\error_S_G3.csv')
videoDir = 'C:\\shuvo\\graph_mining_project\\videos\\'

totalTrainingSample = 0
totalTestingSample = 0
totalFrame = 93266
#DownsampleFactor = 2
#listOfImgIndices = [n for n in range(int(totalFrame/DownsampleFactor))]
numOfFramesDone = 0
numOfFramesInSample = 60
train_label_file = open(final_lbl_dir1, 'w')
test_label_file = open(final_lbl_dir2, 'w')
train_label_file.write('sample_number,label\n')
test_label_file.write('sample_number,label\n')

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
        numOfFrames = v_len
        train = np.random.choice([False, True], p=[0.1, 0.9])
        #line2write = ','.join(tmp)+','+str(numOfFrames)+'\n'
        dir = final_img_dir1
        totalSample = totalTrainingSample
        labelFile = train_label_file
        if not train:
            dir = final_img_dir2
            totalSample = totalTestingSample
            labelFile = test_label_file

        imgData = get_frames(videoFile, numOfFrames)
        numOfSamples = int(numOfFrames/60)
        if numOfSamples>0:
            startFrame = 0
            for sampleNumber in range(numOfSamples):
                labelFile.write(str(totalSample)+','+label+'\n')
                for n,frame in enumerate(imgData[0][startFrame:startFrame+numOfFramesInSample]):
                    img = Image.fromarray(frame, 'RGB')
                    img = img.resize((240, 320), Image.ANTIALIAS)
                    imgName = str(totalSample) + '_' + str(n)
                    img.save(dir + '\\' +  str(imgName) + '.png')
                startFrame += numOfFramesInSample
                totalSample += 1
            if train:
                totalTrainingSample = totalSample
            else:
                totalTestingSample = totalSample
        numOfFramesDone += numOfFrames
        print('--- '+str((numOfFramesDone/totalFrame)*100)+'% frames processed ---')
    file.close()
train_label_file.close()
test_label_file.close()