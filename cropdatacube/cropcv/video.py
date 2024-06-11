import os
import glob
import cv2

def check_folder(folderpath, verbose = False):

    if folderpath is not None:
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
            if verbose: 
                print("following path was created {}".format(folderpath))
    else:
        folderpath = ""
    return folderpath


def from_video_toimages(video_path, outputpath = None, frame_rate = 5, preffix = "image"):

    outputpath = check_folder(outputpath)

    vidcap = cv2.VideoCapture(video_path)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(os.path.join(outputpath, preffix+str(count)+".jpg"), image)     # save frame as JPG file
        return hasFrames

    sec = 0

    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frame_rate
        sec = round(sec, 2)
        success = getFrame(sec)