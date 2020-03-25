import numpy as np
import cv2
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from week_4.metrics.Optical_flow_metrics import *
import os
import pickle
from collections import Counter


def find_Block_Match(i,j,block,frame,searcharea,blocksize ,compensation ,h,w,center=True ,method =cv2.TM_SQDIFF_NORMED):
    #search area by center of the block.

    if center:
        loc1 = [min(i + searcharea,h),min(j + searcharea,w)]
        loc2 = [max(0, i - searcharea), max(0, j - searcharea)]
    confidence=[]
    threshold=0.4  # Apply template Matching
    search_area=frame[loc2[0]:loc1[0],loc2[1]:loc1[1]]
    res = cv2.matchTemplate(search_area, block,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:max_val = (min_val)

    if max_val>threshold and max_val<=1:
        confidence.append(max_val)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if max_val==max(confidence):
            if i in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:top_left=(min_loc)
            else:top_left=max_loc

        centeri = (j - loc2[1])
        centerj = i - loc2[0]
        if compensation=='Forward':
            ofvector = np.array(top_left)-np.array([centeri, centerj])
        else:
            ofvector = np.array([centeri, centerj])- np.array(top_left)
    else:
        ofvector=[0,0]

    return ofvector

def find_Block_Match_color(i,j,block,frame,searcharea,blocksize ,compensation,center=True,method =cv2.TM_SQDIFF_NORMED):
    #search area by center of the block.
    if blocksize>searcharea:
        searcharea=blocksize
    if center:
        maxloc = [i + searcharea, j + searcharea]
        minloc = [max(0, i - searcharea), max(0, j - searcharea)]
    else:
        maxloc = [i + 2*searcharea, j +2* searcharea]
        minloc = [i, j]


    """methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']"""

    img = frame.copy()
    confidence=[]
    threshold=0.99    # Apply template Matching
    search_area=img[minloc[0]:maxloc[0],minloc[1]:maxloc[1],:]
    res = cv2.matchTemplate(search_area, block,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if i in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:max_val = (min_val)

    if max_val>threshold:
        confidence.append(max_val)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if max_val==max(confidence):
            if i in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:top_left=(min_loc)
            else:top_left=max_loc

        centeri=int(searcharea-round((blocksize/2)))
        centerj=centeri
        centerz=centeri
        if compensation=='Forward':
            ofvector = np.array([centeri, centerj]) - np.array(top_left)
        else:
            ofvector = np.array(top_left)-np.array([centeri, centerj])
    else:
        ofvector=[0,0]

    return ofvector

def blockMatching_color(frame1,frame2,blocksize,searcharea,compensation="Forward",method =cv2.TM_SQDIFF_NORMED):
    img1=cv2.imread(frame1)
    img2=cv2.imread(frame2)

    frame1 = cv2.cvtColor(img1.astype('uint8'), cv2.COLOR_BGR2YUV)
    frame2 = cv2.cvtColor(img2.astype('uint8'), cv2.COLOR_BGR2YUV)

    frame=frame1

    if compensation=="Backward":
        frame1= frame2
        frame2= frame

    h=frame1.shape[0]
    w=frame1.shape[1]
    p=frame1.shape[2]

    of=np.zeros((h,w,3))

    bls=[]

    for i in range(0, h-blocksize,blocksize):
        for j in range(0, w-blocksize, blocksize):
                block = frame1[i :i + blocksize, j :j + blocksize,:]
                bls.append(block)
                ofvector =find_Block_Match_color(i,j,block,frame2,searcharea,blocksize,compensation,method)

                of[j :j+blocksize, i:i+blocksize,0]=ofvector[0]
                of[j :j+blocksize, i:i+blocksize,1]=ofvector[1]

    return of,bls


def preprocess_frame(image):
    image = cv2.equalizeHist(image)
    return image

def blockMatching(frame1,frame2,blocksize,searcharea,compensation="Forward",method=cv2.TM_CCORR_NORMED):
    img1=cv2.imread(frame1)
    img2=cv2.imread(frame2)

    frame1 = cv2.cvtColor(img1.astype('uint8'), cv2.COLOR_BGR2GRAY)
    frame1=preprocess_frame(frame1)
    frame2 = cv2.cvtColor(img2.astype('uint8'), cv2.COLOR_BGR2GRAY)
    frame2=preprocess_frame(frame2)


    frame=frame1
    if compensation=="Backward":
        frame1= frame2
        frame2= frame

    h=frame1.shape[0]
    w=frame1.shape[1]

    of=np.zeros((h,w,2),dtype=float)

    for i in range(0, h-blocksize,blocksize):
        for j in range(0, w-blocksize, blocksize):
            block = frame1[i :i + blocksize, j :j + blocksize]
            ofvector =find_Block_Match(i,j,block,frame2,searcharea=searcharea,blocksize=blocksize,compensation=compensation,h=h,w=w,method=method)

            of[i:i+blocksize, j:j+blocksize,0]=ofvector[0]
            of[i :i+blocksize, j:j+blocksize,1]=ofvector[1]


    return of

def grid_searchBlockMatching(blocksizes,searchareas,img1,img2,gt_dir,test_dir,seq,compensation='Forward'):

    filename='blockmatch_'+compensation+'.pkl'
    methods = [cv2.TM_CCOEFF_NORMED]

    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            metrics = pickle.load(file)
    else:
        metrics=[]
        for i in blocksizes:
            for j in searchareas:
                for p in methods:
                    print('Blocksize:',i)
                    print('Searcharea:',j)

                    of=blockMatching(img1,img2,blocksize=i,searcharea=j,compensation=compensation,method=p)
                    flow_gt, flow_test = flow_read(gt_dir,test_dir)
                    MSEN = msen(flow_gt, of, seq, 'blockmatch')
                    PEPN = pepn(flow_gt, of, 3)
                    metrics.append([i,j,MSEN,PEPN])

        with open(filename, 'wb') as f:
            pickle.dump(metrics, f)
    MSENbest=np.inf
    PEPNbest=np.inf
    blocksize=0
    searcharea=0
    for i in metrics:
        if (i[3] < PEPNbest and i[2] < MSENbest):
            PEPNbest=i[3]
            MSENbest=i[2]
            blocksize=i[0]
            searcharea=i[1]

    print('Blocksize:'+str(blocksize))
    print('Searcharea:'+str(searcharea))
    print('MSEN:'+str(MSENbest))
    print('PEPN:'+str(PEPNbest))

    visualization_gridBlockMatching(blocksizes,searchareas,metrics)
    return metrics

def grid_searchBlockMatching_color(blocksizes,searchareas,img1,img2,gt_dir,test_dir,seq,compensation='Forward'):

    filename='gridsearchBlockMatching_color_'+compensation+'.pkl'
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            metrics = pickle.load(file)
    else:
        metrics=[]
        for i in blocksizes:

            for j in searchareas:
                for m in methods:
                    print(i)
                    print(j)
                    print(m)
                    of,bls=blockMatching_color(img1,img2,blocksize=i,searcharea=j,compensation=compensation,method=cv2.TM_CCOEFF_NORMED)
                    flow_gt, flow_test = flow_read(gt_dir,test_dir)
                    MSEN = msen(flow_gt, of[:,:,0:2], seq, 'blockmatch')
                    PEPN = pepn(flow_gt, of[:,:,0:2], 3)
                    metrics.append([i,j,m,MSEN,PEPN])

        with open(filename, 'wb') as f:
            pickle.dump(metrics, f)
    MSENbest=np.inf
    PEPNbest=np.inf
    blocksize=0
    searcharea=0
    method=0
    for i in metrics:
        if (i[3] < PEPNbest and i[4] < MSENbest):
            PEPNbest=i[4]
            MSENbest=i[3]
            method=i[2]
            blocksize=i[0]
            searcharea=i[1]

    print('Blocksize:'+str(blocksize))
    print('Searcharea:'+str(searcharea))
    print('Method:'+str(method))
    print('MSEN:'+str(MSENbest))
    print('PEPN:'+str(PEPNbest))

    return metrics

def visualization_gridBlockMatching(blocksizes,searchareas,metrics):
    blocksizes_val=[]
    searchareas_val=[]
    MSEN=[]
    PEPN=[]
    for i in metrics:
        blocksizes_val.append(i[0])
        searchareas_val.append(i[1])
        MSEN.append(i[2])
        PEPN.append(i[3])

    X, Y = np.meshgrid(searchareas, blocksizes)

    n=0
    Zmsen = np.zeros([len(blocksizes), len(searchareas)])
    for i in range(0,len(blocksizes)):
        for j in range(0,len(searchareas)):
            Zmsen[i,j]=MSEN[n]
            n=n+1

    Zpepn = np.zeros([len(blocksizes), len(searchareas)])
    n=0
    for i in range(0,len(blocksizes)):
        for j in range(0,len(searchareas)):
            Zpepn[i,j]=PEPN[n]
            n=n+1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Zmsen,cmap=cm.coolwarm)
    axis = ["Seacharea", "Blocksize", "MSEN"]
    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.set_zlabel(axis[2])
    fig.colorbar(surf, shrink=0.4, aspect=6)
    plt.title('Blocksize and Searcharea optimization over MSEN')
    plt.savefig('grid_search_MSEN.png')
    plt.show()



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Zpepn,cmap=cm.coolwarm)
    axis = ["Seacharea", "Blocksize", "PEPN"]
    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.set_zlabel(axis[2])
    fig.colorbar(surf, shrink=0.4, aspect=6)
    plt.title('Blocksize and Searcharea optimization over PEPN')
    plt.savefig('grid_search_PEPN.png')
    plt.show()

def pyramidal(image, levelsup, levelsdown):
    layer = image.copy()
    layers=[]
    layerup=layer
    for i in range(levelsup):
        layerup=cv2.pyrUp(layerup)
        layers.append(layerup)


    layers.append(layer)

    layerdown=layer
    for j in range(levelsdown):
        # using pyrDown() function
        layerdown = cv2.pyrDown(layerdown)
        layers.append(layerdown)

    return layers


