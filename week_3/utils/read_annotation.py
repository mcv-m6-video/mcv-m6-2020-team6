import xml.etree.cElementTree as ET

import numpy as np


def read_annotations(file):
    root = ET.ElementTree(file=file).getroot()

    tracks = root.findall('.//track')

    annot = []
    frame = []
    xt = []
    yt = []
    width = []
    heigth = []
    id_det = []

    for track in tracks:

        label = track.attrib['label']

        if label == 'car':



            det_id = int(track.attrib['id'])




            boxes = track.findall('./box')

            for box in boxes:
                frameid = int(box.get('frame'))
                frame.append(frameid)
                xtl = float(box.get('xtl'))
                xt.append(xtl)
                ytl = float(box.get('ytl'))
                yt.append(ytl)
                xbr = float(box.get('xbr'))
                width.append(xbr - xtl + 1)
                ybr = float(box.get('ybr'))
                heigth.append(ybr - ytl + 1)
                id_det.append(det_id)

    id = np.negative(np.ones(len(frame)))

    conf = np.ones(len(frame))

    # Ai city Format [frame, -1, left, top, width, height, conf, -1, -1, -1]

    annot = np.column_stack((frame,id,xt,yt,width,heigth,conf,id_det,id,id))

    ind = np.argsort(annot[:, 0])
    annotations = annot[ind]

    np.savetxt('annotation_only_cars.txt', annotations,delimiter=',',fmt='%d')

    return annotations


"""
file = '/Users/quim/Desktop/untitled1/annotations/AI_CITY_S03_C01_391_764.xml'
read_annotations(file)
"""
