### Week 3

The goal of this week 3 is to learn about using an Object Detection framework, fine tune the object detection network and track movin object of a scene.

#### Installation

```bash
pip install -r requirements.txt
```
#### Execution

 All the implemented tasks can be run from ```main_week_3.py``` which is used to collect all work done.

#### Directory structure

```bash
├── datasets
│   ├── kitti
│   ├── train
├── week_3
│   ├── dl_frameworks
│   │   ├── fine_tune
│   │   ├── Mask_R-CNN
│   │   ├── RetinaNet
│   │   ├── YOLOV3
│   ├── metrics
│   │   ├── evaluation_funcs.py
│   │   ├── Optical_flow_metrics.py
│   │   ├── Optical_flow_visualization.py
├── model
│   │   ├── bbox.py
│   │   ├── frame.py
│   │   ├── frameExtraction.py
│   │   ├── read_annotation.py
│   │   ├── video.py
├── utils
│   │   ├── colors.py
│   │   ├── iou_over_time_RT.py
│   │   ├── show_bb_singleFrame.py
│   │   ├── show_frame.py
│   │   ├── sort.py
│   │   ├── tracking_utils.py
│   ├── week_3_results
│   ├── annotation.txt
│   ├── annotation_only_cars.txt
│   ├── detections_mask_rcnn.txt
│   ├── detections_mask_rcnn_fine_tune.txt
│   ├── detections_mask_rcnn_fine_tune_25.txt
│   ├── detections_mask_rcnn_fine_tune_30.txt
│   ├── detections_mask_rcnn_fine_tune_40.txt
│   ├── detections_retinanet.txt
│   ├── detections_yolo.txt
│   ├── detections_yolo_default.txt
│   ├── fine_tunning.py
│   ├── m6-full_annotation.xml
│   ├── main_week_3.py
│   ├── off_the_shelf.py
│   ├── tracking_kalman.py
│   ├── tracking_overlap.py
│   ├── requirements.txt
```

#### Tasks done during this week:

- Task 1.1: Of-the-Shelf Object Detection
<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_3/week_3_results/yolo_off_the_shelf.gif">
</div>

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_3/week_3_results/retinanet_off_the_shelf.gif">
</div>

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_3/week_3_results/mask_rcnn_off_the_shelf.gif">
</div>


- Task 1.2 : Fine-Tuned Object Detection
  
  
<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_3/week_3_results/Mask%20R-CNN%20fine-tune.gif">
</div>



- Task 2.1: Tracking by maximum overlap

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_3/week_3_results/gif_tracking_overlap.gif">
</div>



- Task 2.2: Tracking by Kalman Filters


<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_3/week_3_results/gif_tracking_Kalman.gif">
</div>


- Task 2.3: IDF1 for Multiple Object Tracking


|| Overlap IDF1 (%) | Kalman IDF1 (%)|
|-|-|-|
|YOLO (OTS)|63.20|66.10|
|RetinaNet (OTS)|65.76|71.85|
|Mask RCNN|70.29|72.43|
|Mask RCNN (FT)|75.24|85.05|

Summary of all the week 3 results:

| | mAP@0.5 | Overlap IDF1 (%) | Kalman IDF1 (%)|
|-|-|-|-|
|YOLO (OTS)|0.5437|63.20|66.10|
|RetinaNet (OTS)|0.5225|65.76|71.85|
|Mask RCNN|0.5720|63.20|72.43|
|Mask RCNN (FT)|0.7232|75.24|85.05|



