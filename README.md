# Video Surveillance for Road Traffic Monitoring

The goal of this project is to learn the basic concepts and techniques related to video sequences processing, mainly for surveillance applications. We will focus on video sequences from outdoor scenarios, with the application of traffic monitoring in mind. The main techniques of video processing will be applied in the context of video surveillance: background modelling, moving object segmentation, motion estimation and compensation and video object tracking.


# Team 6 

| Members | Contact |
| :---         |   :---    | 
| Marc Pérez   | marc.perezq@e-campus.uab.cat | 
| Claudia Baca    | claudiabaca.perez@e-campus.uab.cat  |
| Quim Comas    | joaquim.comas@e-campus.uab.cat  |



### Week 1

The goal of this week 1 is to learn about the datasets to be used and implement the evaluation metrics and graphs used during the module.

#### Installation

```bash
pip install -r requirements.txt
```
#### Execution

 All the implemented tasks can be run from ```main_week_1.py``` which is used to collect all work done.

#### Directory structure

```bash
├── datasets
│   ├── kitti
│   ├── train
├── week_1
│   ├── metrics
│   │   ├── evaluation_funcs.py
│   │   ├── Optical_flow_metrics.py
│   │   ├── Optical_flow_visualization.py
│   ├── model
│   │   ├── bbox.py
│   │   ├── frame.py
│   │   ├── frameExtraction.py
│   │   ├── read_annotation.py
│   │   ├── video.py
│   ├── utils
│   │   ├── iou_over_time_RT.py
│   │   ├── show_bb_singleFrame.py
│   │   ├── show_frame.py
│   ├── annotation.txt
│   ├── main_week_1.py
│   ├── requirements.txt
```

#### Tasks done during this week:

- Task 1: Detection metrics.
  
  To check our IoU implementation we applied two modifications on random displacement and deletion of the provided groundtruth as we       appreciate in the next example tested in frame 391:
  
  ![alt](https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week1/iou_results_frame391.png)
  
  
  Also, we computed the 11 interpolated mean Average Precision (mAP@0.5) to evaluate our object detection in future labs:  

  ![alt](https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week1/precision_recall_11_interp_gt_video_modif1.png)


- Task 2: Detection metrics. Temporal analysis.
  
  In terms of temporal analysis we have used the IoU over time. In the next example we can observe the result of 3 detectors (Mask-RCNN,   YOLO V3, SSD512):
  
  ![alt](https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week1/iou_detectors.png)



- Task 3: Optical flow evaluation metrics.

  We implemented the MSEN and PEPN metrics for Optical Flow evaluation. Also, we introduced a histogram and image error visualization:  

  ![alt](https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week1/histogram_45.png)

  ![alt](https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week1/error_image_45.png)



- Task 4: Visual representation optical flow.

  We have tried two different optical flow visualizations:
  
  Quiver-based visualization 


  ![alt](https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week1/flow_gt_45_quiver.png)

  Color-based visualization


  ![alt](https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week1/flow_gt_45_color.png)











