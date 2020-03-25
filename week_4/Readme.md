### Week 4

The goal of this week 4 is to learn about  estimating the optical flow of a video sequence, use the optical flow to stabilize any potential camera jitter, estimate the optical flow to improve an object tracking algorithm.

#### Installation

```bash
pip install -r requirements.txt

```

To use the PyFlow implementation we recommend to use the original [repository](https://github.com/pathak22/pyflow) to install all the correct packages 

#### Execution

 All the implemented tasks can be run from ```main_week_4.py``` which is used to collect all work done.

#### Directory structure

```bash
├── datasets
│   ├── kitti
│   ├── train
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
├── week_4
│   ├── metrics
│   │   ├── evaluation_funcs.py
│   │   ├── Optical_flow_metrics.py
│   │   ├── Optical_flow_visualization.py
│   ├── pyoptflow
│   ├── results
│   ├── stabilization-stas
│   ├── annotation_only_cars.txt
│   ├── detections_mask_rcnn_fine_tune.txt
│   ├── main_week_3.py
│   ├── optical_flow_off_the_shelf.py
│   ├── Tracking_farneback.py
│   ├── Tracking_Kanade.py
│   ├── Tracking_OF_Kalman.py
│   ├── bloclmatch_Backward.pkl
│   ├── bloclmatch_Forward.pkl
│   ├── videostabilization_off_the_shelf.py
│   ├── requirements.txt
```

#### Tasks done during this week:

- Task 1.1: Optical flow with Block Matching
<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/flow_45_blockMatching_quiver.png">
</div>

- Task 1.2: Off-the-shelf Optical Flow
  
  
<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/pyflow_output.png">
</div>

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/farneback_output.png">
</div>

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/lucas_output%20-%20c%C3%B2pia.png">
</div>

- Task 2.1: Video Stabilization with Block Matching

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/our_stabilization/walk_32.gif">
</div>


- T2.2: Off-the-shelf stabilization


<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/stable_video_fast.avi">
</div>

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/stable_video_orb.avi">
</div>


- T3.1: Object Tracking with Optical Flow

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/comparison.gif">
</div>



||  IDF1 (%)|
|-|-|
|Kalman week 3|85.60|
|Kalman + OF|79.94|

- T3.2: CVPR 2020 AI City Challenge

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/cam-2%20(2).gif">
</div>

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/week_4/results/cam-3.gif">
</div>
