# Week 5

The main goal of this week is to combine everything we've learned so far and come up with a solution to the Track 3 of [AI City Challenge](https://www.aicitychallenge.org) (i.e. a system for multi-camera multi-traget tracking).

## Installation

Please use Python 3.7+.

```bash
pip install -r requirements.txt
```

**Optional**

We optionally use EmbeddingNet as a way to improve our results in Multi-Camera Tracking. In order to use it you first have to download it from [here](https://drive.google.com/open?id=1LsnSwUDOd6CjAIBui4CKZeXnFYCQFdqA) and then run

```bash
pip install -r EmbeddingNet/requirements.txt
```

as well as adjust paths in the code.

## Execution

We provide `stas/main.ipynb` jupyter notebook for evaluating and all basic visualizations

Further methods can be explored in:

- `mtmc_features_frame.py` - MTMC - Histogram based method
- `stas/main_multicamera.py` - MTMC - Visualization with a map

## Task 1: Multi-target single-camera tracking

**Scores** for different sequences

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/filtering.png">
</div>

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/mtsc seq1.png">
</div>

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/mtsc_seq3_full.png">
</div>


**Visualization**

<div align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/single1.gif">
</div>


## Task 2: Multi-Target multi-camera tracking

**Scores** using ground-truth single-camera tracking

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/MTMC gt.png">
</div>

**Scores** using our single-camera tracking

<div align="center">
  <img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/MTMC ours.png">
</div>

**Visualization** using kalman tracking with gps coordinates and EmbeddingNet

<div align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/multihist0.gif">
</div>
<div align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/multihist1.gif">
</div>



**Visualization** using histogram of colors


<div align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/multi1.gif">
</div>
<div align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2020-team6/blob/master/results_week5/multi2.gif">
</div>

