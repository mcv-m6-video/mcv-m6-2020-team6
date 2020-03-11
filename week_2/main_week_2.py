from week_2.background_sota import *
from week_2.map_vs_alpha import *
from week_2.model.backgroundSubstration import *
from week_2.adaptative_optimization import evaluation_adaptative_non_recursive, evaluation_adaptative_recursive
from week_2.utils.preprocessing import *

def task0():

    preprocess_annotations("annotation_track.txt")


def task11():
    """Gaussian modelling and implementation"""

    background_gaussian()


def task12():
    """mAP0.5 vs Alpha"""

    evaluation_map_vs_alpha()


def task21():
    """Adaptive modelling"""

    background_adaptive_gaussian()


def task211(gridsearch):
    """evaluation and optimization of adaptive modelling"""

    if gridsearch:
        evaluation_adaptative_recursive()

    else:
        evaluation_adaptative_non_recursive()


def task22():
    """Comparison adaptive vs non"""


def task3(method):
    """Comparison with state-of-the-art"""

    if method == 'MOG':
        background_subtraction_sota(method, visualize=True)

    elif method == 'MOG2':
        background_subtraction_sota(method, visualize=True)

    elif method == 'KNN':
        background_subtraction_sota(method, visualize=True)


def task4(color,adaptive):
    """Color sequences"""

    background_gaussian_color(color,adaptive)

def task4_grid(color):
    evaluation_map_vs_alpha_color(color)

if __name__ == '__main__':
    #task0()
    #task11()
    #task12()
    #task21()
    #task211(gridsearch=True)
    #task22()
    #task3('MOG')
    #task4(color="hsv",adaptive=True)
    task4_grid(color="hsv")