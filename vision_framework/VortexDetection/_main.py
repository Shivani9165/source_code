import torch
import torch.optim as optim
from PIL import Image
import numpy as np
import argparse
from vision_framework.VortexDetection import Configration
from vision_framework.VortexDetection.Model import (CV_DetectionVorticies_YOLOv3,loaders,
load_checkpoint)
from vision_framework.VortexDetection.PlotResult import (plot_image)
from vision_framework.VortexDetection.utils import convert_yolo2rect, multi_yolo2rect, write_bbox_txt
from vision_framework.VortexDetection.iterative_circle_square import get_points_inside_circles, iterative_circle_square, write_circle_csv
import cv2
from pprint import pprint

def CVMain(input_filename):
    ## RGB Image are needed as test
    IMAGE_FILE = Image.open(input_filename) 
    IMAGE_FILE.load() 

    IMAGE_FILE_NP = np.array(IMAGE_FILE)
    if IMAGE_FILE_NP.shape[2] != 3:
        print('The test image is RGBA, you need to convert to RGB')
        return   # exit

    # Build the architecture of the model based on YOLOv3 according to the paper
    model = CV_DetectionVorticies_YOLOv3(no_classes=2).to(Configration.DEVICE)
    optimizer = optim.Adam(model.parameters())
    ## getting the loaders of specific test image
    loader = loaders(input_filename)
    # load the checkpoint after training the algorithm
    load_checkpoint(Configration.CHECKPOINT, model, optimizer)

    scaled_anchors = (torch.tensor(Configration.ANCHORS)
            * torch.tensor(Configration.R).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(Configration.DEVICE)

    # write the coordinates inside the smaller bounding circles to a file
    bbox_array = plot_image(model, loader, 0.6, 0.5, scaled_anchors)
    # print("initial yolo coords: ",bbox_array) # getting the yolo coords - check
    bbox_array = multi_yolo2rect(bbox_array, IMAGE_FILE_NP.shape)
    # print(len(bbox_array))
    return len(bbox_array)
    # print("transformed rect coords: ",bbox_array) # getting the rect coords - check
    # centers_radius_list = iterative_circle_square(bbox_array, 3)
    # # print("Circle centers radius list: ", centers_radius_list) # getting the circle centers and radius - check
    # circle_region_points_list = get_points_inside_circles(IMAGE_FILE_NP.shape, centers_radius_list)
    # # print("Points inside circle: ", circle_region_points_list) # getting the points inside bounding cirlces - check
    # write_circle_csv(input_filename.split(".")[0]+"_circlebox3.csv", circle_region_points_list)