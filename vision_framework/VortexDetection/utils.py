import numpy as np
import math


# function to convert yolo coordinate system to a rect bounding box 
# may be used to get the coordinates of the rect bounding box to crop vortex region
# the function assumes that the image supplied is grayscale (2 shape parameter)
# np gives shape as height, width
def convert_yolo2rect(yolo_coords, img_shape):
    center_x, center_y, width, height = yolo_coords # get the yolo coords
    start_x = int ((center_x-width/2)*img_shape[1])
    end_x = int ((center_x+width/2)*img_shape[1])
    start_y = int ((center_y-height/2)*img_shape[0])
    end_y = int ((center_y+height/2)*img_shape[0])
    return [start_x, end_x, start_y, end_y]


# change from image size to numerical domain
def change_domain(absolute_bbox_array, img_shape):
    absolute_bbox_array_new = []
    for absolute_bbox in absolute_bbox_array:
        absolute_bbox_array_new.append( (absolute_bbox/img_shape[0])*2*math.pi)
    return absolute_bbox_array_new



# convert the entire bounding box array to absolute coordinates
def multi_yolo2rect(bbox_array, img_shape):
    bbox_array_new = []
    for bbox in bbox_array:
        bbox_array_new.append(np.asarray (convert_yolo2rect(bbox, img_shape)) )
    # print(bbox_array_new)
    # print(change_domain(bbox_array_new, img_shape))
    return bbox_array_new
    # return change_domain(bbox_array_new, img_shape) # for CIDA domain




# write bounding boxes to csv for use by CIDA
def write_bbox_csv(filename, bbox_data):
    f = open(filename, "w")
    f.write("top_left_x, bottom_right_x, top_left_y, bottom_right_y\n")
    for box in bbox_data:
        for coord in box:  # coordinate in each line
            f.write(str(coord))
            f.write(",")
        f.write("\n") # next box
    f.close()


# write bounding boxes to txt for use by CIDA
def write_bbox_txt(filename, bbox_data):
    np.savetxt(filename, bbox_data, "%d")
