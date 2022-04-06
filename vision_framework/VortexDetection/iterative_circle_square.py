import math
import cv2

# FUNCTIONS TO FIND SQUARE AND CIRCLE ++++++++++++++++++++++++++++++++++++++++++++++++++++


# function to find the biggest circle inside the annotation box
def circle_in_rectangle(bbox):
    x1, x2, y1, y2 = bbox
    height = y2-y1
    width = x2-x1
    cx, cy, cr = 0,0,0

    if height<0 or width<0:
        return [(cx,cy),cr]

    # center of the circle
    cx = (x1+x2)//2
    cy = (y1+y2)//2

    if height<=width:
        cr = (y2-y1)//2
    else:
        cr = (x2-x1)//2

    return [(cx,cy),cr]




# function to find the square inside a circle
def square_in_circle(circle_data):
    (cx, cy), cr = circle_data 

    # get the top left point 
    x1 = int(cx - cr*math.sqrt(2)/2)
    y1 = int(cy - cr*math.sqrt(2)/2)

    # get the bottom right point
    x2 = int(cx + cr*math.sqrt(2)/2)
    y2 = int(cy + cr*math.sqrt(2)/2)

    return [x1,x2,y1,y2]




# function to mark/return the circles inside a square/rectangle inside an image
# returns the centers and radius in a list, and the image
# color only used if draw is True
def square_to_circle_downsize(bboxes):
    centers_radius_list = [] # store the the centers and radius to return

    # unpack the bbox, calculate and store the circle
    for bbox in bboxes:
        x1, x2, y1, y2 = bbox   
        center, radius = circle_in_rectangle(bbox) 
        centers_radius_list.append([center, radius])
    
    return centers_radius_list




# function to mark/return the square inside a cirlce in an image
# returns the bboxes as x1,x2,y1,y2 in a list and the image
def circle_to_square_downsize(centers_radius_list):
    sqaures_coords = [] # store the top_left and bottom_right coords of the square to return

    for center_radius in centers_radius_list:
        x1, x2, y1, y2 = square_in_circle(center_radius)
        sqaures_coords.append([x1,x2,y1,y2])

    return sqaures_coords



# if user chooses to use them as pair
# only the final circle will be drawn
# return the image with final circle marked and 
def iterative_circle_square(bboxes, n_iter):


    # circle first then square
    # the last circle portion will be done independently
    for i in range(n_iter-1):

        centers_radius_list = square_to_circle_downsize(bboxes)
        bboxes = circle_to_square_downsize(centers_radius_list)

    # from the final squares, find the circles and draw them
    centers_radius_list = square_to_circle_downsize(bboxes)

    return centers_radius_list



# FUNCTIONS TO GET POINTS INSIDE FINAL CIRCLES +++++++++++++++++++++++++++++++++++++++++++++++


# check if point is inside rectangle
def inside_circle(center, radius, point):
    # using circle formula
    return ( (point[0]-center[0])**2 + (point[1]-center[1])**2 <= radius**2)


# function to get points inside the circular annotations
def get_points_inside_circles(shape, centers_radius_list):

    height, width, _ = shape

    # list of points in each circle
    circle_region_points_list = []

    # Method 1
    # for every bounding circle
    for circle_annotation in centers_radius_list:

        points_inside_circle = [] # list of points inside the circle

        # go thorugh each coord and store it inside a list if it is inside that circle
        for y in range(height):
            for x in range(width):
                if (inside_circle(circle_annotation[0], circle_annotation[1], (x,y))):
                    points_inside_circle.append((x,y))


        circle_region_points_list.append(points_inside_circle)

    return circle_region_points_list





# FUNCTIONS TO PREPARE THE OUTPUT FILE ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# function to write csv file
def write_circle_csv(filename, circle_region_points_list):

    detection_count = len(circle_region_points_list)

    f = open(filename,"w")
    f.write("bbox_num,x,y\n")

    for i in range(detection_count): 
        for coord in circle_region_points_list[i]: # gives the points inside circle_region_points_list[i]
            f.write(str(i)+','+str(coord[0])+','+str(coord[1])+'\n')

    f.close()