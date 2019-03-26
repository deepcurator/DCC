from __future__ import print_function
import math
from scipy.spatial import distance
import cv2

class LineMerger:
    
    def __init__(self):
        self.min_distance_to_merge = 10
        self.min_angle_to_merge = 1
        self.min_point_line_dist = 10
        
        
    def get_lines(self, lines_in):
        if cv2.__version__ < '3.0':
            return lines_in[0]
        return [l[0] for l in lines_in]
    
    
    def get_distance(self,line1, line2):
        dist1 = self.DistancePointLine(line1[0][0], line1[0][1], 
                                  line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = self.DistancePointLine(line1[1][0], line1[1][1], 
                                  line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = self.DistancePointLine(line2[0][0], line2[0][1], 
                                  line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = self.DistancePointLine(line2[1][0], line2[1][1], 
                                  line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    
    
        return min(dist1,dist2,dist3,dist4)
    
    
    
    def merge_lines_pipeline(self, lines):
        super_lines_final = []
        super_lines = []
    
        for line in lines:
            create_new_group = True
            group_updated = False
    
            for group in super_lines:
                for line2 in group:
                    if self.get_distance(line2, line) < self.min_distance_to_merge:
                        # check the angle between lines       
                        orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                        orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))
    
                        if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < self.min_angle_to_merge: 
                            #print("angles", orientation_i, orientation_j)
                            #print(int(abs(orientation_i - orientation_j)))
                            group.append(line)
    
                            create_new_group = False
                            group_updated = True
                            break
    
                if group_updated:
                    break
    
            if (create_new_group):
                new_group = []
                new_group.append(line)
    
                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if self.get_distance(line2, line) < self.min_distance_to_merge:
                        # check the angle between lines       
                        orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                        orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))
    
                        if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < self.min_angle_to_merge: 
                            #print("angles", orientation_i, orientation_j)
                            #print(int(abs(orientation_i - orientation_j)))
    
                            new_group.append(line2)
    
                            # remove line from lines list
                            #lines[idx] = False
                # append new group
                super_lines.append(new_group)


        for group in super_lines:
            super_lines_final.append(self.merge_lines_segments(group))
    
        return super_lines_final
    
    
    def lineMagnitude(self, x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
        return lineMagnitude
        


    def DistancePointLine(self, px, py, x1, y1, x2, y2):
        #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        LineMag = self.lineMagnitude(x1, y1, x2, y2)
    
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine
    
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)
    
        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = self.lineMagnitude(px, py, x1, y1)
            iy = self.lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = self.lineMagnitude(px, py, ix, iy)
    
        return DistancePointLine

    
    def merge_lines_segments(self,lines, use_log=False):
        if(len(lines) == 1):
            return lines[0]
    
        line_i = lines[0]
    
        # orientation
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    
        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])
    
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < 135:
    
            #sort by y
            points = sorted(points, key=lambda point: point[1])
    
            if use_log:
                print("use y")
        else:    
            #sort by x
            points = sorted(points, key=lambda point: point[0])
    
            if use_log:
                print("use x")
    
        return [points[0], points[len(points)-1]]
        
        
#    def _is_point_near(self, px, py, x1, y1, x2, y2):
#        return self.DistancePointLine(px, py, x1, y1, x2, y2)  < min_point_line_dist
               

    def isMergableLine(self, lines, px, py):
        for index in range(len(lines)):
            line = lines[index]
            #print("line index %d"%(index))
            
            for x1, y1, x2, y2 in line:   
                if min(distance.euclidean([px, py], [x1, y1]), distance.euclidean([px, py], [x2, y2])) < self.min_point_line_dist:
                    if distance.euclidean([px, py], [x1, y1])<distance.euclidean([px, py], [x2, y2]):
                        return (True, index, x1, y1)
                    else:
                        return (True, index, x2, y2)
        return (False, -1, -1, -1)