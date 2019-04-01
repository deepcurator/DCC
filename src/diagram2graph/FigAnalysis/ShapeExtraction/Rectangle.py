import numpy as np


class Rectangle:
    def __init__(self, x, y=None, w=None, h=None, m=0, s=0):
        """
        rect [x, y, w, h]
        coordinate: (0,0) pos at top left
        (x1, y1): the top left pos
        (x2, y2): the bottom right pos
        m: rect merged times
        s: area size index
        """
        self.merged_times = m
        self.area_size_index = s
        if isinstance(x, list) or isinstance(x, np.ndarray):
            self.x1 = x[0]
            self.y1 = x[1]
            self.x2 = x[0] + x[2]
            self.y2 = x[1] + x[3]
        else:
            self.x1 = x
            self.y1 = y
            self.x2 = x + w
            self.y2 = y + h

    @staticmethod
    def from_2_pos(x1, y1, x2, y2, m=0, s=0):
        return Rectangle(x1, y1, x2 - x1, y2 - y1, m, s)

    def get_height(self):
        return self.y2 - self.y1

    def get_width(self):
        return self.x2 - self.x1

    def get_area(self):
        return self.get_width() * self.get_height()

    def set_area_size_index(self, s):
        self.area_size_index = s
        return self

    def __eq__(self, other):
        return self.x1 == other.x1 and self.x2 == other.x2 \
               and self.y1 == other.y1 and self.y2 == other.y2

    def __str__(self):
        return "#(%0.2f, %0.2f), (%0.2f, %0.2f)#" % (self.x1, self.y1, self.x2, self.y2)
