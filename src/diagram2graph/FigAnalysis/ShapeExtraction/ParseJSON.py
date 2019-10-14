from __future__ import print_function
import argparse
import os
import cv2
import sys
import code
import csv
import numpy as np
import json
import glob
import subprocess
import difflib
 

class ParseJSON:

	def __init__(self):

		self.temp = 0

	def getCaption(self, fileWithPath):

		image_file_name = os.path.splitext(os.path.basename(fileWithPath))[0]
		abspath = os.path.abspath(fileWithPath)
		parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(abspath)))
		csv_file_path = os.path.join(parent_dir, "pwc_edited_tensorflow_pytorch_image_final.csv")
		paper_title, paper_file_name, paper_conf, paper_year = self.getPaperTitle(image_file_name, csv_file_path)

		jsonFilePath = os.path.join(os.path.dirname(abspath), paper_file_name + ".json")
		
		fp = open(jsonFilePath, 'r')
		json_value = fp.read()
		raw_data = json.loads(json_value)
		
		fig_caption = ""
		fig_text = []
		
		for index, item_dict in enumerate(raw_data):

			if image_file_name in item_dict["renderURL"]:
				fig_caption = item_dict["caption"]
				fig_text = item_dict["imageText"]
				

		return paper_title, paper_file_name, paper_conf, paper_year, fig_caption, fig_text		

		
	def getPaperTitle(self, image_file_name, csv_file_path):
		csvfile = open(csv_file_path, encoding='utf-8')
		reader = list(csv.reader(csvfile))
		max_match = 0
		match_index = 0
		for index, item in enumerate(reader):				
			f_name = item[2].split('/')[-1]				
			matcher = difflib.SequenceMatcher(None, f_name, image_file_name)
			match = matcher.find_longest_match(0, len(f_name), 0, len(image_file_name))
			
			if match.size > max_match :
				max_match = match.size
				match_index = index 

		paper_title = reader[match_index][1]
		paper_file_name = (reader[match_index][2].split('/')[-1]).replace(".html", "").replace(".pdf", "")
		return paper_title, paper_file_name, reader[match_index][3], reader[match_index][4]


	def isResult(self, caption):
		if (caption.lower().find('result') != -1 or caption.lower().find('plot') != -1 or caption.lower().find('graph') != -1 or caption.lower().find('image') != -1 ): 
			return True
		return False

	def isDiag(self, caption):
		if (caption.lower().find('diagram') != -1 or caption.lower().find('network') != -1 or caption.lower().find('framework') != -1 or 
			caption.lower().find('flowchart') != -1 or caption.lower().find('flow chart') != -1 or caption.lower().find('architecture') != -1) : 
			return True
		return False


