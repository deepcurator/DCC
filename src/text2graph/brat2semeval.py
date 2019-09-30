

from os import listdir
from os.path import isfile, join
import time
import yaml


# data_path = "./Data/Abstracts-annotated/"
# output_path = "./Data/SemEvalV2/"
##data_path   = "C:/Home02/src02/DCCdev_921/grobid/Text_Files_In_Sentences_V3_Partial/"
##output_path = "C:/Home02/src02/DCCdev_921/grobid/Text_Files_In_Sentences_V3_Partial/SemEvalFiles/"
# data_path = "C:/Home02/src02/DCCdev/raw_data/RawData_BreakBrat/"
# output_path = "C:/Home02/src02/DCCdev/raw_data/RawData_BreakBrat/SemEvalFiles/"

config = yaml.safe_load(open('../../conf/conf.yaml'))
data_path = config['SENTENCE_ANNOTATED_TEXT_PATH']
output_path = config['SENTENCE_ANNOTATED_TEXT_PATH_SEMEVAL']

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

relation_dict = {}
for ii, file in enumerate(onlyfiles):
    if file.endswith(".ann"):
        print("%s ----> %d" % (file, ii))
        with open(data_path + file, encoding="utf8") as f:
            for line in f:
                if (line == "" or line == "\n" or line is None):
                    continue
                #print(line)
                content = line.strip().split("\t")
                if (len(content) == 2):
                    # print(content)
                    rel_array = []
                    if file in relation_dict:
                        rel_array = relation_dict[file]
                    rel_array.append(content)
                    relation_dict[file] = rel_array

print(relation_dict)

import copy

def insert(source_str, insert_str, pos):
    return source_str[:pos] + insert_str + source_str[pos:]


semeval_output_file_name = 'SemEval_Output.txt'
semeval_output_file_name_path = join(output_path, semeval_output_file_name)

with open(semeval_output_file_name_path, 'w', encoding="utf8") as output_file:
    index = 1
    for file in relation_dict.keys():
        relationships_of_file = relation_dict[file]
        # print(relationships_of_file)
        for relationship in relationships_of_file:
            # print(relationship)
            # if (index == 80):
            #     print(relationship)
            content = relationship[1].split(" ")
            relationship_type = content[0]
            e1 = content[1]
            e2 = content[2]
            # print(relationship_type)
            # print(e1)
            # print(e2)
            if (e1.startswith("Arg1:") or e1.startswith("Arg2:")):
                e1 = e1[5:]
            if (e2.startswith("Arg1:") or e2.startswith("Arg2:")):
                e2 = e2[5:]
            rel_reversed = 0
            # print(e1)
            # print(e2)
            with open(data_path + file, encoding="utf8") as f:
                for line in f:
                    if (line == "" or line == "\n" or line is None):
                        continue
                    content = line.strip().split("\t")
                    if (content[0] == e1):
                        e1_start = int(content[1].split(" ")[
                                           1])  ### changed by ioannis 2019/9/13 (added "-1") --- for reading UWA data...
                        e1_end = int(content[1].split(" ")[
                                         2])  ### changed by ioannis 2019/9/13 (added "-1") --- for reading UWA data...
                    if (content[0] == e2):
                        e2_start = int(content[1].split(" ")[
                                           1])  ### changed by ioannis 2019/9/13  (added "-1")--- for reading UWA data...
                        e2_end = int(content[1].split(" ")[
                                         2])  ### changed by ioannis 2019/9/13  (added "-1")--- for reading UWA data...

            if (e2_start < e1_start):
                rel_reversed = 1
                temp_start = copy.deepcopy(e1_start)
                temp_end = copy.deepcopy(e1_end)
                e1_start = copy.deepcopy(e2_start)
                e1_end = copy.deepcopy(e2_end)
                e2_start = copy.deepcopy(temp_start)
                e2_end = copy.deepcopy(temp_end)

            if (e2_start < e1_end):
                print("SKIPPUNG")
                print(e1 + " " + str(e1_start) + " " + str(e1_end))
                print(e2 + " " + str(e2_start) + " " + str(e2_end))
                continue

            if (index == 80):
                print(e1 + " " + str(e1_start) + " " + str(e1_end))
                print(e2 + " " + str(e2_start) + " " + str(e2_end))

            file_txt = file.replace(".ann", ".txt")
            output_str = ""
            with open(data_path + file_txt, encoding="utf8") as f:
                # txt = f.readline()
                txt_all = f.read().splitlines()
                for txt in txt_all:
                    if txt == "":  ### added by ioannis - 2019/9/13
                        print("empty line....")
                    else:
                        # print(txt)
                        # print(txt[e1_start:e1_end])
                        # print(txt[e2_start:e2_end])
                        txt = insert(txt, "</e2>", e2_end)
                        txt = insert(txt, "<e2>", e2_start)
                        txt = insert(txt, "</e1>", e1_end)
                        txt = insert(txt, "<e1>", e1_start)
                        # print(str(index) + "\t" + "\"" + txt + "\"")
                        output_str += str(index) + "\t" + "\"" + txt + "\"\n"
                        # print(output_str)
                        if (rel_reversed == 0):
                            # print(relationship_type + "(e1, e2)")
                            output_str += relationship_type + "(e1, e2)\n"
                        else:
                            # print(relationship_type + "(e2, e1)")
                            output_str += relationship_type + "(e1, e2)\n"
                        # print("Comment:\n")
                        output_str += "Comment:\n\n"
            output_file.write(output_str)
            index += 1
            print("processing file ", index)
