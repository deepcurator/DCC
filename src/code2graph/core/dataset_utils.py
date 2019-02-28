from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

import os
import importlib
import sys
import time
import wget
import zipfile

class datasetParser:
    '''Parser class to parse a given github repository
        Args: parent_dir: Parent directory path of the github repository wanted
        to be parsed.

    '''
    def __init__ (self, parent_dir = None):
        '''Initializing the class'''
        self._parent_dir = parent_dir
        self._main_file_path = None
        self._main_file_name = None
        self._main_file_dir = None
        self._code_lines = None
        self._compile_index = None
        self._prefix_for_modification = None
        self._newfile = None
        pass

    def findMain (self, parent_dir = None):
        '''Function to find the main file'''
        main_file_list = []
        if(parent_dir!=None):
            self._parent_dir = parent_dir
        for root, dirs, files in os.walk(self._parent_dir, topdown=True):
            for name in files:
                if(name.endswith(".py")):
                    file_path = os.path.join(root, name)
                    try:
                        f = open(file_path, 'r')
                    except:
                        continue
                    file_contents = f.read()
                    keyword = "if __name__ == '__main__':"
                    alt_keyword = "if __name__ == \"__main__\":"
                    if (keyword in file_contents) or (alt_keyword in file_contents):
                        if "test" not in name:
                            main_file_list.append(file_path)
                            condition_to_select = (("model" in name) or \
                                                    ("train" in name)) and \
                                                    (self._main_file_path==None)
                            if condition_to_select:
                                self._main_file_path = file_path
                    else:
                        continue
                else:
                    continue

        if self._main_file_path==None:
            self._main_file_path = min(main_file_list, key=len)

        if(self._main_file_path!=None):
            main_path_splitted = self._main_file_path.split('/')
            self._main_file_dir = '/'.join(main_path_splitted[:-1])
            self._main_file_name = ((main_path_splitted[-1]).split('.'))[0]
        pass

    def findMainKeras (self, parent_dir = None):
        '''Function to find the main file for Keras'''
        main_file_list = []
        if(parent_dir!=None):
            self._parent_dir = parent_dir
        for root, dirs, files in os.walk(self._parent_dir, topdown=True):
            for name in files:
                if(name.endswith(".py")):
                    file_path = os.path.join(root, name)
                    try:
                        f = open(file_path, 'r')
                        file_contents = f.read()
                    except:
                        continue
                    keyword = ".compile("
                    if (keyword in file_contents):
                        main_file_list.append(file_path)
                        condition_to_select = (("model" in name) or \
                                                ("train" in name)) and \
                                                (self._main_file_path==None)
                        if condition_to_select:
                            self._main_file_path = file_path
                    else:
                        continue
                else:
                    continue

        if self._main_file_path==None:
            if(main_file_list):
                self._main_file_path = min(main_file_list, key=len)
                main_path_splitted = self._main_file_path.split('/')
                self._main_file_dir = '/'.join(main_path_splitted[:-1])
                self._main_file_name = ((main_path_splitted[-1]).split('.'))[0]
                return 1
            else:
                return 0
        else:
            main_path_splitted = self._main_file_path.split('/')
            self._main_file_dir = '/'.join(main_path_splitted[:-1])
            self._main_file_name = ((main_path_splitted[-1]).split('.'))[0]
            return 1
        pass

    def getLinebyLine(self, main_file_path = None):
        '''Function to convert the main file to code lines'''
        if(main_file_path!=None): self._main_file_path = main_file_path
        f = open(self._main_file_path, 'r')
        file_contents = f.read()
        line_by_line = file_contents.split('\n')[:-1]
        line_list = []
        temp = ''
        for line in line_by_line:
            if '#' in line:
                line = line[:line.index('#')]
            line = line.rstrip()
            if line and (line[-1]==',' or line[-1]=='\\' or line[-1]=='('):
                if not temp:  #If temp is empty then it is a new line
                    if(line[-1]=='\\'):
                        temp = temp + line[:-1]
                    else:
                        temp = temp + line
                else: #Otherwise it is a continuation of line
                    if(line[-1]=='\\'):
                        temp = temp + (line.lstrip())[:-1] #Then add it to the previous portion
                    else:
                        temp = temp + line.lstrip() #Then add it to the previous portion
            else:
                if(temp):
                    if line==')' and temp[-1]==',': #If the line is just a closing poaranthesis
                        temp = temp[:-1] #The previous line was ending with a comma
                    temp = temp + line.lstrip()
                    line_list.append(temp)
                    temp = ''
                else:
                    line_list.append(line)
        self._code_lines = line_list
        pass

    def findCompileKeras(self, code_lines = None):
        '''Function to find the compile line in Keras'''
        if(code_lines!=None): self._code_lines = code_lines
        for line in self._code_lines:
            if '.compile(' in line:
                space_count = 0
                for i in line: #How many spaces does it have in front?
                    if i==' ': space_count+=1
                    else: break
                if(space_count): #If it is using spaces then add spaces
                    self._prefix_for_modification = space_count * ' '
                else: #If it is using tabs
                    self._prefix_for_modification = line.count('\t') * '\t'
                self._compile_index = self._code_lines.index(line)
        pass

    def modifyCodeKeras(self):
        '''Function to modify the code if compile is found'''
        #These are the lines to add:
        line_1 = self._prefix_for_modification + 'from keras import backend as K'
        line_2 = self._prefix_for_modification + 'import tensorflow as tf'
        line_3 = self._prefix_for_modification + 'sess = K.get_session()'
        line_4 = self._prefix_for_modification + 'writer = tf.summary.FileWriter(\''
        line_4 = line_4 + os.path.join(self._parent_dir, '')
        line_4 = line_4 + '\', sess.graph)'
        line_5 = self._prefix_for_modification + 'import sys'
        line_6 = self._prefix_for_modification + 'sys.exit()'
        self._code_lines.insert(self._compile_index+1, line_1)
        self._code_lines.insert(self._compile_index+2, line_2)
        self._code_lines.insert(self._compile_index+3, line_3)
        self._code_lines.insert(self._compile_index+4, line_4)
        self._code_lines.insert(self._compile_index+5, line_5)
        self._code_lines.insert(self._compile_index+6, line_6)
        pass

    def saveModifiedCode(self):
        '''Save the modified code to a new file with extension '_modified' '''
        new_code = '\n'.join(self._code_lines)
        self._newfile = self._main_file_name + '_modified.py'
        new_file_path = os.path.join(self._main_file_dir, self._newfile)
        f = open(new_file_path, 'w')
        f.write(new_code)
        f.close()
        pass

    def installRequirements(self, errorname = None):
        '''Fuction to install the required libraries'''
        sys.path.insert(0, self._main_file_dir)
        try:
            i = importlib.import_module(self._main_file_name)
        except ImportError as error:
            # Output expected ImportErrors.
            try:
                x = error.name
            except AttributeError as e:
                error.name = (str(error).split(' '))[-1]
            if(error.name != errorname): #Prevent infinite loop
                to_install = error.name
                os.system('python -m pip install ' + to_install)
                self.installRequirements(errorname = error.name)
        except Exception as e:
            pass
        pass

    def runtheCode(self):
        '''Function to run the code'''
        #os.system('python ' + os.path.join(self._main_file_dir, self._newfile))
        os.chdir(self._main_file_dir)
        os.system('python ' + self._newfile)
        # file = open('/home/aymat/Research/finished_papers', 'a')
        # file.write((self._parent_dir.split('/'))[-1] + '\n')
        # file.close()
        pass

def parseCode(parent_dir = None):
    '''Main Parser used to parse a given github repository folder'''
    parser = datasetParser(parent_dir=parent_dir)
    if(parser.findMainKeras()):
        print("\nKeras is found!")
        parser.getLinebyLine()
        print ("\nParsing data for: {}.py...".format(parser._main_file_name))
        parser.findCompileKeras()
        print ("\nModiying the code...")
        parser.modifyCodeKeras()
        parser.saveModifiedCode()
        print("\nCode has been successfully modified and saved as {}".format(parser._newfile))
        print("\nInstalling Requirements if there is any...")
        parser.installRequirements()
        print("\nRequirements have been successfully installed!!")
        try:
            print("\nGetting the event file...\n")
            parser.runtheCode()
            print("\nEvent file has been successfully acquired and saved under {}!".format(parser._parent_dir))
        except:
            pass


#Written by Arquimedes Canedo
def fetch_paper(paper_link, path):
    '''function to download the paper pdf with a given arxiv link'''
    decomposed = paper_link.split('/')
    if 'arxiv' in decomposed[2]:
        decomposed[3] = 'pdf'
        decomposed[4] += '.pdf'
        reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + decomposed[3] + "/" + decomposed[4]
        filename = wget.download(reconstructed, out=path) #Saves the pdf file to given path
        #print(reconstructed)
    else:
        print("Don't know how to fetch %s" % paper_link)

#Written by Arquimedes Canedo
def fetch_code(code_link, path):
    '''function to download the github repository as a zip file'''
    decomposed = code_link.split('/')
    assert len(decomposed) == 5
    if 'github' in decomposed[2]:
        decomposed.append('archive/master.zip')
        reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + decomposed[3] + "/" + decomposed[4] + "/" + decomposed[5]
        filename = wget.download(reconstructed, out=path) #Saves the github repository to the given path
        #print(reconstructed)
    else:
        print("Don't know how to fetch %s" % paper_link)


def unzip_code(path):
    '''function to unzip the zip file with the given path'''
    output_path = os.path.split(path)[0]
    zip_obj = zipfile.ZipFile(path, 'r')
    zip_obj.extractall(output_path)
    zip_obj.close()


def delete_file(path):
    '''function to delete file with a given path'''
    try:
        os.remove(path)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))


def unzip_all(dir_path):
    '''function to unzip all of the zip files in a directory and its sub-directories'''
    for path, dir_list, file_list in os.walk(dir_path):
        for file_name in file_list:
            if file_name.endswith(".zip"):
                abs_file_path = os.path.join(path, file_name)
                unzip_code(abs_file_path)
                delete_file(abs_file_path)
