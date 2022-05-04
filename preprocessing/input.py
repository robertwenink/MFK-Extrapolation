"""
This file contains the methods for defining the parameters and search space space of the algorithm.
furthermore the object contains the data samples other possible outputs.
"""
import os.path, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import tkinter as tk
from tkinter import filedialog
import json
import time
import numpy as np

from preprocessing.GUIfunctions import GUI

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc

from utils.formatting_utils import correct_fileformatX

INPUTS_DIR = os.path.join(os.path.dirname(__file__), "input_files")

def convert_np_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')
        
class Input:
    def __init__(self, option=0):
        """
        Initialise the input class, and specify how the inputs are being read.
        Three input read options are available: reading/opening from file (1), using the previously specified file (0),
        or opening GUI in which said file will be created (2).
        """
        if option == 0:
            try:
                print("Trying to read previously defined file")
                f = open(os.path.join(INPUTS_DIR, "previous_inputfile.txt"), "r+")
                self.filename = f.readline()
                f.close()
                self.read_input()
            except:
                option = 1
                print(
                    "No previously specified file (correctly) defined, falling back to opening file"
                )
        if option == 1:
            self.open_file_prompt()
        if option == 2:
            self.open_gui()

    def open_file_prompt(self):
        filename = ""

        root = tk.Tk()
        filepath = filedialog.askopenfilename(
            initialdir=INPUTS_DIR,
            title="Open input file",
            filetypes=(("json files", "*.json"), ("all files", "*.*")),
        )
        root.destroy()

        self.filename = filepath.split(os.path.sep)[-1]

        try:
            self.read_input()
        except:
            print("Clicked cancel or invalid file, falling back to GUI")
            self.open_gui()

    def write_previous_filename(self):
        f = open(os.path.join(INPUTS_DIR, "previous_inputfile.txt"), "w+")
        f.write(self.filename)
        f.close()

    def read_X(self):
        """
        Function that converts input X of the json file from a list of lists to a nested list of nd.arrays.
        If length of the resulting list is 1, we have a single fidelity solution.
        """
        if hasattr(self,'X'):
            for i in range(len(self.X)):
                # this implies we always should define our X as a list of 2d ndarray
                self.X[i] = np.array(self.X[i],dtype=np.float64)
                assert self.X[i].ndim == 2, "not retrieving a 2 dimensional (sub)X!"

            if len(self.X) == 1:
                self.X = self.X[0]

    def read_Z(self):
        """
        See docstring of read_X.
        """
        if hasattr(self,'Z'):
            for i in range(len(self.Z)):
                # this implies we always should define our Z as a list of 1d ndarray
                self.Z[i] = np.array(self.Z[i],dtype=np.float64)
                assert self.Z[i].ndim == 1, "not retrieving a 1 dimensional (sub)Z!"
                
            if len(self.Z) == 1:
                self.Z = self.Z[0]

    def read_input(self):
        """Read the json file inputs and converting it to class attributes"""
        file_path = os.path.join(INPUTS_DIR, self.filename)
        print("Now reading {}".format(file_path))

        with open(file_path) as json_file:
            data_dict = json.load(json_file)

        # make each self.data_dict key a variable in our input object
        for key in data_dict:
            if not key == "filename":
                setattr(self, key, data_dict[key])
        
        # NOTE this part is a bit of hardcoding, in order to convert some lists back to np.array    
        self.read_X()
        self.read_Z()
        self.search_space[1] = np.array(self.search_space[1])
        self.search_space[2] = np.array(self.search_space[2])
        
        print("Reading successful")
        self.write_previous_filename()

    def create_new_filename(self, optional=""):
        if optional != "":
            optional = "_" + optional
        self.filename = time.strftime("(%Y-%m-%d)-(%H-%M-%S)") + optional + ".json"
    
    def create_input_file(self):
        """Can be used to create or update the input file according to the contents of __dict__"""

        if hasattr(self,'X'):
            self.X = correct_fileformatX(self.X)

        if hasattr(self,'Z'):
            self.Z = correct_fileformatX(self.Z)
            
        # dumping
        json.dump(
            self.__dict__, open(os.path.join(INPUTS_DIR, self.filename), "w"), default=convert_np_to_list, indent=4
        )
        self.write_previous_filename()

    def open_gui(self):
        app = qtw.QApplication([])

        gui = GUI()
        gui.show()

        app.exec_()

        self.__dict__ = gui.get_gui_dict()
        if self.__dict__ != {}:
            self.create_new_filename(gui.get_optional_filename())
            self.create_input_file()
            self.read_input()
        else:
            # in case of manually closing gui
            sys.exit()
