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

INPUTS_DIR = os.path.join(os.path.dirname(__file__), "input_files")

class Encoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_dict_to_array(d):
    # Base case: if value is a numerical list (of lists), convert to NumPy array
    if isinstance(d, list):
        # in case of list of lists of numerical
        if (all(isinstance(i, list) and all(isinstance(j, (int, float)) for j in i) for i in d) and len({len(i) for i in d}) == 1):
            return np.array(d)
        # in case of list of numerical
        elif all(isinstance(i, (int, float)) for i in d):
            return np.array(d)
        # then a list (of lists) but not fully numerical
        else:
            return [convert_dict_to_array(value) for value in d]
    
    # Recursive case: if value is a dictionary go level deeper
    if isinstance(d, dict):
        return {int(key) if key.isdigit() else key: convert_dict_to_array(value) for key, value in d.items()}

    # Return value as is if it is not a numerical list or dictionary
    return d


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


    def read_input(self):
        """Read the json file inputs and converting it to class attributes"""
        file_path = os.path.join(INPUTS_DIR, self.filename)
        print("Now reading {}".format(file_path))

        with open(file_path) as json_file:
            data_dict = json.load(json_file)

        data_dict = convert_dict_to_array(data_dict)

        # make each self.data_dict key a variable in our input object
        for key in data_dict:
            if not key == "filename":
                # if exists: sets attribute 'model' too, containing the state dict of the model.
                setattr(self, str(key), data_dict[key])
        
        print("Reading successful")
        self.write_previous_filename()

    def create_new_filename(self, optional=""):
        if optional != "":
            optional = "_" + optional
        self.filename = time.strftime(self.solver_str + " {}d ".format(self.d) + "(%Y-%m-%d)-(%H-%M-%S)") + optional + ".json"  # type: ignore 
   
    def create_input_file(self, model = None, endstate = False):
        """Can be used to create or update the input file according to the contents of __dict__ and model.get_state()"""
 
        d = self.__dict__
        if model != None:
            if endstate:
                d['model_end'] = model.get_state()
            else:
                d['model'] = model.get_state()

        # dumping
        json.dump(
           d, open(os.path.join(INPUTS_DIR, self.filename), "w"), cls = Encoder, indent=4
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
