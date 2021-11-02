from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc

_translate = qtc.QCoreApplication.translate

import sys

if __name__ == "__main__":
    import os.path

    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from preprocessing.GUIbase import Ui_Form

from sampling.solvers.solver import get_solver_name_list, get_solver
from sampling.initial_sampling import get_doe_name_list
from models.kriging.kernel import get_available_kernel_names


class GUI(qtw.QWidget, Ui_Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.set_solver_combobox()
        self.set_kernel_combobox()
        self.set_doe_combobox()

        self.dimension_spinbox_changed()
        self.solver_changed()

        self.dimensionSpinBox.valueChanged.connect(self.dimension_spinbox_changed)

        self.selectSolverComboBox.currentTextChanged.connect(self.solver_changed)

        self.createSetupFileButton.clicked.connect(self.set_gui_dict)

        self.gui_data = {}

    def add_parameter_range(self, par_nr):
        """
        Adds a field in the gui for defining the parameter range.
        Indexing of ParameterRangeHorizontalLayout_par{} starts at 1 corresponding to dimension d.
        """
        self.ParameterRangeHorizontalLayout = qtw.QHBoxLayout()
        self.ParameterRangeHorizontalLayout.setObjectName(
            "ParameterRangeHorizontalLayout_par{}".format(par_nr)
        )

        self.RangeName = qtw.QLineEdit(self.gridLayoutWidget)
        self.RangeName.setObjectName("RangeName_par{}".format(par_nr))
        self.ParameterRangeHorizontalLayout.addWidget(self.RangeName)

        self.line_1 = qtw.QFrame(self.gridLayoutWidget)
        self.line_1.setFrameShape(qtw.QFrame.VLine)
        self.line_1.setFrameShadow(qtw.QFrame.Sunken)
        self.line_1.setObjectName("line_1_par{}".format(par_nr))
        self.ParameterRangeHorizontalLayout.addWidget(self.line_1)

        self.RangeLowerbound = qtw.QLineEdit(self.gridLayoutWidget)
        self.RangeLowerbound.setObjectName("RangeLowerbound_par{}".format(par_nr))
        self.ParameterRangeHorizontalLayout.addWidget(self.RangeLowerbound)

        self.line_2 = qtw.QFrame(self.gridLayoutWidget)
        self.line_2.setFrameShape(qtw.QFrame.VLine)
        self.line_2.setFrameShadow(qtw.QFrame.Sunken)
        self.line_2.setObjectName("line_2_par{}".format(par_nr))
        self.ParameterRangeHorizontalLayout.addWidget(self.line_2)

        self.RangeUpperbound = qtw.QLineEdit(self.gridLayoutWidget)
        self.RangeUpperbound.setObjectName("RangeUpperbound_par{}".format(par_nr))
        self.ParameterRangeHorizontalLayout.addWidget(self.RangeUpperbound)

        self.ParameterRangeVerticalLayout.addLayout(self.ParameterRangeHorizontalLayout)

        _translate = qtc.QCoreApplication.translate
        self.RangeName.setText(_translate("Form", "x{}".format(par_nr)))
        self.RangeLowerbound.setText(_translate("Form", "-3"))
        self.RangeUpperbound.setText(_translate("Form", "3"))

    def remove_parameter_range(self, par_nr):
        layout = self.ParameterRangeVerticalLayout.findChild(
            qtw.QHBoxLayout, "ParameterRangeHorizontalLayout_par{}".format(par_nr)
        )
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        layout.deleteLater()

    def dimension_spinbox_changed(self):
        count = self.ParameterRangeVerticalLayout.count() - 1
        for i in range(count, self.dimensionSpinBox.value()):
            self.add_parameter_range(i)
        for i in range(self.dimensionSpinBox.value(), count):
            self.remove_parameter_range(i)

        solver_string = self.get_solver_str()
        solver = get_solver(name=solver_string)
        self.set_search_space(
            solver.get_preferred_search_space(self.dimensionSpinBox.value())
        )

    def set_gui_dict(self):
        """
        Retrieve the data from the GUI form and return it in a dictionary.
        """
        gui_data = self.gui_data

        # SOLVER & OBJECTIVE
        gui_data["solver_str"] = self.get_solver_str()
        gui_data["d"] = int(self.dimensionSpinBox.value())

        # search_space structured as data matrix X: [[d0_name,d1_name,d2_name,...],[lb0,lb1,...],[ub0,ub1,...]]
        search_space = [[], [], []]

        for i in range(gui_data["d"]):
            horlay = self.ParameterRangeVerticalLayout.findChild(
                qtw.QHBoxLayout, "ParameterRangeHorizontalLayout_par{}".format(i)
            )
            RangeName = str(horlay.itemAt(0).widget().text())
            RangeLowerbound = float(horlay.itemAt(2).widget().text())
            RangeUpperbound = float(horlay.itemAt(4).widget().text())
            search_space[0].append(RangeName)
            search_space[1].append(RangeLowerbound)
            search_space[2].append(RangeUpperbound)
        gui_data["search_space"] = search_space

        # TODO hard
        # objectiveFunctionHorizontalLayout
        #     objectiveFunctionLineEdit
        #     objectiveFunctionMinMaxComboBox()
        # https://stackoverflow.com/questions/11112046/create-a-lambda-function-from-a-string-properly
        # gui_data['objective'] = lambda x: objectiveFunctionLineEdit value to non-string

        # # KRIGING OPTIONS
        # selectKrigingTypeComboBox
        gui_data["kriging_type"] = str(self.selectKrigingTypeComboBox.currentText())

        gui_data["kernel"] = str(self.correlationKernelComboBox.currentText())

        # multifidelityCheckBox
        gui_data["multifidelity"] = self.multifidelityCheckBox.isChecked()

        # parallelSamplingCheckBox
        gui_data["parallel"] = self.parallelSamplingCheckBox.isChecked()

        # krigingSearchAlgorithmComboBox
        gui_data["kriging_search_algorithm"] = str(
            self.krigingSearchAlgorithmComboBox.currentText()
        )

        # stoppingCriterionComboBox
        gui_data["stopping_criterion"] = str(
            self.stoppingCriterionComboBox.currentText()
        )

        # stoppingCriterionValueLineEdit
        gui_data["stopping_criterion_value"] = str(
            self.stoppingCriterionValueLineEdit.text()
        )

        # infillCriterionComboBox
        gui_data["infill_criterion"] = str(self.infillCriterionComboBox.currentText())

        # infillSearchAlgorithmComboBox
        gui_data["infill_search_algorithm"] = str(
            self.infillSearchAlgorithmComboBox.currentText()
        )

        # hyperparameterSearchAlgorithmComboBox
        gui_data["hyperparameter_search_algorithm"] = str(
            self.hyperparameterSearchAlgorithmComboBox.currentText()
        )

        # doEComboBox
        gui_data["DoE"] = str(self.doEComboBox.currentText())

        # # OUTPUT OPTIONS
        gui_data["SAVE_DATA"] = self.saveSampleDataCheckBox.isChecked()

        # livePlotCheckBox
        gui_data["live_plot"] = self.livePlotCheckBox.isChecked()

        # dimensionsToPlotLineEdit
        gui_data["d_plot"] = self.parse_dimensions_to_plot(
            str(self.dimensionsToPlotLineEdit.text()), search_space
        )

        # savePlotCheckBox
        gui_data["save_plot"] = self.savePlotCheckBox.isChecked()

        if gui_data["d_plot"] != False:
            self.close()

    def parse_dimensions_to_plot(self, text, search_space):
        pars = [s.strip() for s in text.split(",")]
        if pars == [""]:
            # then return the first dimensions for plotting
            return [i for i in range(min(2, len(search_space[0])))]

        res = []
        for i in pars:
            if i.isnumeric():
                i = int(i)
                if i < len(search_space[0]) and i >= 0:
                    res.append(i)
                else:
                    raise Exception(
                        "Plotting parameter does not exist, check if number is in dimensional range."
                    )
                    return False
            else:
                try:
                    res.append(search_space[0].index(i))
                except ValueError:
                    print("Plotting parameter does not exist, check for typos.")
                    return False
        return res

    def get_solver_str(self):
        return (
            str(self.selectSolverComboBox.currentText())
            .replace("internal: ", "")
            .replace("external: ", "")
        )

    def get_gui_dict(self):
        if self.gui_data is not None:
            return self.gui_data
        else:
            raise Exception

    def set_gui_form(self, data_dict):
        """this function sets the gui data depending on a previously defined file"""

        # doing this retains our sample data as well
        self.gui_data = data_dict
        # TODO

    def get_optional_filename(self):
        """function for returning the additional optional filename identifier"""
        return self.optionalFilenameLineEdit.text()

    def set_kernel_combobox(self):
        """This sets the dropdownlist for selecting the Kriging kernel"""
        self.correlationKernelComboBox.clear()
        self.correlationKernelComboBox.addItems(get_available_kernel_names())

    def set_doe_combobox(self):
        """This sets the dropdownlist for selecting the Kriging kernel"""
        self.doEComboBox.clear()
        self.doEComboBox.addItems(get_doe_name_list())

    def set_solver_combobox(self):
        """This sets the dropdownlist for selecting the solver"""
        self.selectSolverComboBox.clear()
        self.selectSolverComboBox.addItems(get_solver_name_list())

    def solver_changed(self):
        solver_string = self.get_solver_str()
        solver = get_solver(name=solver_string)

        self.dimensionSpinBox.setMaximum(solver.max_d)
        self.dimensionSpinBox.setMinimum(solver.min_d)
        self.dimensionRangeLabel.setText(
            _translate("Form", "[{}, {}] :".format(solver.min_d, solver.max_d))
        )

        # has to be behind any dimension spinbox functions
        self.set_search_space(
            solver.get_preferred_search_space(self.dimensionSpinBox.value())
        )

        self.set_input_parameter_list_widget(solver)
        self.set_output_parameter_list_widget(solver)

    def set_search_space(self, search_space):
        if search_space is not None:
            for i in range(self.dimensionSpinBox.value()):
                horlay = self.ParameterRangeVerticalLayout.findChild(
                    qtw.QHBoxLayout, "ParameterRangeHorizontalLayout_par{}".format(i)
                )
                horlay.itemAt(0).widget().setText(
                    _translate("Form", str(search_space[0][i]))
                )
                horlay.itemAt(2).widget().setText(
                    _translate("Form", str(search_space[1][i]))
                )
                horlay.itemAt(4).widget().setText(
                    _translate("Form", str(search_space[2][i]))
                )
        else:
            # reset parameter range gui
            for i in range(self.dimensionSpinBox.value()):
                self.remove_parameter_range(i)
            for i in range(self.dimensionSpinBox.value()):
                self.add_parameter_range(i)

    def set_input_parameter_list_widget(self, solver):
        # solverInputParameterListWidget
        self.solverInputParameterListWidget.clear()
        self.solverInputParameterListWidget.addItems(solver.input_parameter_list)

    def set_output_parameter_list_widget(self, solver):
        # solverOutputParameterListWidget
        self.solverOutputParameterListWidget.clear()
        self.solverOutputParameterListWidget.addItems(solver.output_parameter_list)


if __name__ == "__main__":

    app = qtw.QApplication([])

    widget = GUI()
    widget.show()
    app.exec_()
