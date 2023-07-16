import mainwindow
import generator
import filter

import sys
import os
import math
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem, QWidget
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication



class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = mainwindow.Ui_Form()
        self.ui.setupUi(self)
    def open_generator_window(self):
        generator_window.show()

    def open_filter_window(self):
        filtering_window.show()

class GeneratorWindow(QMainWindow):
    def __init__(self, parent=None):
        super(GeneratorWindow, self).__init__(parent)
        self.ui = generator.Ui_MainWindow()
        self.ui.setupUi(self)

    def open_mzCCS_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(
            self, "choose file", os.getcwd(),
            "Excel Files(*.xlsx *.xls *.csv)")
        try:
            if fileName == '' and self.original_mzCCS_path == '':
                QMessageBox.warning(self, "Warning", 'Please import  compound file.')
            elif fileName != '':
                self.ui.mzccs_path.setText(fileName)
                self.original_mzCCS_path = fileName
        except AttributeError:
            QMessageBox.warning(self, "Warning", 'Please import correct  compound file.')

    def linear_model(self,x,a, b):
        return a * x + b
    def lm(self, data):
        X = data["m/z"]
        Y = data["CCS"]
        p_fit, prov = curve_fit(self.linear_model, X, Y)
        coef = p_fit[0]
        inter = p_fit[1]
        Y_hat = [self.linear_model(i, p_fit[0], p_fit[1]) for i in X]
        SSE = np.sum((Y - Y_hat) ** 2)
        SST = np.sum((Y - np.mean(Y)) ** 2)
        R2 = round(1 - (SSE / SST), 4)
        reg_equ = "y=" + "%.4f" % coef + "*x+" + "%.4f" % inter
        return coef, inter, reg_equ, R2

    def power_func(self,x, a, b):
        return np.log(a) + b * np.log(x)
    def pf(self,data):
        X = data["m/z"]
        Y = data["CCS"]
        LNY = np.log(data["CCS"])
        p_fit, prov = curve_fit(self.power_func, X, LNY)
        m = p_fit[0]
        n = p_fit[1]
        LNY_hat = [self.power_func(i, p_fit[0], p_fit[1]) for i in X]
        SSE = np.sum((LNY - LNY_hat) ** 2)
        SST = np.sum((LNY - np.mean(LNY)) ** 2)
        R2 = round(1 - (SSE / SST), 4)
        reg_equ = "y=" + "%.4f" % m + "x^" + "%.4f" % n
        return m, n, reg_equ, R2

    def choose_model(self,data):
        max_R2 = max(self.lm(data)[3], self.pf(data)[3])
        if self.lm(data)[3] == max_R2:
            model_name = 'linear_model'
        elif self.pf(data)[3] == max_R2:
            model_name = 'power_function'
        return model_name

    def generate_equ(self,data):
        model = ['linear model','power func']
        equations = [self.lm(data)[2], self.pf(data)[2]]
        R2 = [self.lm(data)[3], self.pf(data)[3]]
        Best = [self.choose_model(data),'']
        data_generation = pd.DataFrame([model,equations, R2, Best]).T
        data_generation.columns = ['Model','Equation', 'R^2', 'Best']
        return data_generation

    def ex_fitting_equation(self):
        try:
            if self.original_mzCCS_path.split('.')[-1] == 'xlsx' or self.original_mzCCS_path.split('.')[-1] == 'xls':
                mzCCSdata = pd.read_excel(self.original_mzCCS_path)
            elif self.original_mzCCS_path.split('.')[-1] == 'csv':
                mzCCSdata = pd.read_csv(self.original_mzCCS_path)

            generated_data = self.generate_equ(mzCCSdata)

            if generated_data.shape[0] > 0:
                self.ui.tableWidget.setRowCount(generated_data.shape[0])
                for i in range(generated_data.shape[0]):
                    for j in range(generated_data.shape[1]):
                        item = QTableWidgetItem(str(generated_data.iloc[i, j]))
                        self.ui.tableWidget.setItem(i, j, item)
                        j += 1
                    i += 1
            else:
                self.ui.tableWidget.setRowCount(0)
                QMessageBox.information(self, "Message", 'Not Matched.')

        except:
            QMessageBox.warning(self, "Warning", "Please import correct compound file.")


class FilteringWindow(QMainWindow):
    def __init__(self, parent=None):
        super(FilteringWindow, self).__init__(parent)
        self.ui = filter.Ui_MainWindow()
        self.ui.setupUi(self)

    def open_mzCCS_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(
            self, "choose file", os.getcwd(),
            "Excel Files(*.xlsx *.xls *.csv)")
        try:
            if fileName == '' and self.original_mzCCS_path == '':
                QMessageBox.warning(self, "Warning", 'Please import correct compound file.')
            elif fileName != '':
                self.ui.mzccs_path.setText(fileName)
                self.original_mzCCS_path = fileName
        except AttributeError:
            QMessageBox.warning(self, "Warning", 'Please import correct compound file.')

    def open_ex_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(
            self, "choose file", os.getcwd(),
            "Excel Files(*.xlsx *.xls *.csv)")
        try:
            if fileName == '' and self.ex_data_path == '':
                QMessageBox.warning(self, "Warning", 'Please import correct experimental data.')
            elif fileName != '':
                self.ui.ex_data_file_path.setText(fileName)
                self.ex_data_path = fileName
        except AttributeError:
            QMessageBox.warning(self, "Warning", 'Please import correct experimental data.')

    def open_result_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getSaveFileName(
            self, "save file", os.getcwd(),
            "Excel Files(*.xlsx *.xls *.csv)")
        try:
            if fileName == '' and self.result_file_path == '':
                QMessageBox.warning(self, "Warning", 'Please choose a correct result path.')
            elif fileName != '':
                self.ui.filter_result_path.setText(fileName)
                self.result_file_path = fileName
        except AttributeError:
            QMessageBox.warning(self, "Warning", 'Please choose a correct result path.')

    def linear_model(self,x,a, b):
        return a * x + b
    def lm(self, data):
        X = data["m/z"]
        Y = data["CCS"]
        p_fit, prov = curve_fit(self.linear_model, X, Y)
        coef = p_fit[0]
        inter = p_fit[1]
        Y_hat = [self.linear_model(i, p_fit[0], p_fit[1]) for i in X]
        SSE = np.sum((Y - Y_hat) ** 2)
        MSE = round(np.sum((Y - Y_hat) ** 2) / len(Y), 4)
        SST = np.sum((Y - np.mean(Y)) ** 2)
        R2 = round(1 - (SSE / SST), 4)
        reserr = Y - Y_hat
        std_res = np.std(reserr)
        confidence_interval = 2.58 * std_res / math.sqrt(len(data))
        prediction_interval = 2.58 * std_res
        reg_equ = "y=" + "%.4f" % coef + "*x+" + "%.4f" % inter
        return coef, inter, reg_equ, R2, MSE, confidence_interval, prediction_interval

    def power_func(self, x, a, b):
        return np.log(a) + b * np.log(x)
    def pf(self, data):
        X = data["m/z"]
        Y = data["CCS"]
        LNY = np.log(Y)
        p_fit, prov = curve_fit(self.power_func, X, LNY)
        m = p_fit[0]
        n = p_fit[1]
        LNY_hat = [self.power_func(i, p_fit[0], p_fit[1]) for i in X]
        Y_hat = np.exp(LNY_hat)
        SSE = np.sum((LNY - LNY_hat) ** 2)
        MSE = round(np.sum((LNY - LNY_hat) ** 2) / len(Y), 4)
        SST = np.sum((LNY - np.mean(LNY)) ** 2)
        R2 = round(1 - (SSE / SST), 4)
        reserr = np.log(Y / Y_hat)
        std_res = np.std(reserr)
        confidence_interval = math.exp(2.58 * std_res / math.sqrt(len(data)))
        prediction_interval = math.exp(2.58 * std_res)
        reg_equ = "y=" + "%.4f" % m + "x^" + "%.4f" % n
        return m, n, reg_equ, R2, MSE, confidence_interval, prediction_interval

    def choose_model(self,data):
        max_R2 = max(self.lm(data)[3], self.pf(data)[3])
        if self.lm(data)[3] == max_R2:
            model_name = 'linear_model'
        elif self.pf(data)[3] == max_R2:
            model_name = 'power_function'
        return model_name

    def filter_successful_generate(self):
        QMessageBox.information(self, "Message", "Filtered Successfully.")

    def filt_process(self, data):
        try:
            if self.original_mzCCS_path.split('.')[-1] == 'xlsx' or self.original_mzCCS_path.split('.')[-1] == 'xls':
                mzCCSdata = pd.read_excel(self.original_mzCCS_path)
            elif self.original_mzCCS_path.split('.')[-1] == 'csv':
                mzCCSdata = pd.read_csv(self.original_mzCCS_path)

            filtdata = data.copy()
            filtdata['Predictive statistics_99%'] = 0
            model_name = self.choose_model(mzCCSdata)

            if model_name == 'linear_model':
                equ = self.lm(mzCCSdata)[2]
                R2 = self.lm(mzCCSdata)[3]
                for i in filtdata.index:
                    CCS = filtdata.loc[i, 'CCS']
                    exmz = filtdata.loc[i, 'm/z']
                    CCShat = self.lm(mzCCSdata)[0] * exmz + self.lm(mzCCSdata)[1]
                    if (CCShat - self.lm(mzCCSdata)[5]) <= CCS <= (CCShat + self.lm(mzCCSdata)[5]):
                        filtdata.loc[i, 'Predictive statistics_99%'] = 'In 0.99 CI'
                    elif CCS <= (CCShat - self.lm(mzCCSdata)[6]) or CCS >= (CCShat + self.lm(mzCCSdata)[6]):
                        filtdata.loc[i, 'Predictive statistics_99%'] = 'Out of 0.99 PI'
                    else:
                        filtdata.loc[i, 'Predictive statistics_99%'] = 'In 0.99 PI but Out of 0.99 CI'
                    i += 1
                    self.ui.textEdit_2.setText('Please waiting...{}%'.format(round(i / len(filtdata) * 100, 1)))
                    QApplication.processEvents()

            elif model_name == 'power_function':
                equ = self.pf(mzCCSdata)[2]
                R2 = self.pf(mzCCSdata)[3]
                for i in filtdata.index:
                    CCS = filtdata.loc[i, 'CCS']
                    exmz = filtdata.loc[i, 'm/z']
                    CCShat = self.pf(mzCCSdata)[0] * exmz ** self.pf(mzCCSdata)[1]
                    if (CCShat / self.pf(mzCCSdata)[5]) <= CCS <= (CCShat * self.pf(mzCCSdata)[5]):
                        filtdata.loc[i, 'Predictive statistics_99%'] = 'In 0.99 CI'
                    elif (CCS >= CCShat * self.pf(mzCCSdata)[6]) or (CCS <= CCShat / self.pf(mzCCSdata)[6]):
                        filtdata.loc[i, 'Predictive statistics_99%'] = 'Out of 0.99 PI'
                    else:
                        filtdata.loc[i, 'Predictive statistics_99%'] = 'In 0.99 PI but Out of 0.99 CI'
                    i += 1
                    self.ui.textEdit_2.setText('Please waiting...{}%'.format(round(i / len(filtdata) * 100, 1)))
                    QApplication.processEvents()

            d = dict()
            d['filtdata'] = filtdata
            d['modelname'] = model_name
            d['equ'] = equ
            d['R2'] = R2
            return d

        except:
            QMessageBox.warning(self, "Warning", "Please import correct compound file.")

    def ex_filt_result(self):
        try:
            if self.ex_data_path.split('.')[-1] == 'xlsx' or self.ex_data_path.split('.')[-1] == 'xls':
                exmzCCSdata = pd.read_excel(self.ex_data_path)
            elif self.ex_data_path.split('.')[-1] == 'csv':
                exmzCCSdata = pd.read_csv(self.ex_data_path)

            list = self.filt_process(exmzCCSdata)
            filtdata = list['filtdata']
            modelname = list['modelname']
            equ = list['equ']
            R2 = list['R2']

            num = filtdata.iloc[:, 0].size
            In99CI_num = (filtdata['Predictive statistics_99%'] == 'In 0.99 CI').sum()
            Out99PI_num = (filtdata['Predictive statistics_99%'] == 'Out of 0.99 PI').sum()
            In99PI_Out99CI_num = (filtdata['Predictive statistics_99%'] == 'In 0.99 PI but Out of 0.99 CI').sum()

            self.ui.filter_tips.setText('{}'.format(num) + " experimental data were found." + '\n'
                                        + '{}'.format(In99CI_num) + " of " + '{}'.format(num) + " data points in 0.99CI." + '\n'
                                        + '{}'.format(In99PI_Out99CI_num) + " of " + '{}'.format(num) + " data points in 0.99 PI but out of 0.99CI." + '\n'
                                        + '{}'.format(Out99PI_num) + " of " + '{}'.format(num) + " data points out of 0.99PI." + '\n'
                                        +"Regression model of the best IM-MS trendline:" + '\n' + '{}'.format(modelname) + '\n'
                                        +"Equation(m/z, CCS)→(x, y): " + '{}'.format(equ) + '\n'
                                        + "R^2 = " + "%.4f" % R2 + '\n'
                                        )

            try:
                if self.result_file_path.split('.')[-1] == 'xlsx' or self.result_file_path.split('.')[-1] == 'xls':
                    filtdata.to_excel(self.result_file_path)
                elif self.result_file_path.split('.')[-1] == 'csv':
                    filtdata.to_csv(self.result_file_path)

                self.filter_successful_generate()
                self.ui.mzccs_path.setText("")
                self.ui.ex_data_file_path.setText("")
                self.ui.filter_result_path.setText("")

            except:
                QMessageBox.warning(self, "Warning", "Please choose a correct save path.")
        except:
            QMessageBox.warning(self, "Warning", "Please import correct compound file.")


if __name__ == '__main__':
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    generator_window = GeneratorWindow()
    filtering_window = FilteringWindow()
    sys.exit(app.exec_())



