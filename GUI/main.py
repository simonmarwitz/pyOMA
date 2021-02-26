from matplotlib.widgets import RectangleSelector
from PyQt5 import QtWidgets
import sys
import numpy as np


# imports GUI and PreprocessingTools 
#sys.path.append(os.path.abspath("E:/OneDrive/BHU_NHRE/Python/2017_PreProcessGUI/modal_analysis/"))
import filter_GUI as fGUI
import PreprocessingTools as ppt
import modified_GUI as design


class PreProcessApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(PreProcessApp, self).__init__(parent)
        self.setupUi(self)
        self.openTestFile.clicked.connect(self.openTestFileClick)
        self.channelBox.currentTextChanged.connect(self.updatePlot)
        self.prevChanButton.clicked.connect(self.prevChanButtonClick)
        self.nextChanButton.clicked.connect(self.nextChanButtonClick)
        self.logXCheck.clicked.connect(self.logXCheckClick)
        self.logYCheck.clicked.connect(self.logYCheckClick)
        self.plotLogState = 0
        self.visualizationTypeSelect.currentTextChanged.connect(self.updatePlot)
        self.startTimeEdit.setText("0")
        self.endTimeEdit.setText("0")
        self.startFreqEdit.setText("0")
        self.endFreqEdit.setText("0")
        self.buttonSaveFigure.clicked.connect(self.buttonSaveFigureClick)
        self.startTimeEdit.textChanged.connect(self.startTimeEditChange)
        self.endTimeEdit.textChanged.connect(self.endTimeEditChange)
        self.startFreqEdit.textChanged.connect(self.startFreqEditChange)
        self.endFreqEdit.textChanged.connect(self.endFreqEditChange)
        self.toggle_selectorTime = RectangleSelector(self.canvasTime.axes, self.canvasTimeSelection,
                                                drawtype='box', useblit=True,
                                                button=[1],  # don't use middle button
                                                interactive=False)
        self.toggle_selectorFreq = RectangleSelector(self.canvasFreq.axes, self.canvasFreqSelection,
                                                drawtype='box', useblit=True,
                                                button=[1],  # don't use middle button
                                                interactive=False)
        self.canvasTime.mpl_connect('button_press_event', self.canvasTimeZoomOut)
        self.canvasFreq.mpl_connect('button_press_event', self.canvasFreqZoomOut)
        
        self.prepApply.clicked.connect(self.prepApplyClick)
        
        self.filter_app = QtWidgets.QApplication(sys.argv)
        self.FilterWindow = QtWidgets.QMainWindow()
        self.filterUI = fGUI.Ui_FilterWindow()
        self.filterUI.setupUi(self.FilterWindow)
        self.filterUI.filterButtonCancel.clicked.connect(self.filterButtonCancelClick)
        self.filterUI.filterButtonOK.clicked.connect(self.filterButtonOKClick)
        self.filterUI.filterType.currentIndexChanged.connect(self.filterChagesFunc)
        self.filterUI.filterDesign.currentIndexChanged.connect(self.filterChagesFunc)
    
    def prepApplyClick(self):    
        if self.prepFunc.currentText() == "Filter":
            self.FilterWindow.show()
    
    def filterButtonCancelClick(self):
        self.FilterWindow.hide()
    
    def filterChagesFunc(self):   
        f_type = self.filterUI.filterType.currentIndex()
        f_design = self.filterUI.filterDesign.currentIndex()
        # enable/disable ripple values
        if f_type == 0:
            self.filterUI.filterPassRp.setEnabled(False)
            self.filterUI.filterStopRp.setEnabled(False)
            self.filterUI.filterPassRp.setText("")
            self.filterUI.filterStopRp.setText("")
        elif f_type == 1:
            self.filterUI.filterPassRp.setEnabled(True)
            self.filterUI.filterStopRp.setEnabled(False)
            self.filterUI.filterStopRp.setText("")
        elif f_type == 2:
            self.filterUI.filterPassRp.setEnabled(False)
            self.filterUI.filterStopRp.setEnabled(True)
            self.filterUI.filterPassRp.setText("")
        elif f_type == 3:
            self.filterUI.filterPassRp.setEnabled(True)
            self.filterUI.filterStopRp.setEnabled(True)
        elif f_type == 4:
            self.filterUI.filterPassRp.setEnabled(False)
            self.filterUI.filterStopRp.setEnabled(False)
            self.filterUI.filterPassRp.setText("")
            self.filterUI.filterStopRp.setText("")
        # enable/disable frequency options
        if f_design == 0 or f_design == 1:
            self.filterUI.filterLowFreq.setEnabled(True)
            self.filterUI.filterHighFreq.setEnabled(False)
            self.filterUI.filterHighFreq.setText("")
        elif f_design == 2:
            self.filterUI.filterLowFreq.setEnabled(True)
            self.filterUI.filterHighFreq.setEnabled(True)
        
    
    def filterButtonOKClick(self):
        ftype_name = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
        ftype_name = ftype_name[self.filterUI.filterType.currentIndex()]
        f_order = self.filterUI.filterOrderSlider.value()
        
        try:
            f_lowfreq = abs(int(self.filterUI.filterLowFreq.text()))
            if self.filterUI.filterHighFreq.isEnabled():
                f_highfreq = abs(int(self.filterUI.filterHighFreq.text()))
            else:
                f_highfreq = None
            if self.filterUI.filterPassRp.isEnabled():
                f_passRp = abs(int(self.filterUI.filterPassRp.text()))
            else:
                f_passRp = None
            if self.filterUI.filterStopRp.isEnabled():
                f_stopRp = abs(int(self.filterUI.filterStopRp.text()))
            else:
                f_stopRp = None
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", "Inputs can only be numbers.")
            self.FilterWindow.activateWindow()
        
        RpRsVal = [f_passRp, f_stopRp]
        self.data.filter_data(lowpass=f_lowfreq, highpass=f_highfreq, overwrite=True, order=f_order, ftype=ftype_name,  RpRs = RpRsVal, plot_filter=False)
        self.FilterWindow.hide()
        self.updatePlot()
    
    def openTestFileClick(self):    
        # loads a file, changes string to display name
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self)
        fileName = filePath.split("/")
        self.label_2.setText(fileName[len(fileName)-1])
        
        # start preprocessing class
        measurement = np.loadtxt(filePath)
        self.sampling_rate, _ = QtWidgets.QInputDialog.getDouble(self, "Sampling Rate","Value [Hz]:", 128, 0, 999999, 1)
        self.data = ppt.PreprocessData(measurement, self.sampling_rate)
        
        # insert values for time and frequency in plot parameters
        self.startTimeEdit.setText("0")
        self.endTimeEdit.setText(str((self.data.total_time_steps)/self.sampling_rate))
        self.startFreqEdit.setText("0")
        self.endFreqEdit.setText(str(self.sampling_rate/2))
        
        # populates channel list 
        channel_list = [str(i) for i in range(self.data.num_analised_channels)]
        self.channelBox.clear()
        self.channelBox.addItems(channel_list)
        
        
    def updatePlot(self):
        time_a = int(float(self.startTimeEdit.text())*self.sampling_rate)
        time_z = int(float(self.endTimeEdit.text())*self.sampling_rate)
        
        if self.visualizationTypeSelect.currentText() == "Time, FFT":
            # get data to plot
            time, accel = self.data.get_time_accel(int(self.channelBox.currentText()))
            
            time = time[time_a:time_z]
            accel = accel[time_a:time_z]
            
            fft_freq, fft_sum = self.data.get_fft()
            fft_sum = fft_sum[int(self.channelBox.currentText()), :]
            
            freq_a = int(4*(float(self.startFreqEdit.text())/self.sampling_rate/2)*len(fft_freq))
            freq_z = int(4*(float(self.endFreqEdit.text())/self.sampling_rate/2)*len(fft_freq))
            
            fft_freq = fft_freq[freq_a:freq_z]
            fft_sum = fft_sum[freq_a:freq_z]
            
            # plot time x acceleration
            sbp = self.canvasTime.figure.add_subplot(111)
            sbp.clear()
            if self.plotLogState == 0:
                sbp.plot(time, accel)
            elif self.plotLogState == 1:
                sbp.semilogx(time, accel)
            elif self.plotLogState == 2:
                sbp.semilogy(time, accel)
            else:
                sbp.loglog(time, accel)
            self.canvasTime.draw()
            
            # plot fft
            sbp = self.canvasFreq.figure.add_subplot(111)
            sbp.clear()
            if self.plotLogState == 0:
                sbp.plot(fft_freq, fft_sum)
            elif self.plotLogState == 1:
                sbp.semilogx(fft_freq, fft_sum)
            elif self.plotLogState == 2:
                sbp.semilogy(fft_freq, fft_sum)
            else:
                sbp.loglog(fft_freq, fft_sum)
            self.canvasFreq.draw() 
        
        elif self.visualizationTypeSelect.currentText() == "Time, PSD":
            # get data to plot
            time, accel = self.data.get_time_accel(int(self.channelBox.currentText()))
            time = time[time_a:time_z]
            accel = accel[time_a:time_z]
            
            if self.windLenEdit.text() != "":
                n_lines = int(self.windLenEdit.text())
            else:
                n_lines=256
                
            if self.rectWindSelect.isChecked():
                window='boxcar'
            elif self.hanWindSelect.isChecked():
                window='hann'
            else:
                window='hann'
                
            psd_mats, freqs = self.data.psd_welch(n_lines, False, window)
            
            # plot time x acceleration
            sbp = self.canvasTime.figure.add_subplot(111)
            sbp.clear()
            if self.plotLogState == 0:
                sbp.plot(time, accel)
            elif self.plotLogState == 1:
                sbp.semilogx(time, accel)
            elif self.plotLogState == 2:
                sbp.semilogy(time, accel)
            else:
                sbp.loglog(time, accel)
            self.canvasTime.draw()
            
            # plot PSD
            # has to be evaluated again!!!
            # not functional
            psd_vector = np.abs(psd_mats[int(self.channelBox.currentText()), :])
            psd_vector = np.mean(psd_vector, axis=0)
            
            freq_a = int(4*(float(self.startFreqEdit.text())/self.sampling_rate/2)*len(psd_vector))
            freq_z = int(4*(float(self.endFreqEdit.text())/self.sampling_rate/2)*len(psd_vector))
            
            freqs = freqs[freq_a:freq_z]
            psd_vector = psd_vector[freq_a:freq_z]
            
            sbp = self.canvasFreq.figure.add_subplot(111)
            sbp.clear()
            if self.plotLogState == 0:
                sbp.plot(freqs, psd_vector)
            elif self.plotLogState == 1:
                sbp.semilogx(freqs, psd_vector)
            elif self.plotLogState == 2:
                sbp.semilogy(freqs, psd_vector)
            else:
                sbp.loglog(freqs, psd_vector)
            self.canvasFreq.draw()
            
            
    
    def prevChanButtonClick(self):
        if self.channelBox.currentIndex() != 0:
            self.channelBox.setCurrentIndex(self.channelBox.currentIndex()-1)
        
    def nextChanButtonClick(self):
        items = [self.channelBox.itemText(i) for i in range(self.channelBox.count())]
        items = len(items) - 1
        if self.channelBox.currentIndex() != items:
            self.channelBox.setCurrentIndex(self.channelBox.currentIndex()+1)
            
    def logXCheckClick(self):
        if self.logXCheck.checkState() == 2:
            if self.logYCheck.checkState() == 2:
                self.plotLogState = 3 # plots log x log               
            else:
                self.plotLogState = 1 # log only on x axis
        else:
            if self.logYCheck.checkState() == 2:
                self.plotLogState = 2 # log only on y axis
            else:
                self.plotLogState = 0 # normal plot
        self.updatePlot()
                
    def logYCheckClick(self):
        if self.logXCheck.checkState() == 2:
            if self.logYCheck.checkState() == 2:
                self.plotLogState = 3 # plots log x log
            else:
                self.plotLogState = 1 # log only on x axis
        else:
            if self.logYCheck.checkState() == 2:
                self.plotLogState = 2 # log only on y axis
            else:
                self.plotLogState = 0 # normal plot
        self.updatePlot()
        
    def buttonSaveFigureClick(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save time signal figure")
        fig = self.canvasTime.figure
        fig.savefig(fileName, dpi=300)
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save frequency sprectrum figure")
        fig = self.canvasFreq.figure
        fig.savefig(fileName, dpi=300)
        
    def toggle_selectorTime(self, event):
        pass
    
    def toggle_selectorFreq(self, event):
        pass
        
    def canvasTimeSelection(self, eclick, erelease):
        x_ini = round(eclick.xdata, 2)
        x_end = round(erelease.xdata,2)
        x_min = min([x_ini, x_end])
        x_max = max([x_ini, x_end])
        self.startTimeEdit.setText(str(x_min))
        self.endTimeEdit.setText(str(x_max))
#        self.updatePlot()
#        self.canvasTime.axes.relim()
    
    def canvasFreqSelection(self, eclick, erelease):
        x_ini = round(eclick.xdata, 2)
        x_end = round(erelease.xdata,2)
        x_min = min([x_ini, x_end])
        x_max = max([x_ini, x_end])
        self.startFreqEdit.setText(str(x_min))
        self.endFreqEdit.setText(str(x_max))
#        self.updatePlot()
#        self.canvasFreq.axes.relim()
        
    def canvasTimeZoomOut(self, event):
        if event.button == 3:
            self.startTimeEdit.setText(str(0))
            self.endTimeEdit.setText(str(round((self.data.total_time_steps-1)/self.sampling_rate, 2)))        
    
    def canvasFreqZoomOut(self, event):
        if event.button == 3:
            self.startFreqEdit.setText("0")
            self.endFreqEdit.setText(str(round(self.sampling_rate/2, 1)))        
        
    def startTimeEditChange(self):
        time_a = int(float(self.startTimeEdit.text())*self.sampling_rate)
        if time_a < 0:
            self.startTimeEdit.setText("0")
        self.updatePlot()
        
    def endTimeEditChange(self):
        time_z = int(float(self.endTimeEdit.text())*self.sampling_rate)
        if time_z > (self.data.total_time_steps-1):
            self.endTimeEdit.setText(str(round((self.data.total_time_steps-1)/self.sampling_rate, 2)))
        self.updatePlot()
        
    def startFreqEditChange(self):
        freq_a = int(float(self.startFreqEdit.text()))
        if freq_a < 0:
            self.startFreqEdit.setText("0")
        self.updatePlot()
        
    def endFreqEditChange(self):
        freq_z = int(float(self.endFreqEdit.text()))
        if freq_z > self.sampling_rate/2:
            self.endFreqEdit.setText(str(round(self.sampling_rate/2, 1)))
        self.updatePlot()
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    form = PreProcessApp()
    form.show() 
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
    
