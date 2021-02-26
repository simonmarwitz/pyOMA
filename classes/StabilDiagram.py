# -*- coding: utf-8 -*-
'''
Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''

import numpy as np

import scipy.cluster
import scipy.spatial
import scipy.stats

import sys

# make qt application not crash on errors


def my_excepthook(type, value, tback):
    # log the exception here

    # then call the default handler
    sys.__excepthook__(type, value, tback)

sys.excepthook = my_excepthook

import os

import collections
from operator import itemgetter
from random import shuffle

import matplotlib
# check if python is running in headless mode i.e. as a server script
if 'DISPLAY' in os.environ:
    matplotlib.use("Qt5Agg", force=True)
from matplotlib import rcParams, patches
from matplotlib import ticker

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, FigureCanvasQT
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.text import TextPath, FontProperties
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import matplotlib.cm
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plot

plot.rc('figure', figsize=[8.5039399474194, 5.255723925793184], dpi=100,)
plot.rc('font', size=10)
plot.rc('legend', fontsize=10, labelspacing=0.1)
plot.rc('axes', linewidth=0.2)
plot.rc('xtick.major', width=0.2)
plot.rc('ytick.major', width=0.2)
plot.ioff()


from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QPushButton,\
    QCheckBox, QButtonGroup, QLabel, QComboBox, \
    QTextEdit, QGridLayout, QFrame, QVBoxLayout, QAction,\
    QFileDialog,  QMessageBox, QApplication, QRadioButton,\
    QLineEdit, QSizePolicy, QDoubleSpinBox
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot,  QObject, qInstallMessageHandler, QTimer, QEventLoop

NoneType = type(None)

try:
    from SSICovRef import BRSSICovRef, PogerSSICovRef
except ImportError:
    BRSSICovRef = NoneType
    PogerSSICovRef = NoneType
try:    
    from SSIData import SSIData, SSIDataMC
except ImportError:
    SSIData = NoneType
    SSIDataMC = NoneType
    
try:
    from VarSSIRef import VarSSIRef
except ImportError:
    VarSSIRef = NoneType
    
try:
    from PRCE import PRCE
except ImportError:
    PRCE = NoneType

try:
    from PLSCF import PLSCF
except ImportError:
    PLSCF = NoneType
    
try:
    from ERA import ERA
except ImportError:
    ERA = NoneType
    
from PreprocessingTools import PreprocessData



import warnings
import logging
logger = logging.getLogger('')

def resizeEvent_(self, event):
    w = event.size().width()
    h = event.size().height()
    dpival = self.figure.dpi
    winch, hinch = self.figure.get_size_inches()
    aspect = winch / hinch
    if w / h <= aspect:
        h = w / aspect
    else:
        w = h * aspect
    winch = w / dpival
    hinch = h / dpival
    self.figure.set_size_inches(winch, hinch)
    FigureCanvasBase.resize_event(self)
    self.draw()
    self.update()
    QWidget.resizeEvent(self, event)
    
'''
TODO:
scale markers right on every platform
frequency range as argument or from ssi params, sampling freq
add switch to choose between "unstable only in ..." or "stable in ..."
(select and merge several poles with a rectangular mouse selection)
distinguish beetween stabilization criteria and filtering criteria
add zoom and sliders (horizontal/vertical) for the main figure
distinguish between  "export results" and "save state"
'''


class StabilGUI(QMainWindow):

    def __init__(self, stabil_plot, cmpl_plot, msh_plot=None):

        QMainWindow.__init__(self)
        self.setWindowTitle('Stabilization Diagram: {} - {}'.format(
            stabil_plot.stabil_calc.setup_name, stabil_plot.stabil_calc.start_time))

        self.stabil_plot = stabil_plot
        self.stabil_calc = stabil_plot.stabil_calc
        if self.stabil_calc.state < 2:
            self.stabil_calc.calculate_stabilization_masks()

        self.create_menu()
        self.create_main_frame(cmpl_plot, msh_plot)

        self.histo_plot_f = None
        self.histo_plot_sf = None
        self.histo_plot_d = None
        self.histo_plot_sd = None
        self.histo_plot_dr = None
        self.histo_plot_mac = None
        self.histo_plot_mpc = None
        self.histo_plot_mpd = None

        for mode in self.stabil_calc.select_modes:
            self.mode_selector_add(mode)

        self.setGeometry(0, 0, 1800, 1000)
        
        self.show()

        for widg in [self.mode_val_view, self.current_value_view]:
            widg.setText('\n \n \n \n \n \n \n')
            height = widg.document().size().toSize().height() + 3
            widg.setFixedHeight(height)
        #self.plot_selector_msh.setChecked(True)
        self.update_stabil_view()

    def create_main_frame(self, cmpl_plot, msh_plot):
        '''
        set up all the widgets and other elements to draw the GUI
        '''
        main_frame = QWidget()

        df_max = self.stabil_calc.df_max * 100
        dd_max = self.stabil_calc.dd_max * 100
        dmac_max = self.stabil_calc.dmac_max * 100
        d_range = self.stabil_calc.d_range
        mpc_min = self.stabil_calc.mpc_min
        mpd_max = self.stabil_calc.mpd_max

        self.fig = self.stabil_plot.fig
        
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.canvas.setParent(main_frame)
        if self.stabil_plot.cursor is None:
            self.stabil_plot.init_cursor()
        self.stabil_plot.cursor.show_current_info.connect(
            self.update_value_view)
        self.stabil_plot.cursor.mode_selected.connect(self.mode_selector_add)
        self.stabil_plot.cursor.mode_deselected.connect(
            self.mode_selector_take)
        main_layout = QHBoxLayout()

        left_pane_layout = QVBoxLayout()
        left_pane_layout.addStretch(1)
        palette = QPalette()
        palette.setColor(QPalette.Base, Qt.transparent)

        self.current_value_view = QTextEdit()
        self.current_value_view.setFrameShape(QFrame.Box)
        self.current_value_view.setPalette(palette)

        self.diag_val_widget = QWidget()

        fra_1 = QFrame()
        fra_1.setFrameShape(QFrame.Panel)
        fra_1.setLayout(self.create_stab_val_widget(df_max=df_max,
                                                    dd_max=dd_max, d_mac=dmac_max, d_range=d_range, mpc_min=mpc_min,
                                                    mpd_max=mpd_max))
        left_pane_layout.addWidget(fra_1)

        left_pane_layout.addStretch(2)

        fra_2 = QFrame()
        fra_2.setFrameShape(QFrame.Panel)
        fra_2.setLayout(self.create_diag_val_widget())
        left_pane_layout.addWidget(fra_2)

        left_pane_layout.addStretch(2)
        left_pane_layout.addWidget(self.current_value_view)
        left_pane_layout.addStretch(1)

        #right_pane_layout = QVBoxLayout()

        self.plot_selector_c = QCheckBox('Mode Shape in Complex Plane')
        self.plot_selector_c.toggled.connect(self.toggle_cpl_plot)
        self.plot_selector_msh = QCheckBox('Mode Shape in Spatial Model')
        self.plot_selector_msh.toggled.connect(self.toggle_msh_plot)

#         self.group = QButtonGroup()
#         self.group.addButton(self.plot_selector_c)
#         self.group.addButton(self.plot_selector_msh)

        self.mode_selector = QComboBox()
        self.mode_selector.currentIndexChanged[
            int].connect(self.update_mode_val_view)

        self.mode_plot_widget = QWidget()
        self.cmplx_plot_widget = QWidget()

        self.cmpl_plot = cmpl_plot
        #fig = self.cmpl_plot.fig
        #canvas1 = FigureCanvasQTAgg(fig)
        #canvas1.setParent(self.cmplx_plot_widget)

        self.msh_plot = msh_plot
#         if msh_plot is not None:
#             fig = msh_plot.fig
#             #FigureCanvasQTAgg.resizeEvent = resizeEvent_
#             #canvas2 = fig.canvas.switch_backends(FigureCanvasQTAgg)
#             #canvas2.resizeEvent = types.MethodType(resizeEvent_, canvas2)
#             canvas2 = fig.canvas
#             #self.canvas.resize_event = resizeEvent_
#             #self.canvas.resize_event  = funcType(resizeEvent_, self.canvas, FigureCanvasQTAgg)
#             #msh_plot.canvas = canvas2
#             #self.canvas.mpl_connect('button_release_event', self.update_lims)
#             #if fig.get_axes():
#             #    fig.get_axes()[0].mouse_init()
#             canvas2.setParent(self.mode_plot_widget)
#         else:
#             canvas2 = QWidget()

        lay = QHBoxLayout()
        #lay.addWidget(canvas1)
        self.cmplx_plot_widget.setLayout(lay)
        self.cmpl_plot.plot_diagram()

        lay = QHBoxLayout()
        #lay.addWidget(canvas2)
        self.mode_plot_widget.setLayout(lay)

        self.mode_val_view = QTextEdit()
        self.mode_val_view.setFrameShape(QFrame.Box)

        self.mode_val_view.setPalette(palette)
        left_pane_layout.addStretch(1)
        left_pane_layout.addWidget(self.mode_selector)

        left_pane_layout.addWidget(self.plot_selector_c)
        left_pane_layout.addWidget(self.plot_selector_msh)
        left_pane_layout.addStretch(2)
        self.mode_plot_layout = QVBoxLayout()
        self.mode_plot_layout.addWidget(self.cmplx_plot_widget)
        left_pane_layout.addLayout(self.mode_plot_layout)
        left_pane_layout.addStretch(2)
        left_pane_layout.addWidget(self.mode_val_view)
        left_pane_layout.addStretch(1)

        main_layout.addLayout(left_pane_layout)
        main_layout.addWidget(self.canvas)
        main_layout.setStretchFactor(self.canvas, 1)
        #main_layout.addLayout(right_pane_layout)
        vbox = QVBoxLayout()
        vbox.addLayout(main_layout)
        vbox.addLayout(self.create_buttons())
        main_frame.setLayout(vbox)
        self.stabil_plot.fig.set_facecolor('none')
        self.setCentralWidget(main_frame)
        self.current_mode = (0, 0)

        return

    def create_buttons(self):
        b0 = QPushButton('Apply')
        b0.released.connect(self.update_stabil_view)
        b1 = QPushButton('Save Figure')
        b1.released.connect(self.save_figure)

        b2 = QPushButton('Export Results')
        b2.released.connect(self.save_results)
        b3 = QPushButton('Save State')
        b3.released.connect(self.save_state)
        b4 = QPushButton('OK and Close')
        b4.released.connect(self.close)

        lay = QHBoxLayout()

        lay.addWidget(b0)
        lay.addWidget(b1)
        lay.addWidget(b2)
        lay.addWidget(b3)
        lay.addWidget(b4)
        lay.addStretch()

        return lay

    def create_histo_plot_f(self):
        '''
        create
        show/hide
        update stable
        '''
        # print('here')
        array = self.stabil_calc.freq_diffs
        self.histo_plot_f = self.create_histo_plot(array,
                                                   self.histo_plot_f, title='Frequency Differences (percent)',
                                                   ranges=(0, 1),
                                                   select_ranges=[
                                                       float(self.df_edit.text()) / 100],
                                                   select_callback=[lambda x: [self.df_edit.setText(str(x * 100)), self.update_stabil_view()]])

    def create_histo_plot_sf(self):
        '''
        create
        show/hide
        update stable
        '''
        array = np.ma.array(
            self.stabil_calc.modal_data.std_frequencies / self.stabil_calc.modal_data.modal_frequencies)
        self.histo_plot_sf = self.create_histo_plot(array,
                                                    self.histo_plot_sf, title='CoV Frequencies (percent)',
                                                    ranges=(0, 1),
                                                    select_ranges=[
                                                        float(self.stdf_edit.text()) / 100],
                                                    select_callback=[lambda x: [self.stdf_edit.setText(str(x * 100)), self.update_stabil_view()]])

    def create_histo_plot_d(self):
        '''
        create
        show/hide
        update stable
        '''
        array = self.stabil_calc.damp_diffs
        self.histo_plot_d = self.create_histo_plot(array,
                                                   self.histo_plot_d,
                                                   title='Damping Differences (percent)',
                                                   ranges=(0, 1),
                                                   select_ranges=[
                                                       float(self.dd_edit.text()) / 100],
                                                   select_callback=[lambda x: [self.dd_edit.setText(str(x * 100)), self.update_stabil_view()]])

    def create_histo_plot_sd(self):
        '''
        create
        show/hide
        update stable
        '''
        array = np.ma.array(
            self.stabil_calc.modal_data.std_damping / self.stabil_calc.modal_data.modal_damping)
        self.histo_plot_sd = self.create_histo_plot(array,
                                                    self.histo_plot_sd, title='CoV Damping (percent)',
                                                    ranges=(0, 1),
                                                    select_ranges=[
                                                        float(self.stdd_edit.text()) / 100],
                                                    select_callback=[lambda x: [self.stdd_edit.setText(str(x * 100)), self.update_stabil_view()]])

    def create_histo_plot_dr(self):
        '''
        create
        show/hide
        update stable
        '''
        array = np.ma.array(self.stabil_calc.modal_data.modal_damping)
        self.histo_plot_dr = self.create_histo_plot(array,
                                                    self.histo_plot_dr, title='Damping range ',
                                                    ranges=(0, 10),
                                                    select_ranges=[
                                                        float(self.d_min_edit.text()), float(self.d_max_edit.text())],
                                                    select_callback=[lambda x: [self.d_min_edit.setText(str(x)), self.update_stabil_view()], lambda x: [self.d_max_edit.setText(str(x)), self.update_stabil_view()]])

    def create_histo_plot_mac(self):
        '''
        create
        show/hide
        update stable
        '''
        array = self.stabil_calc.MAC_diffs
        self.histo_plot_mac = self.create_histo_plot(array,
                                                     self.histo_plot_mac, title='MAC Diffs (percent)',
                                                     ranges=(0, 1),
                                                     select_ranges=[
                                                         float(self.mac_edit.text()) / 100],
                                                     select_callback=[lambda x: [self.mac_edit.setText(str(x * 100)), self.update_stabil_view()]])

    def create_histo_plot_mpc(self):
        '''
        create
        show/hide
        update stable
        '''
        array = self.stabil_calc.MPC_matrix
        self.histo_plot_mpc = self.create_histo_plot(array,
                                                     self.histo_plot_mpc, title='MPC',
                                                     ranges=(0, 1),
                                                     select_ranges=[
                                                         float(self.mpc_edit.text()), None],
                                                     select_callback=[lambda x: [self.mpc_edit.setText(str(x)), self.update_stabil_view()], str])

    def create_histo_plot_mpd(self):
        '''
        create
        show/hide
        update stable
        '''
        array = self.stabil_calc.MPD_matrix
        self.histo_plot_mpd = self.create_histo_plot(array,
                                                     self.histo_plot_mpd, title='MPD',
                                                     ranges=(0, 90),
                                                     select_ranges=[
                                                         float(self.mpd_edit.text())],
                                                     select_callback=[lambda x: [self.mpd_edit.setText(str(x)), self.update_stabil_view()]])
 
    def show_MC_plot(self):

        b=self.sender().isChecked()
        self.stabil_plot.show_MC(b)

    def create_histo_plot(self, array, plot_obj, title='', ranges=None, select_ranges=[None], select_callback=[None]):
        '''
        should work like following:
        
        button press    if None        -> visible = True, create
                        if visible     -> visible = False, hide
                        if not visible -> visible = True, show
        update stabil   if None        -> skip
                        if visible     -> visible = visible, update
                        if not visible -> visible = visible, update
        close button                   -> visible = False, hide
        
        but doesn't, since the function can not distinguish between 
            "button press" and "update stabil"
        '''
        old_mask = np.copy(array.mask)
        array.mask = np.ma.nomask

        mask_stable = self.stabil_calc.get_stabilization_mask('mask_stable')
        if len(array.shape) == 3:
            mask = array == 0
            array.mask = mask
            a = np.min(array, axis=2)
            a.mask = mask_stable
            stable_data = a.compressed()
        else:
            array.mask = mask_stable
            stable_data = array.compressed()

        if plot_obj is None:  # create
            #print('here again')
            mask_pre = self.stabil_calc.get_stabilization_mask('mask_pre')

            if len(array.shape) == 3:
                mask = array == 0
                array.mask = mask
                a = np.min(array, axis=2)
                a.mask = mask_pre
                all_data = a.compressed()
            else:
                array.mask = mask_pre
                all_data = array.compressed()

            plot_obj = HistoPlot(all_data, stable_data, title, ranges,
                                 select_ranges=select_ranges, select_callback=select_callback)
        else:  # update
            plot_obj.update_histo(stable_data, select_ranges)

        array.mask = old_mask

        # show (hide is accomplished by just closing the window closeEvent was overridden to self.hide() )
        # print('show')
        if plot_obj.visible:
            plot_obj.show()
        return plot_obj

    def update_mode_val_view(self, index):
        # display information about currently selected mode
        i = self.stabil_calc.select_modes[index]

        n, f, stdf, d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC,ex_1,ex_2 = self.stabil_calc.get_modal_values(
            i)
        if self.stabil_calc.capabilities['std']:
            import scipy.stats
            num_blocks = self.stabil_calc.modal_data.num_blocks
            stdf = scipy.stats.t.ppf(0.975,num_blocks)*stdf/np.sqrt(num_blocks)
            stdd = scipy.stats.t.ppf(0.975,num_blocks)*stdd/np.sqrt(num_blocks)
            
        self.current_mode = i
        s = ''
        for text, val in [('Frequency=%1.3fHz, \n' % (f),       f   ),
                          ('CI Frequency ± %1.3e, \n' % (stdf), stdf),
                          ('Order=%1.0f, \n' % (n),             n   ),
                          ('Damping=%1.3f%%,  \n' % ( d),       d   ),
                          ('CI Damping ± %1.3e,  \n' % (stdd),  stdd),
                          ('MPC=%1.5f, \n' % (mpc),             mpc ),
                          ('MP=%1.3f\u00b0, \n' % (mp),         mp  ),
                          ('MPD=%1.5f\u00b0, \n' % (mpd),       mpd ),
                          ('dMP=%1.3f\u00b0, \n' % (dmp),       dmp ),
                          #('dMPD=%1.5f\u00b0, \n' % (dmpd),     dmpd),
                          ('MTN=%1.5f, \n' % (mtn),             mtn ),
                          ('MC=%1.5f, \n' % (MC),               MC  ),
                          ('Ext=%1.5f\u00b0, \n' % (ex_1),      ex_1),
                          ('Ext=%1.3f\u00b0, \n' % (ex_2),      ex_2)
                          ]:
            if val is not np.nan:
                s += text
        self.mode_val_view.setText(s)
        height = self.mode_val_view.document().size().toSize().height() + 3
        self.mode_val_view.setFixedHeight(height)
        self.update_mode_plot(i, mp)

    @pyqtSlot(tuple)
    def mode_selector_add(self, i):
        # add mode tomode_selector and select it
        n, f, stdf, d, stdd, mpc, mp, mpd,dmp, dmpd, mtn, MC, ex_1, ex_2 = self.stabil_calc.get_modal_values(
            i)
        index = self.stabil_calc.select_modes.index(i)
        #print(n,f,d,mpc, mp, mpd)
        # print(index)
        text = '{} - {:2.3f}'.format(index, f)
        self.mode_selector.currentIndexChanged[
            int].disconnect(self.update_mode_val_view)
        self.mode_selector.addItem(text)
        found = False
        for index in range(self.mode_selector.count()):
            if text == self.mode_selector.itemText(index):
                found = True
        self.mode_selector.setCurrentIndex(index)
        self.update_mode_val_view(index)
        self.mode_selector.currentIndexChanged[
            int].connect(self.update_mode_val_view)

    @pyqtSlot(tuple)
    def mode_selector_take(self, i_):
        if self.current_mode == i_:
            if self.stabil_calc.select_modes:
                self.current_mode = self.stabil_calc.select_modes[0]
            else:
                self.current_mode = (0, 0)
        self.mode_selector.currentIndexChanged[
            int].disconnect(self.update_mode_val_view)
        self.mode_selector.clear()

        for index, i in enumerate(self.stabil_calc.select_modes):
            n,  f, stdf,  d, stdd, mpc, mp, mpd,dmp, dmpd, mtn, MC, ex_1, ex_2  = self.stabil_calc.get_modal_values(
                i)
            text = '{} - {:2.3f}'.format(index, f)
            self.mode_selector.addItem(text)
            if self.current_mode == i:
                for ind in range(self.mode_selector.count()):
                    if text == self.mode_selector.itemText(ind):
                        break
            else:
                ind = 0

        if self.mode_selector.count():
            self.mode_selector.setCurrentIndex(ind)
            self.update_mode_val_view(ind)
        self.mode_selector.currentIndexChanged[
            int].connect(self.update_mode_val_view)

    def update_mode_plot(self, i, mpd=None):
        # update the plot of the currently selected mode
        msh = self.stabil_calc.get_mode_shape(i)
        self.cmpl_plot.scatter_this(msh, mpd)
        # time.sleep(1)
        if self.msh_plot is not None:
            self.msh_plot.plot_this(i)

    def toggle_msh_plot(self, b):
        # change the type of mode plot
        # print('msh',b)
        if b:
            self.msh_plot.show()
        else:
            self.msh_plot.hide()

    def toggle_cpl_plot(self, b):
        # change the type of mode plot

        # print('cpl',b)
        if b:
            self.cmpl_plot.show()
        else:
            self.cmpl_plot.hide()

    def update_value_view(self, i):

        n, f, stdf, d, stdd, mpc, mp, mpd,dmp, dmpd, mtn, MC, ex_1, ex_2 = self.stabil_calc.get_modal_values(
            i)
        

        if self.stabil_calc.capabilities['std']:
            import scipy.stats
            num_blocks = self.stabil_calc.modal_data.num_blocks
            stdf = scipy.stats.t.ppf(0.975,num_blocks)*stdf/np.sqrt(num_blocks)
            stdd = scipy.stats.t.ppf(0.975,num_blocks)*stdd/np.sqrt(num_blocks)
            
        self.current_mode = i
        s = ''
        for text, val in [('Frequency=%1.3fHz, \n' % (f),       f   ),
                          ('CI Frequency ± %1.3e, \n' % (stdf), stdf),
                          ('Order=%1.0f, \n' % (n),             n   ),
                          ('Damping=%1.3f%%,  \n' % ( d),       d   ),
                          ('CI Damping ± %1.3e,  \n' % (stdd),  stdd),
                          ('MPC=%1.5f, \n' % (mpc),             mpc ),
                          ('MP=%1.3f\u00b0, \n' % (mp),         mp  ),
                          ('MPD=%1.5f\u00b0, \n' % (mpd),       mpd ),
                          ('dMP=%1.3f\u00b0, \n' % (dmp),       dmp ),
                          #('dMPD=%1.5f\u00b0, \n' % (dmpd),     dmpd),
                          ('MTN=%1.5f, \n' % (mtn),             mtn ),
                          ('MC=%1.5f, \n' % (MC),               MC  ),
                          ('Ext=%1.5f\u00b0, \n' % (ex_1),      ex_1),
                          ('Ext=%1.3f\u00b0, \n' % (ex_2),      ex_2)
                          ]:
            if val is not np.nan:
                s += text

        self.current_value_view.setText(s)
        height = self.current_value_view.document(
        ).size().toSize().height() + 3
        self.current_value_view.setFixedHeight(height)

    def update_stabil_view(self):
        df_max = float(self.df_edit.text()) / 100
        dd_max = float(self.dd_edit.text()) / 100

        if self.stabil_calc.capabilities['std']:
            stdf_max = float(self.stdf_edit.text()) / 100
        else:
            stdf_max = None

        if self.stabil_calc.capabilities['std']:
            stdd_max = float(self.stdd_edit.text()) / 100
        else:
            stdd_max = None

        if self.stabil_calc.capabilities['msh']:
            dmac_max = float(self.mac_edit.text()) / 100
        else:
            dmac_max = None

        d_range = (
            float(self.d_min_edit.text()), float(self.d_max_edit.text()))

        if self.stabil_calc.capabilities['msh']:
            mpc_min = float(self.mpc_edit.text())
            mpd_max = float(self.mpd_edit.text())
        else:
            mpc_min = None
            mpd_max = None
            
        if self.stabil_calc.capabilities['MC']:
            MC_min = float(self.MC_edit.text())
        else:
            MC_min = None

        f_range = (float(self.freq_low.text()), float(self.freq_high.text()))
        order_range = (int(self.n_low.text()), int(
            self.n_step.text()), int(self.n_high.text()))

        self.stabil_plot.update_stabilization(df_max=df_max, stdf_max=stdf_max, dd_max=dd_max, stdd_max=stdd_max,
                                              dmac_max=dmac_max, d_range=d_range, mpc_min=mpc_min, mpd_max=mpd_max,
                                              MC_min = MC_min, order_range=order_range)
        self.stabil_plot.update_xlim(f_range)
        self.stabil_plot.update_ylim((order_range[0], order_range[2]))

        if self.histo_plot_f is not None:
            self.create_histo_plot_f()
        if self.histo_plot_sf is not None:
            self.create_histo_plot_sf()
        if self.histo_plot_d is not None:
            self.create_histo_plot_d()
        if self.histo_plot_sd is not None:
            self.create_histo_plot_sd()
        if self.histo_plot_dr is not None:
            self.create_histo_plot_dr()
        if self.histo_plot_mac is not None:
            self.create_histo_plot_mac()
        if self.histo_plot_mpc is not None:
            self.create_histo_plot_mpc()
        if self.histo_plot_mpd is not None:
            self.create_histo_plot_mpd()

    def create_menu(self):
        '''
        create the menubar and add actions to it
        '''
        def add_actions(target, actions):
            for action in actions:
                if action is None:
                    target.addSeparator()
                else:
                    target.addAction(action)

        def create_action(text, slot=None, shortcut=None,
                          icon=None, tip=None, checkable=False,
                          signal="triggered()"):
            action = QAction(text, self)
            if icon is not None:
                action.setIcon(QIcon(":/%s.png" % icon))
            if shortcut is not None:
                action.setShortcut(shortcut)
            if tip is not None:
                action.setToolTip(tip)
                action.setStatusTip(tip)
            if slot is not None:
                getattr(action, signal.strip('()')).connect(slot)
            if checkable:
                action.setCheckable(True)
            return action

        file_menu = self.menuBar().addMenu("&File")

        load_file_action = create_action("&Save plot",
                                         shortcut="Ctrl+S",
                                         slot=None,
                                         tip="Save the plot")
        quit_action = create_action("&Quit",
                                    slot=self.close,
                                    shortcut="Ctrl+Q",
                                    tip="Close the application")

        add_actions(file_menu,
                    (load_file_action, None, quit_action))

        help_menu = self.menuBar().addMenu("&Help")

    def create_stab_val_widget(self, df_max=1, dd_max=5, d_mac=1,
                               d_range=(0, 5), mpc_min=0.9, mpd_max=15):
        layout = QGridLayout()
        layout.addWidget(QLabel('Stabilization Criteria'), 1, 1, 1, 3)

        layout.setColumnStretch(2, 1)
        i = 2

        if self.stabil_calc.capabilities['f']:
            layout.addWidget(QLabel('Frequency [%]'), i, 1)
            self.df_edit = QLineEdit(str(df_max))
            self.df_edit.setMaxLength(8)
            self.df_edit.setFixedWidth(60)
            layout.addWidget(self.df_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_f)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['std']:
            layout.addWidget(QLabel('CoV F. [% of F]'), i, 1)
            self.stdf_edit = QLineEdit('100')
            self.stdf_edit.setMaxLength(8)
            self.stdf_edit.setFixedWidth(60)
            layout.addWidget(self.stdf_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_sf)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['d']:
            layout.addWidget(QLabel('Damping[%]'), i, 1)
            self.dd_edit = QLineEdit(str(dd_max))
            self.dd_edit.setMaxLength(8)
            self.dd_edit.setFixedWidth(60)
            layout.addWidget(self.dd_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_d)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['std']:
            layout.addWidget(QLabel('CoV D. [% of D]'), i, 1)
            self.stdd_edit = QLineEdit('100')
            self.stdd_edit.setMaxLength(8)
            self.stdd_edit.setFixedWidth(60)
            layout.addWidget(self.stdd_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_sd)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['msh']:
            layout.addWidget(QLabel('MAC [%]'), i, 1)
            self.mac_edit = QLineEdit(str(d_mac))
            self.mac_edit.setMaxLength(8)
            self.mac_edit.setFixedWidth(60)
            layout.addWidget(self.mac_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_mac)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['d']:
            layout.addWidget(QLabel('Damping range [%]'), i, 1)
            self.d_min_edit = QLineEdit(str(d_range[0]))
            self.d_min_edit.setMaxLength(8)
            self.d_min_edit.setFixedWidth(60)
            self.d_max_edit = QLineEdit(str(d_range[1]))
            self.d_max_edit.setMaxLength(8)
            self.d_max_edit.setFixedWidth(60)
            lay = QHBoxLayout()
            lay.addStretch()
            lay.addWidget(self.d_min_edit)
            lay.addWidget(QLabel('to'))
            lay.addWidget(self.d_max_edit)
            layout.addLayout(lay, i, 2, 1, 2)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_dr)
            layout.addWidget(button, i, 4)
            i += 1

        layout.setRowStretch(i, 2)
        i += 1

        if self.stabil_calc.capabilities['msh']:
            layout.addWidget(QLabel('MPC_min '), i, 1)
            self.mpc_edit = QLineEdit(str(mpc_min))
            self.mpc_edit.setMaxLength(8)
            self.mpc_edit.setFixedWidth(60)
            layout.addWidget(self.mpc_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_mpc)
            layout.addWidget(button, i, 4)
            i += 1

            layout.addWidget(QLabel('MPD_max [°]'), i, 1)
            self.mpd_edit = QLineEdit(str(mpd_max))
            self.mpd_edit.setMaxLength(8)
            self.mpd_edit.setFixedWidth(60)
            layout.addWidget(self.mpd_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_mpd)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['mtn']:
            layout.addWidget(QLabel('MTN_max []'), i, 1)
            self.mtn_edit = QLineEdit('0')
            self.mtn_edit.setMaxLength(8)
            self.mtn_edit.setFixedWidth(60)
            layout.addWidget(self.mtn_edit, i, 3)
            button = QPushButton('Show Histo')
            button.released.connect(self.create_histo_plot_mtn)
            layout.addWidget(button, i, 4)
            i += 1
            
        if self.stabil_calc.capabilities['MC']:
            layout.addWidget(QLabel('MC_min []'), i, 1)
            self.MC_edit = QLineEdit('0')
            self.MC_edit.setMaxLength(8)
            self.MC_edit.setFixedWidth(60)
            layout.addWidget(self.MC_edit, i, 3)
            button = QPushButton('Show MC')
            button.setCheckable(True)
            button.released.connect(self.show_MC_plot)
            layout.addWidget(button, i, 4)
            i += 1

        if self.stabil_calc.capabilities['auto']:
            b0 = QPushButton('Clear automatically')
            b0.released.connect(self.prepare_auto_clearing)

            self.num_iter_edit = QLineEdit(str(self.stabil_calc.num_iter))

            layout.addWidget(b0, i, 1)
            layout.addWidget(self.num_iter_edit, i, 2)
            i += 1

            b1 = QPushButton('Classify automatically')
            b1.released.connect(self.prepare_auto_classification)

            self.use_stabil_box = QCheckBox('Use Stabilization')
            self.use_stabil_box.setTristate(False)
            self.use_stabil_box.setChecked(False)
            self.threshold_box = QLineEdit()
            self.threshold_box.setPlaceholderText(
                str(self.stabil_calc.threshold))

            layout.addWidget(b1, i, 1)
            layout.addWidget(self.use_stabil_box, i, 2)
            layout.addWidget(self.threshold_box, i, 3)
            i += 1

            b2 = QPushButton('Select automatically')
            b2.released.connect(self.prepare_auto_selection)

            self.num_modes_box = QLineEdit(str(0))

            layout.addWidget(b2, i, 1)
            layout.addWidget(self.num_modes_box, i, 2)

        return layout

    def prepare_auto_clearing(self):
        assert self.stabil_calc.capabilities['auto']
        num_iter = int(self.num_iter_edit.text())


        if isinstance(self.stabil_calc, StabilCluster):
            self.stabil_calc.automatic_clearing(num_iter)
            self.threshold_box.setPlaceholderText(
                str(self.stabil_calc.threshold))
            self.stabil_plot.update_stabilization()
            self.check_clear.setChecked(True)

    def prepare_auto_classification(self):
        assert self.stabil_calc.capabilities['auto']
        use_stabil = self.use_stabil_box.isChecked()

        threshold = self.threshold_box.text()
        #print(threshold, type(threshold), threshold.isnumeric())
        if threshold.isnumeric():
            threshold = float(threshold)
        else:
            threshold = None

        if isinstance(self.stabil_calc, StabilCluster):
            self.stabil_calc.automatic_classification(threshold, use_stabil)
            self.check_select.setChecked(True)

    def prepare_auto_selection(self):
        assert self.stabil_calc.capabilities['auto']
        num_modes = self.num_modes_box.text()
        if num_modes.isnumeric():
            num_modes = int(num_modes)
        else:
            num_modes = 0

        # print(self.stabil_calc.select_modes)
        for datapoint in reversed(self.stabil_calc.select_modes):
            self.stabil_plot.cursor.remove_datapoint(datapoint)
        # print(self.stabil_calc.select_modes)

        if isinstance(self.stabil_calc, StabilCluster):
            self.stabil_calc.automatic_selection(num_modes)
            #self.stabil_plot.update_stabilization()
            # self.stabil_calc.plot_selection()

    def create_diag_val_widget(self, show_sf=True, show_sd=True,
                               show_sv=True, show_sa=True, show_all=True,
                               show_psd=False, snap_to='sa', f_range=(0, 0), n_range=(0, 1, 0)):
        layout = QGridLayout()

        layout.addWidget(QLabel('View Settings'), 1, 1, 1, 2)

        i = 2

#         check_sf = QCheckBox('Unstable in Frequency only')
#         check_sf.setChecked(show_sf)
#         self.stabil_plot.toggle_df(show_sf)
#         check_sf.stateChanged.connect(self.stabil_plot.toggle_df)
#         snap_sf = QRadioButton()
#         snap_sf.toggled.connect(self.stabil_plot.snap_frequency)
#         snap_sf.setChecked(snap_to == 'sf')
#         layout.addWidget(check_sf, i, 1)
#         layout.addWidget(snap_sf, i, 2)
#         i += 1
# 
#         check_sd = QCheckBox('Unstable in Damping only')
#         check_sd.setChecked(show_sd)
#         self.stabil_plot.toggle_dd(show_sd)
#         check_sd.stateChanged.connect(self.stabil_plot.toggle_dd)
#         snap_sd = QRadioButton()
#         snap_sd.toggled.connect(self.stabil_plot.snap_damping)
#         snap_sd.setChecked(snap_to == 'sd')
#         layout.addWidget(check_sd, i, 1)
#         layout.addWidget(snap_sd, i, 2)
#         i += 1
# 
#         if self.stabil_calc.capabilities['msh']:
#             check_sv = QCheckBox('Unstable in MAC only')
#             check_sv.setChecked(show_sv)
#             self.stabil_plot.toggle_dmac(show_sv)
#             check_sv.stateChanged.connect(self.stabil_plot.toggle_dmac)
#             snap_sv = QRadioButton()
#             snap_sv.toggled.connect(self.stabil_plot.snap_vector)
#             snap_sv.setChecked(snap_to == 'sv')
#             layout.addWidget(check_sv, i, 1)
#             layout.addWidget(snap_sv, i, 2)
#             i += 1        

        check_sa = QCheckBox('Stable Pole')
        check_sa.setChecked(show_sa)
        self.stabil_plot.toggle_stable(show_sa)
        check_sa.stateChanged.connect(self.stabil_plot.toggle_stable)
        snap_sa = QRadioButton()
        snap_sa.toggled.connect(self.stabil_plot.snap_stable)
        snap_sa.setChecked(snap_to == 'sa')
        layout.addWidget(check_sa, i, 1)
        layout.addWidget(snap_sa, i, 2)
        i += 1

        check_all = QCheckBox('All Poles')
        check_all.setChecked(show_all)
        self.stabil_plot.toggle_all(show_all)
        check_all.stateChanged.connect(self.stabil_plot.toggle_all)
        snap_all = QRadioButton()
        snap_all.toggled.connect(self.stabil_plot.snap_all)
        snap_all.setChecked(snap_to == 'all')
        layout.addWidget(check_all, i, 1)
        layout.addWidget(snap_all, i, 2)
        i += 1

        if self.stabil_calc.capabilities['auto']:
            show_clear = self.stabil_calc.state >= 3
            check_clear = QCheckBox('AutoClear')
            check_clear.setChecked(show_clear)
            self.check_clear = check_clear
            self.stabil_plot.toggle_clear(show_clear)
            check_clear.stateChanged.connect(self.stabil_plot.toggle_clear)
            snap_clear = QRadioButton()
            snap_clear.toggled.connect(self.stabil_plot.snap_clear)
            snap_clear.setChecked(snap_to == 'clear')
            layout.addWidget(check_clear, i, 1)
            layout.addWidget(snap_clear, i, 2)
            i += 1

            show_select = self.stabil_calc.state >= 4
            check_select = QCheckBox('AutoSelect')
            check_select.setChecked(show_select)
            self.check_select = check_select
            self.stabil_plot.toggle_select(show_select)
            check_select.stateChanged.connect(self.stabil_plot.toggle_select)
            snap_select = QRadioButton()
            snap_select.toggled.connect(self.stabil_plot.snap_select)
            snap_select.setChecked(snap_to == 'select')
            layout.addWidget(check_select, i, 1)
            layout.addWidget(snap_select, i, 2)
            i += 1

        if self.stabil_calc.capabilities['data']:
            psd_check = QCheckBox('Show PSD')
            psd_check.setChecked(show_psd)
            self.stabil_plot.plot_fft(show_psd)
            psd_check.stateChanged.connect(self.stabil_plot.plot_fft)
            layout.addWidget(psd_check, i, 1, 1, 2)
            i += 1        


        lay = QHBoxLayout()
        lay.addWidget(QLabel('Freq. range:'))
        if f_range[1] == 0:
            f_range = (f_range[0], self.stabil_calc.get_max_f())
        self.freq_low = QLineEdit('{:2.3f}'.format(f_range[0]))
        self.freq_low.setFixedWidth(60)
        self.freq_high = QLineEdit('{:2.3f}'.format(f_range[1] * 1.05))
        self.freq_high.setFixedWidth(60)
        lay.addWidget(self.freq_low)
        lay.addWidget(QLabel('to'))
        lay.addWidget(self.freq_high)
        lay.addWidget(QLabel('[Hz]'))
        layout.addLayout(lay, i, 1, 1, 2)
        i += 1

        lay = QHBoxLayout()
        lay.addWidget(QLabel('Order. range (low:step:high):'))
        if n_range[2] == 0:
            n_range = (
                n_range[0], n_range[1], self.stabil_calc.modal_data.max_model_order)
        self.n_low = QLineEdit('{:2d}'.format(n_range[0]))
        self.n_low.setFixedWidth(60)
        self.n_step = QLineEdit('{:2d}'.format(n_range[1]))
        self.n_step.setFixedWidth(60)
        self.n_high = QLineEdit('{:2d}'.format(n_range[2]))
        self.n_high.setFixedWidth(60)
        lay.addWidget(self.n_low)
        lay.addWidget(self.n_step)
        lay.addWidget(self.n_high)
        layout.addLayout(lay, i, 1, 1, 2)
        i += 1

        return layout

    def save_figure(self, fname=None):

        # copied and modified from
        # matplotlib.backends.backend_qt4.NavigationToolbar2QT
        canvas = self.stabil_plot.ax.figure.canvas

        filetypes = canvas.get_supported_filetypes_grouped()
        sorted_filetypes = list(filetypes.items())
        sorted_filetypes.sort()
        default_filetype = canvas.get_default_filetype()

        startpath = rcParams.get('savefig.directory', '')
        startpath = os.path.expanduser(startpath)
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        if fname is None:
            fname, ext = QFileDialog.getSaveFileName(self, caption="Choose a filename to save to",
                                                     directory=start, filter=filters)
            # print(fname)
        self.stabil_plot.save_figure(fname)

    def save_results(self):

        fname, fext = QFileDialog.getSaveFileName(self, caption="Choose a directory to save to",
                                                  directory=os.getcwd(), filter='Text File (*.txt)')
        #fname, fext = os.path.splitext(fname)

        if fext != 'txt':
            fname += '.txt'

        self.stabil_calc.export_results(fname)

    def save_state(self):

        fname, fext = QFileDialog.getSaveFileName(self, caption="Choose a directory to save to",
                                                  directory=os.getcwd(), filter='Numpy Archive File (*.npz)')

        if fname == '':
            return
        fname, fext = os.path.splitext(fname)

        if fext != 'npz':
            fname += '.npz'

        self.stabil_calc.save_state(fname)
        self.close()

    def closeEvent(self, *args, **kwargs):
        if self.msh_plot is not None:
            self.msh_plot.mode_shape_plot.stop_ani()

        #self.stabil_calc.select_modes = self.stabil_calc.select_modes
        self.deleteLater()

        return QMainWindow.closeEvent(self, *args, **kwargs)
    
    def keyPressEvent(self, e):
        #print(e.key())
        if e.key() == Qt.Key_Return or e.key()== Qt.Key_Enter:
            self.update_stabil_view()
            #print('2')
        super().keyPressEvent(e)

class StabilCalc(object):

    def __init__(self, modal_data, prep_data=None, **kwargs):

        super().__init__()
        print(type(modal_data))
        assert isinstance(
            modal_data, (BRSSICovRef, SSIData, VarSSIRef, PRCE, 
                         PLSCF, SSIDataMC, PogerSSICovRef, ERA))

        self.modal_data = modal_data
        
        self.extra_func = None

        self.setup_name = modal_data.setup_name
        self.start_time = modal_data.start_time

        if prep_data is not None:
            assert isinstance(prep_data, PreprocessData)
        self.prep_data = prep_data
        
        has_mode_shapes = self.modal_data.__dict__.get(
            'mode_shapes', None) is not None
        has_variances = self.modal_data.__dict__.get(
            'std_frequencies', None) is not None
        has_data = prep_data is not None
        has_MC = self.modal_data.__dict__.get(
            'modal_contributions', None) is not None
        has_ev = self.modal_data.__dict__.get(
            'eigenvalues', None) is not None

        self.capabilities = {'f': 1,
                             'd': 1,
                             'msh': has_mode_shapes,
                             'std': has_variances,
                             'ev': has_ev,
                             'mtn': 0,
                             'MC': has_MC,
                             'auto': isinstance(self, StabilCluster),
                             'data': has_data}
        
        
        if self.capabilities['ev']:
            self.masked_lambda = np.ma.array(self.modal_data.eigenvalues, fill_value = 0)
            
        self.masked_frequencies = np.ma.array(
            self.modal_data.modal_frequencies, fill_value = 0)
        self.masked_damping = np.ma.array(self.modal_data.modal_damping, fill_value = 0)

        max_model_order = self.modal_data.max_model_order
        self.num_solutions = self.modal_data.modal_frequencies.shape[1]
        
        self.order_dummy = np.ma.array(
            [[order] * self.num_solutions for order in range(max_model_order)], fill_value = 0)

        # stable-in-... masks
        self.masks = {'mask_pre':   None,  # some constraints (f>0.0, order_range, etc)
                      'mask_ad':    None,  # absolute damping
                      'mask_stdf':    None,  # uncertainty frequency
                      'mask_stdd':    None,  # uncertainty damping
                      'mask_mpc':  None,  # absolute modal phase collinearity
                      'mask_mpd':  None,  # absolute mean phase deviation
                      'mask_mtn':  None,  # absolute modal transfer norm
                      'mask_df':    None,  # difference frequency
                      'mask_dd':    None,  # difference damping
                      'mask_dmac':  None,  # difference mac
                      'mask_dev':   None,  # difference eigenvalue
                      'mask_dmtn':  None,  # difference modal transfer norm
                      'mask_stable':   None  # stable in all criteria
                      }

        # only-unstable-in-... masks
        self.nmasks = {'mask_ad':    None,  # absolute damping
                       'mask_stdf':    None,  # uncertainty frequency
                       'mask_stdd':    None,  # uncertainty damping
                       # absolute modal phase collineratity
                       'mask_ampc':  None,
                       'mask_ampd':  None,  # absolute mean phase deviation
                       'mask_amtn':  None,  # absolute modal transfer norm
                       'mask_df':    None,  # difference frequency
                       'mask_dd':    None,  # difference damping
                       'mask_dmac':  None,  # difference mac
                       'mask_dev':   None,  # difference eigenvalue
                       'mask_dmtn':  None,  # difference modal transfer norm
                       }

        self.select_modes = []
        self.select_callback = None
        self.state = 0


        #print(self.capabilities)
        # self.calculate_soft_critera_matrices()

    def calculate_soft_critera_matrices(self):
        print('Checking stabilisation criteria...')

        # Direction 1: model order, Direction 2: current pole, Direction 3:
        # previous pole:
        max_model_order = self.modal_data.max_model_order
        num_solutions = self.num_solutions
        capabilities = self.capabilities
        
        lambda_diffs = np.ma.zeros(
            (max_model_order, num_solutions, num_solutions), fill_value=0)
        freq_diffs = np.ma.zeros(
            (max_model_order, num_solutions, num_solutions), fill_value=0)
        damp_diffs = np.ma.zeros(
            (max_model_order, num_solutions, num_solutions), fill_value=0)
        
        if capabilities['msh']:
            MAC_diffs = np.ma.zeros(
                (max_model_order, num_solutions, num_solutions), fill_value=0)
            MPD_diffs = np.ma.zeros(
                (max_model_order, num_solutions, num_solutions), fill_value=0)
            MP_diffs = np.ma.zeros(
                (max_model_order, num_solutions, num_solutions), fill_value=0)
    
            MPC_matrix = np.ma.zeros(
                (max_model_order, num_solutions), fill_value=0)
            MP_matrix = np.ma.zeros(
                (max_model_order, num_solutions), fill_value=0)
            MPD_matrix = np.ma.zeros(
                (max_model_order, num_solutions), fill_value=0)
        
        if capabilities['ev']:
            prev_lambda_row = self.masked_lambda.data[0,:]
        prev_freq_row = self.masked_frequencies[0, :]
        prev_damp_row = self.modal_data.modal_damping[0, :]
        if capabilities['msh']:
            prev_mode_shapes_row = self.modal_data.mode_shapes[:, :, 0]

        # tuple with array of indizes of non-zero frequencies
        if capabilities['ev']:
            prev_non_zero_entries = np.nonzero(prev_lambda_row.imag)
        else:
            prev_non_zero_entries = np.nonzero(prev_freq_row)
        prev_length = len(prev_non_zero_entries[0])
        if capabilities['ev']:
            prev_lambda = prev_lambda_row[prev_non_zero_entries]
        prev_freq = prev_freq_row[prev_non_zero_entries]
        prev_damp = prev_damp_row[prev_non_zero_entries]
        
        if capabilities['msh']:
            prev_mode_shapes = \
               prev_mode_shapes_row[:, prev_non_zero_entries[0]]
        
            prev_MPD, prev_MP_new = self.calculateMPD(prev_mode_shapes)
            prev_MP_new[prev_MP_new>90] -= 180 # in range [-90,90]
            
        printsteps = list(np.linspace(1,max_model_order, 100, dtype=int))
        for curr_order in range(1, max_model_order):                
            while curr_order in printsteps: 
                del printsteps[0]
                print('.',end='', flush=True)
            
            if capabilities['ev']:
                curr_lambda_row = self.masked_lambda.data[(curr_order),:]
            
            curr_freq_row = self.masked_frequencies[(curr_order), :]

            curr_damp_row = self.modal_data.modal_damping[
                (curr_order), :]
            
            if capabilities['msh']:
                curr_mode_shapes_row = \
                    self.modal_data.mode_shapes[:, :, curr_order]
                    
            # catches zeros and also real poles, which should have been removed in remove conjugates already        
            if capabilities['ev']:
                curr_non_zero_entries = np.nonzero(curr_lambda_row.imag)
            else:
                curr_non_zero_entries = np.nonzero(curr_freq_row)


            curr_length = len(curr_non_zero_entries[0])
            
            #print(curr_length)
            if not curr_length:
                continue
            if capabilities['ev']:
                curr_lambda = curr_lambda_row[curr_non_zero_entries]
            curr_freq = curr_freq_row[curr_non_zero_entries]
            curr_damp = curr_damp_row[curr_non_zero_entries]
            
            if capabilities['msh']:
                curr_mode_shapes = \
                    curr_mode_shapes_row[:, curr_non_zero_entries[0]]
            if capabilities['ev']:
                
                div_lambda = np.maximum(
                    np.repeat(np.expand_dims(np.ma.abs(prev_lambda), axis=1), 
                              curr_lambda.shape[0], axis=1),
                    np.repeat(np.expand_dims(np.ma.abs(curr_lambda), axis=0), 
                              prev_lambda.shape[0], axis=0))
            
            div_freq = np.maximum(
                np.repeat(np.expand_dims(np.abs(prev_freq), axis=1), 
                          curr_freq.shape[0], axis=1),
                np.repeat(np.expand_dims(np.abs(curr_freq), axis=0), 
                          prev_freq.shape[0], axis=0))
            
            div_damp = np.maximum(
                np.repeat(np.expand_dims(np.abs(prev_damp), axis=1), 
                          curr_damp.shape[0], axis=1),
                np.repeat(np.expand_dims(np.abs(curr_damp), axis=0), 
                          prev_damp.shape[0], axis=0))

            if capabilities['msh']:
                mac_diffs = np.transpose(1 - self.calculateMAC(
                    prev_mode_shapes[:, :prev_length], curr_mode_shapes[:,:curr_length]))
                #print(mac_diffs)
                MAC_diffs[curr_order, curr_non_zero_entries[0], :prev_length] = \
                    mac_diffs

                MPC_matrix[curr_order, curr_non_zero_entries[0]] = self.calculateMPC(
                    curr_mode_shapes[:,:curr_length])
                
                curr_MPD, curr_MP = self.calculateMPD(curr_mode_shapes[:,:curr_length])
                MPD_matrix[curr_order, curr_non_zero_entries[0]], MP_matrix[
                    curr_order, curr_non_zero_entries[0]] = curr_MPD, curr_MP
                    
            if capabilities['ev']:
                lambda_diffs[curr_order, curr_non_zero_entries[0], :len(prev_lambda)] = np.abs((np.repeat(
                    np.expand_dims(prev_lambda, axis=1), curr_lambda.shape[0], axis=1) - curr_lambda) / div_lambda).T
            freq_diffs[curr_order, curr_non_zero_entries[0], :len(prev_freq)] = np.abs((np.repeat(
                np.expand_dims(prev_freq, axis=1), curr_freq.shape[0], axis=1) - curr_freq) / div_freq).T
            damp_diffs[curr_order, curr_non_zero_entries[0], :len(prev_damp)] = np.abs((np.repeat(
                np.expand_dims(prev_damp, axis=1), curr_damp.shape[0], axis=1) - curr_damp) / div_damp).T
                
            if capabilities['msh']:
                
                div_MPD = np.maximum(np.repeat(np.expand_dims(np.abs(prev_MPD), axis=1), curr_MPD.shape[0], axis=1),
                                            np.repeat(np.expand_dims(np.abs(curr_MPD), axis=0), prev_MPD.shape[0], axis=0))
                
                MPD_diffs[curr_order, curr_non_zero_entries[0], :len(prev_MPD)] = np.abs((np.repeat(
                    np.expand_dims(prev_MPD, axis=1), curr_MPD.shape[0], axis=1) - curr_MPD) / div_MPD).T
                
                curr_MP_new = np.copy(curr_MP) # in range [0,180]
                curr_MP_new[curr_MP_new>90] -= 180 # in range [-90,90]
                
                div_MP = np.maximum(np.repeat(np.expand_dims(np.abs(prev_MP_new), axis=1), curr_MP_new.shape[0], axis=1),
                                            np.repeat(np.expand_dims(np.abs(curr_MP_new), axis=0), prev_MP_new.shape[0], axis=0))
                
                MP_diffs[curr_order, curr_non_zero_entries[0], :len(prev_MP_new)] = np.abs((np.repeat(
                    np.expand_dims(prev_MP_new, axis=1), curr_MP_new.shape[0], axis=1) - curr_MP_new) / div_MP).T
            
            
            
            if capabilities['ev']:
                prev_lambda = curr_lambda
            prev_freq = curr_freq
            prev_damp = curr_damp
            
            if capabilities['msh']:
                prev_mode_shapes = curr_mode_shapes
                prev_MPD = curr_MPD
                prev_MP_new = curr_MP_new
            
            prev_length = curr_length
            prev_non_zero_entries = curr_non_zero_entries
        if capabilities['ev']:
            self.lambda_diffs = lambda_diffs
        self.freq_diffs = freq_diffs
        self.damp_diffs = damp_diffs   
        self.MAC_diffs = MAC_diffs   
        self.MPD_diffs = MPD_diffs 
        self.MP_diffs = MP_diffs 
          
        self.MPD_matrix = MPD_matrix  
        self.MP_matrix= MP_matrix  
        self.MPC_matrix = MPC_matrix  
        
        
        self.state = 1

        print('.',end='\n', flush=True)
        warnings.warn('Test')
    @staticmethod
    def calculateMAC(v1, v2):
        '''
        expects modeshapes in columns of v1 and/or v2
        outputs mac:
        [MAC(v1[:,0],v2[:,0]),   MAC(v1[:,0],v2[:,1],
         MAC(v1[:,1],v2[:,0]),   MAC(v1[:,1],v2[:,1]]
        '''

        v1_norms = np.einsum('ij,ij->j', v1, v1.conj())
        v2_norms = np.einsum('ij,ij->j', v2, v2.conj())
        MAC = np.abs(np.dot(v1.T, v2.conj()))**2 \
            / np.outer(v1_norms, v2_norms)

        return MAC.real

    @staticmethod
    def calculateMPC(v):

        MPC = np.abs(np.sum(v**2, axis=0))**2 \
            / np.abs(np.einsum('ij,ij->j', v, v.conj()))**2

        return MPC

    @staticmethod
    def calculateMPD(v, weighted=True, regression_type='usv'):
        assert regression_type in ['ortho', 'arithm', 'usv']
        if regression_type == 'ortho':
            # orthogonal regression through origin
            # http://mathforum.org/library/drmath/view/68362.html
            real_ = np.real(v).copy()
            imag_ = np.imag(v).copy()
            ssxy = np.einsum('ij,ij->j', real_, imag_)
            ssxx = np.einsum('ij,ij->j', real_, real_)
            ssyy = np.einsum('ij,ij->j', imag_, imag_)

            MP = np.arctan2(2 * ssxy, (ssxx - ssyy)) / 2

            # rotates complex plane by angle MP
            v_r = v * np.exp(-1j * MP)  # (np.cos(-MP)+1j*np.sin(-MP))
            # calculates phase in range -180 and 180
            phase = np.angle(v_r, True)

            # rotates into 1st and 4th quadrant
            phase[phase > 90] -= 180
            phase[phase < -90] += 180
            # calculates standard deviation

            if not weighted:
                MPD = np.std(phase, axis=0)
            else:
                MPD = np.sqrt(np.average(
                    np.power(phase - np.mean(phase, axis=0), 2), weights=np.absolute(v_r), axis=0))

            #print(np.mean(phase, axis=0), np.sqrt( 
            #    np.mean(np.power(phase, 2), axis=0)), np.std(phase, axis=0), MPD)

            MP *= 180 / np.pi

        elif regression_type == 'arithm':
            phase = np.angle(v, True)

            phase[phase < 0] += 180

            if not weighted:
                MP = np.mean(phase, axis=0)
            else:
                MP = np.average(phase, weights=np.absolute(v), axis=0)

            if not weighted:
                MPD = np.std(phase, axis=0)
            else:
                MPD = np.sqrt(
                    np.average(np.power(phase - MP, 2), weights=np.absolute(v), axis=0))

        elif regression_type == 'usv':

            MP = np.zeros(v.shape[1])
            MPD = np.zeros(v.shape[1])

            for k in range(v.shape[1]):
                mode_shape = np.array(
                    [np.array(v[:, k]).real, np.array(v[:, k]).imag]).T

                U, S, V_T = np.linalg.svd(mode_shape, full_matrices=False)
                # print(U.shape,S.shape,V_T.shape)
                numerator = []
                denominator = []

                import warnings
                for j in range(len(v[:, k])):
                    v[j, k]
                    V_T[1, 1]
                    V_T[1, 0]
                    V_T[0, 1]
                    V_T[1, 1]
                    if weighted:
                        weight = np.abs(v[j, k])
                    else:
                        weight = 1
                    numerator_i = weight * np.arccos(np.abs(V_T[1, 1] * np.array(v[j, k]).real - V_T[
                                                     1, 0] * np.array(v[j, k]).imag) / (np.sqrt(V_T[0, 1]**2 + V_T[1, 1]**2) * np.abs(v[j, k])))
                    warnings.filterwarnings("ignore")
                    # when the arccos function returns NaN, it means that the value should be set 0
                    # the RuntimeWarning might occur since the value in arccos
                    # can be slightly bigger than 0 due to truncations
                    if np.isnan(numerator_i):
                        numerator_i = 0
                    numerator.append(numerator_i)
                    denominator.append(weight)

                MPD[k] = np.degrees((sum(numerator) / sum(denominator)))
                MP[k] = np.degrees(np.arctan(-V_T[1, 0] / V_T[1, 1]))
                # MP in [-pi/2, pi/2] = [-90, 90]
                # phase=np.angle(v[:,k]*np.exp(-1j*np.radians(MP[k])),True)
                # print(np.mean(phase))
                # phase[phase>90]-=180
                # phase[phase<-90]+=180
                # print(np.mean(phase),np.sqrt(np.mean(phase**2)),np.std(phase),MPD[k])

        MP[MP < 0] += 180  # restricted to +imag region
        MPD[MPD < 0] *= -1
        
        #MP [0,180]
        #MPD >= 0
        return MPD, MP

    def export_results(self, fname, binary=False):

        if self.select_modes:
            
            self.masked_frequencies.mask = np.ma.nomask
            self.order_dummy.mask = np.ma.nomask
            
            select_modes = self.select_modes
            selected_freq = [self.masked_frequencies[index]
                             for index in self.select_modes]
            select_modes = [x for (y,x) in sorted(zip(selected_freq,select_modes), key=lambda pair: pair[0])]
            
            selected_freq = [self.masked_frequencies[index]
                             for index in select_modes]
            selected_damp = [self.modal_data.modal_damping[index]
                             for index in select_modes]
            selected_order = [self.order_dummy[index]
                              for index in select_modes]
            if self.capabilities['msh']:
                selected_MPC = [self.MPC_matrix[index]
                                for index in select_modes]
                selected_MP = [self.MP_matrix[index]
                               for index in select_modes]
                selected_MPD = [self.MPD_matrix[index]
                                for index in select_modes]

            if self.capabilities['std']:
                selected_stdf = [self.modal_data.std_frequencies[index]
                                 for index in select_modes]
                selected_stdd = [self.modal_data.std_damping[index]
                                 for index in select_modes]
                selected_stdmsh = np.zeros(
                    (self.modal_data.mode_shapes.shape[0], len(select_modes)), dtype=complex)
                
            if self.capabilities['MC']:
                selected_MC = [self.modal_data.modal_contributions[index]
                                 for index in select_modes]
                
            if self.capabilities['msh']:
                selected_modes = np.zeros(
                    (self.modal_data.mode_shapes.shape[0], len(select_modes)), dtype=complex)

                for num, ind in enumerate(select_modes):
                    row_index = ind[0]
                    col_index = ind[1]
                    mode_tmp = self.modal_data.mode_shapes[
                        :, col_index, row_index]
                    if self.capabilities['std']:
                        std_mode = self.modal_data.std_mode_shapes[
                            :, col_index, row_index]

                    # scaling of mode shape
                    abs_mode_tmp = np.abs(mode_tmp)
                    index_max = np.argmax(abs_mode_tmp)
                    this_max = mode_tmp[index_max]
                    
                    if not self.capabilities['std']:
                        mode_tmp = mode_tmp / this_max

                    selected_modes[:, num] = mode_tmp

                    if self.capabilities['std']:
                        selected_stdmsh[:, num] = std_mode
        else:
            return

        freq_str = ''
        damp_str = ''
        ord_str = ''

        if self.capabilities['msh']:
            msh_str = ''
            mpc_str = ''
            mp_str = ''
            mpd_str = ''
        if self.capabilities['std']:
            std_freq_str = ''
            std_damp_str = ''
            std_msh_str = ''
        if self.capabilities['MC']:
            MC_str = ''
            
        for col in range(len(select_modes)):
            freq_str += '{:3.3f} \t\t'.format(selected_freq[col])
            damp_str += '{:3.3f} \t\t'.format(selected_damp[col])
            ord_str += '{:3d} \t\t'.format(selected_order[col])

            if self.capabilities['msh']:
                mpc_str += '{:3.3f}\t \t'.format(selected_MPC[col])
                mp_str += '{:3.2f} \t\t'.format(selected_MP[col])
                mpd_str += '{:3.2f} \t\t'.format(selected_MPD[col])

            if self.capabilities['std']:
                std_damp_str += '{:3.3e} \t\t'.format(selected_stdd[col])
                std_freq_str += '{:3.3e} \t\t'.format(selected_stdf[col])
            
            if self.capabilities['MC']:
                MC_str += '{:3.3f} \t\t'.format(selected_MC[col])
                
        if self.capabilities['msh']:
            for row in range(selected_modes.shape[0]):
                msh_str += '\n           \t\t'
                if self.capabilities['std']:
                    std_msh_str += '\n           \t\t'
                for col in range(selected_modes.shape[1]):
                    msh_str += '{:+3.4f} \t'.format(selected_modes[row, col])
                    if self.capabilities['std']:
                        std_msh_str += '{:+3.3e} \t'.format(
                            selected_stdmsh[row, col])

        export_modes = 'MANUAL MODAL ANALYSIS\n'
        export_modes += '=======================\n'
        export_modes += 'Frequencies [Hz]:\t' + freq_str + '\n'
        if self.capabilities['std']:
            export_modes += 'Standard deviations of the Frequencies [Hz]:\t' + \
                std_freq_str + '\n'
        export_modes += 'Damping [%]:\t\t' + damp_str + '\n'
        if self.capabilities['std']:
            export_modes += 'Standard deviations of the Damping [%]:\t' + \
                std_damp_str + '\n'
        if self.capabilities['MC']:
            export_modes += 'Modal Contributions of the mode [-]:\t' + \
                MC_str + '\n'
        if self.capabilities['msh']:
            export_modes += 'Mode shapes:\t\t' + msh_str + '\n'
        if self.capabilities['std']:
            export_modes += 'Standard Deviations of the Mode shapes:\t\t' + \
                std_msh_str + '\n'
        export_modes += 'Model order:\t\t' + ord_str + '\n'
        if self.capabilities['msh']:
            export_modes += 'MPC [-]:\t\t' + mpc_str + '\n'
            export_modes += 'MP  [\u00b0]:\t\t' + mp_str + '\n'
            export_modes += 'MPD [-]:\t\t' + mpd_str + '\n\n'
        #              + 'SSI parameters\n'
        #              + '=======================\n'\
        #              + 'Maximum order :\t\t'     + str(self.modal_data.max_model_order) + '\n'\
        #              + 'Block rows :\t\t'        + str(self.num_block_rows)     + '\n'\
        #              + 'Block columns :\t\t'     + str(self.num_block_columns)  + '\n'
        #              + 'Decimation :\t\t'        + str(dec_fact)       + '\n'\
        #              + 'Filtering :\t\t'         + str(filt_w)

        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if binary:
            out_dict = {'selected_freq': selected_freq,
                        'selected_damp': selected_damp,
                        'selected_order': selected_order}

            if self.capabilities['msh']:
                out_dict['selected_MPC'] = selected_MPC
                out_dict['selected_MP'] = selected_MP
                out_dict['selected_MPD'] = selected_MPD
                out_dict['selected_modes'] = selected_modes

            if self.capabilities['std']:
                out_dict['selected_stdf'] = selected_stdf
                out_dict['selected_stdd'] = selected_stdd
                out_dict['selected_stdmsh'] = selected_stdmsh

            np.savez_compressed(fname, **out_dict)

        else:
            f = open(fname, 'w')
            f.write(export_modes)
            f.close()

    def calculate_stabilization_masks(self, order_range=None, d_range=None,
                                      stdf_max=None, stdd_max=None,
                                      mpc_min=None, mpd_max=None,  mtn_min=None,
                                      df_max=None, dd_max=None, dmac_max=None,
                                      dev_min=None, dmtn_min=None, MC_min=None):
        if self.state < 1:
            self.calculate_soft_critera_matrices()

        if order_range is None:
            order_range = (0, 1, self.modal_data.max_model_order)
        if d_range is None:
            d_range = (0, 100)
        if stdf_max is None:
            stdf_max = 100
        if stdd_max is None:
            stdd_max = 100
        if mpc_min is None:
            mpc_min = 0
        if mpd_max is None:
            mpd_max = 90
        if mtn_min is None:
            mtn_min = 0
        if df_max is None:
            df_max = 0.01
        if dd_max is None:
            dd_max = 0.05
        if dmac_max is None:
            dmac_max = 0.02
        if dev_min is None:
            dev_min = 0.02
        if dmtn_min is None:
            dmtn_min = 0.02
        if MC_min is None:
            MC_min = 0

        self.state = 2

        self.update_stabilization_masks(order_range, d_range, stdf_max, stdd_max,
                                        mpc_min, mpd_max, mtn_min, df_max,
                                        dd_max, dmac_max, dev_min, dmtn_min, MC_min)

    def update_stabilization_masks(self, order_range=None, d_range=None,
                                   stdf_max=None, stdd_max=None,
                                   mpc_min=None, mpd_max=None,  mtn_min=None,
                                   df_max=None, dd_max=None, dmac_max=None,
                                   dev_min=None, dmtn_min=None, MC_min=None):
        if self.state < 2:
            self.calculate_stabilization_masks()
        # update
        #print(order_range , d_range , stdf_max , stdd_max, mpc_min, mpd_max,  mtn_min,df_max, dd_max, dmac_max, dev_min, dmtn_min)
        if order_range is not None:
            self.order_range = order_range
        if d_range is not None:
            self.d_range = d_range
        if stdf_max is not None:
            self.stdf_max = stdf_max
        if stdd_max is not None:
            self.stdd_max = stdd_max
        if mpc_min is not None:
            self.mpc_min = mpc_min
        if mpd_max is not None:
            self.mpd_max = mpd_max
        if mtn_min is not None:
            self.mtn_min = mtn_min
        if df_max is not None:
            self.df_max = df_max
        if dd_max is not None:
            self.dd_max = dd_max
        if dmac_max is not None:
            self.dmac_max = dmac_max
        if dev_min is not None:
            self.dev_min = dev_min
        if dmtn_min is not None:
            self.dmtn_min = dmtn_min
        if MC_min is not None:
            self.MC_min = MC_min

        self.masked_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask

        mask_pre = self.masked_frequencies != 0

        if order_range is not None:
            start, step, stop = order_range
            start = max(0, start)
            stop = min(stop, self.modal_data.max_model_order)
            mask_order = np.zeros_like(mask_pre)
            for order in range(start, stop, step):
                mask_order = np.logical_or(
                    mask_order, self.order_dummy == order)
            mask_pre = np.logical_and(mask_pre, mask_order)

        self.masks['mask_pre'] = mask_pre

        if d_range is not None:
            assert isinstance(d_range, (tuple, list))
            assert len(d_range) == 2
            mask = np.logical_and(
                mask_pre, self.modal_data.modal_damping >= d_range[0])
            mask = np.logical_and(
                mask, self.modal_data.modal_damping <= d_range[1])
            self.masks['mask_ad'] = mask

        if stdf_max is not None and self.capabilities['std']:
            mask = self.modal_data.std_frequencies <= stdf_max * \
                self.modal_data.modal_frequencies
            mask = np.logical_and(mask_pre, mask)
            self.masks['mask_stdf'] = mask
            #import warnings
            #warnings.warn('Uncertainty bounds are not yet implemented! Ignoring!')

        if stdd_max is not None and self.capabilities['std']:
            mask = self.modal_data.std_damping <= stdd_max * \
                self.modal_data.modal_damping
            mask = np.logical_and(mask_pre, mask)
            self.masks['mask_stdd'] = mask
            #import warnings
            #warnings.warn('Uncertainty bounds are not yet implemented! Ignoring')

        if mpc_min is not None:
            mask = np.logical_and(mask_pre, self.MPC_matrix >= mpc_min)
            self.masks['mask_mpc'] = mask

        if mpd_max is not None:
            mask = np.logical_and(mask_pre, self.MPD_matrix <= mpd_max)
            self.masks['mask_mpd'] = mask

        if mtn_min is not None:
            import warnings
            warnings.warn(
                'Modal Transfer Norm is not yet implemented! Ignoring')
            
        if MC_min is not None and self.capabilities['MC']:
            mask = np.logical_and(mask_pre, self.modal_data.modal_contributions >= MC_min)
            self.masks['mask_MC'] = mask

        full_masks = []
        if df_max is not None:
            # rel freq diffs for each pole with all previous poles,
            # for all poles and orders results in 3d array
            # compare those rel freq diffs with df_max
            # and reduce 3d array to 2d array, by applying logical_or
            # along each poles axis (diff with all previous)
            mask_sf_all = np.logical_and(
                self.freq_diffs != 0, self.freq_diffs <= df_max)
            mask_sf_red = np.any(mask_sf_all, axis=2)
            # print(np.any(mask_sf_all))
            self.masks['mask_df'] = np.logical_and(mask_pre, mask_sf_red)
            full_masks.append(mask_sf_all)

        if dd_max is not None:
            mask_sd_all = np.logical_and(
                self.damp_diffs != 0, self.damp_diffs <= dd_max)
            mask_sd_red = np.any(mask_sd_all, axis=2)
            # print(np.any(mask_sd_all))
            self.masks['mask_dd'] = np.logical_and(mask_pre, mask_sd_red)
            full_masks.append(mask_sd_all)

        if dmac_max is not None:
            mask_sv_all = np.logical_and(
                self.MAC_diffs != 0, self.MAC_diffs <= dmac_max)
            mask_sv_red = np.any(mask_sv_all, axis=2)
            # print(np.any(mask_sv_all))
            self.masks['mask_dmac'] = np.logical_and(mask_pre, mask_sv_red)
            full_masks.append(mask_sv_all)

        if dev_min is not None:
            import warnings
            warnings.warn(
                'Eigenvalues are not available/implemented! Ignoring')

        if dmtn_min is not None:
            import warnings
            warnings.warn(
                'Modal Transfer Norm is not yet implemented! Ignoring')
            

        # check if all stability criteria are satisfied for all current poles
        if full_masks:
            stable_mask_full = np.ones_like(full_masks[0])
            for mask in full_masks:
                stable_mask_full = np.logical_and(stable_mask_full, mask)
            stable_mask = np.any(stable_mask_full, axis=2)
        else:
            stable_mask = mask_pre
        # print(np.any(stable_mask))
        self.masks['mask_stable'] = None
        for mask_name, mask in self.masks.items():
            if mask_name == 'mask_autosel':
                continue
            if mask_name == 'mask_autoclear':
                continue
            if mask is not None:
                stable_mask = np.logical_and(stable_mask, mask)
        self.masks['mask_stable'] = stable_mask

        # compute the only-unstable-in-... masks
        self.nmasks = {name: np.logical_not(
            stable_mask) for name, mask in self.masks.items() if mask is not None}

        for nname, nmask in self.nmasks.items():
            if nname in ['mask_pre', 'mask_stable', 'mask_autosel', 'mask_autoclear']:
                continue
            for name, mask in self.masks.items():
                if mask is None or name in ['mask_pre', 'mask_stable', 'mask_autosel', 'mask_autoclear']:
                    continue
                if name == nname:
                    nmask = np.logical_and(nmask, np.logical_not(mask))
                else:
                    nmask = np.logical_and(nmask, mask)
            self.nmasks[nname] = nmask

        self.nmasks['mask_stable'] = stable_mask
        self.nmasks['mask_pre'] = self.masks['mask_pre']
        self.nmasks['mask_autoclear'] = np.logical_not(
            self.masks.get('mask_autoclear', None))
        self.nmasks['mask_autosel'] = np.logical_not(
            self.masks.get('mask_autosel', None))

    def get_stabilization_mask(self, name):
        # print(name)
        mask = self.nmasks.get(name)

        if mask is None:
            mask = self.nmasks['mask_pre']
            print('!')

        return np.logical_not(mask)

    def get_max_f(self):
        if self.prep_data is not None:
            return self.prep_data.sampling_rate/2
        elif isinstance(self.modal_data, PogerSSICovRef):
            return self.modal_data.sampling_rate/2
        else:
            return float(np.amax(self.masked_frequencies))
            
        
    def get_modal_values(self, i):
        # needed for gui
        assert isinstance(i, (list, tuple))
        assert len(i) == 2
        assert i[0] <= self.modal_data.max_model_order
        assert i[1] <= self.num_solutions

        n = self.order_dummy[i]
        f = self.masked_frequencies[i]
        d = self.modal_data.modal_damping[i]

        if self.capabilities['msh']:
            mpc = self.MPC_matrix[i]
            mp = self.MP_matrix[i]
            mpd = self.MPD_matrix[i]
            MP_diffs = self.MP_diffs[i]
            #print(np.nonzero(MP_diffs)[0]) 
            if len(np.nonzero(MP_diffs)[0])>=1:
                dmp = np.min(MP_diffs[np.nonzero(MP_diffs)])
            else:
                dmp=0
            MPD_diffs = self.MPD_diffs[i]
            if len(np.nonzero(MPD_diffs)[0])>=1:
                dmpd = np.min(MPD_diffs[np.nonzero(MPD_diffs)])
            else:
                dmpd=0
        else:
            mpc = np.nan
            mp = np.nan
            mpd = np.nan
            dmp = np.nan
            dmpd = np.nan

        if self.capabilities['std']:
            stdf = self.modal_data.std_frequencies[i]
        else:
            stdf = np.nan

        if self.capabilities['std']:
            stdd = self.modal_data.std_damping[i]
        else:
            stdd = np.nan

        if self.capabilities['mtn']:
            mtn = np.nan
        else:
            mtn = np.nan
            
        if self.capabilities['MC']:
            MC = self.modal_data.modal_contributions[i]
        else:
            MC = np.nan
            
        if self.extra_func is not None:
            ex_1,ex_2 = self.extra_func(self.modal_data, i, True)
        else:
            ex_1, ex_2 = np.nan, np.nan
        
        return n,  f, stdf,  d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC, ex_1,ex_2

    def get_mode_shape(self, i):
        assert isinstance(i, (list, tuple))
        assert len(i) == 2
        assert i[0] <= self.modal_data.max_model_order
        assert i[1] <= self.num_solutions
        return self.modal_data.mode_shapes[:, i[1], i[0]]

    def save_state(self, fname):

        print('Saving results to  {}...'.format(fname))        
        
        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        out_dict = {'self.state':self.state}

        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time'] = self.start_time
        
        if self.state>=1:
            if self.capabilities['ev']:
                out_dict['self.lambda_diffs'] = np.array(self.lambda_diffs)
            out_dict['self.freq_diffs'] = np.array(self.freq_diffs)
            out_dict['self.damp_diffs'] = np.array(self.damp_diffs)
            
            if self.capabilities['msh']:
                out_dict['self.MAC_diffs'] = np.array(self.MAC_diffs)
                out_dict['self.MPD_diffs'] = np.array(self.MPD_diffs)
                out_dict['self.MP_diffs'] = np.array(self.MP_diffs)
            
                out_dict['self.MPC_matrix'] = np.array(self.MPC_matrix)
                out_dict['self.MP_matrix'] = np.array(self.MP_matrix)
                out_dict['self.MPD_matrix'] = np.array(self.MPD_matrix)

        if self.state>=2:
            out_dict['self.order_range'] = self.order_range
            out_dict['self.d_range'] = self.d_range
            if self.capabilities['std']:
                out_dict['self.stdf_max'] = self.stdf_max
                out_dict['self.stdd_max'] = self.stdd_max
            if self.capabilities['msh']:
                out_dict['self.mpc_min'] = self.mpc_min
                out_dict['self.mpd_max'] = self.mpd_max
                out_dict['self.mtn_min'] = self.mtn_min   
                 
            out_dict['self.df_max'] = self.df_max
            out_dict['self.dd_max'] = self.dd_max
            if self.capabilities['msh']:
                out_dict['self.dmac_max'] = self.dmac_max
            out_dict['self.dev_min'] = self.dev_min
            if self.capabilities['mtn']:
                out_dict['self.dmtn_min'] = self.dmtn_min
            if self.capabilities['MC']:
                out_dict['self.MC_min'] = self.MC_min
            out_dict['self.masks'] = self.masks
            out_dict['self.nmasks'] = self.nmasks
            
        if self.capabilities['auto']:
            if self.state>=3:                
                out_dict['self.num_iter'] = self.num_iter 
                out_dict['self.threshold'] = self.threshold
                out_dict['self.clear_ctr'] = self.clear_ctr  
            if self.state>=4:        
                out_dict['self.use_stabil'] = self.use_stabil
                out_dict['self.proximity_matrix_sq'] = self.proximity_matrix_sq
                out_dict['self.cluster_assignments'] = self.cluster_assignments
            if self.state>=5:
                out_dict['self.select_clusters'] = self.select_clusters  
                out_dict['self.nr_poles'] = self.nr_poles         
                out_dict['self.selection_cut_off'] = self.selection_cut_off
        
        out_dict['self.select_modes'] = self.select_modes

        np.savez_compressed(fname, **out_dict)

    @classmethod
    def load_state(cls, fname, modal_data, prep_data=None):
        print('Now loading previous results from  {}'.format(fname))

        in_dict = np.load(fname, allow_pickle=True)
        
        if 'self.state' in in_dict:
            state= float(in_dict['self.state'])
        else:
            return
        
        setup_name = str(in_dict['self.setup_name'].item())
        start_time = in_dict['self.start_time'].item()

        assert setup_name == modal_data.setup_name
        #assert start_time == modal_data.start_time

        stabil_data = cls(modal_data, prep_data)
        
        if state>=1:
            if stabil_data.capabilities['ev']:
                stabil_data.lambda_diffs = np.ma.array(in_dict['self.lambda_diffs'])
            stabil_data.freq_diffs = np.ma.array(in_dict['self.freq_diffs'])
            stabil_data.damp_diffs = np.ma.array(in_dict['self.damp_diffs'])
            
            if stabil_data.capabilities['msh']:
                stabil_data.MAC_diffs = np.ma.array(in_dict['self.MAC_diffs'])
                stabil_data.MPD_diffs = np.ma.array(in_dict['self.MPD_diffs'])
                stabil_data.MP_diffs = np.ma.array(in_dict['self.MP_diffs'])
                
                stabil_data.MPC_matrix = np.ma.array(in_dict['self.MPC_matrix'])
                stabil_data.MP_matrix = np.ma.array(in_dict['self.MP_matrix'])
                stabil_data.MPD_matrix = np.ma.array(in_dict['self.MPD_matrix'])
                
        if state >=2:
            stabil_data.order_range = tuple(in_dict['self.order_range'])
            stabil_data.d_range = tuple(in_dict['self.d_range'])
            if stabil_data.capabilities['std']:
                stabil_data.stdf_max = float(in_dict['self.df_max'])
                stabil_data.stdd_max = float(in_dict['self.stdd_max'])
            if stabil_data.capabilities['msh']:
                stabil_data.mpc_min = float(in_dict['self.mpc_min'])
                stabil_data.mpd_max = float(in_dict['self.mpd_max'])
                stabil_data.mtn_min = float(in_dict['self.mtn_min'])
            
            stabil_data.df_max = float(in_dict['self.df_max'])
            stabil_data.dd_max = float(in_dict['self.dd_max'])
            if stabil_data.capabilities['msh']:
                stabil_data.dmac_max = float(in_dict['self.dmac_max'])
            stabil_data.dev_min = float(in_dict['self.dev_min'])
            if stabil_data.capabilities['mtn']:
                stabil_data.dmtn_min = float(in_dict['self.dmtn_min'])
            if stabil_data.capabilities['MC']:
                stabil_data.MC_min = float(in_dict['self.MC_min'])
        
            stabil_data.masks = in_dict['self.masks'].item()
            stabil_data.nmasks = in_dict['self.nmasks'].item()
 
        if stabil_data.capabilities['auto']:
            if state>=3:                
                stabil_data.num_iter = int(in_dict['self.num_iter'])
                stabil_data.threshold = float(in_dict['self.threshold'])
                stabil_data.clear_ctr = in_dict['self.clear_ctr']
            if state>=4:
                stabil_data.use_stabil = bool(in_dict['self.use_stabil'])
                stabil_data.proximity_matrix_sq = in_dict['self.proximity_matrix_sq']
                stabil_data.cluster_assignments = in_dict['self.cluster_assignments']
            if state>=5:
                stabil_data.select_clusters = list(in_dict['self.select_clusters'])
                stabil_data.nr_poles = list(in_dict['self.nr_poles'])  
                stabil_data.selection_cut_off = float(in_dict['self.selection_cut_off'])

        select_modes = [tuple(a)
                                    for a in in_dict['self.select_modes']]
        frequencies =[stabil_data.masked_frequencies[index[0],index[1]] for index in select_modes]
        stabil_data.select_modes = [x for _,x in sorted(zip(frequencies, select_modes))]
        
        
        stabil_data.state = state
        return stabil_data


class StabilCluster(StabilCalc):

    def __init__(self, modal_data, prep_data=None):
        '''
        stab_* in %
        '''
        super().__init__(modal_data, prep_data)#
        
        assert self.capabilities['ev']
        
        self.num_iter = 20000
        
        self.weight_f = 1
        self.weight_MAC = 1
        self.weight_d = 1
        self.weight_lambda = 1
        self.threshold = None
        self.use_stabil = False

        """ The automatic modal analysis done in three stages clustering. 
        1st stage: values sorted according to their soft and hard criteria by a 2-means partitioning algorithm
        2nd stage: hierarchical clustering with automatic or user defined intercluster distance 
                   the automatic distance is based on the 'df', 'dd' and 'MAC' values from the centroids obtained in the first stage
                   ---------------------------------------------
                   | d = weight*df + 1 - weight*MAC + weight*dd |
                   ---------------------------------------------
        3rd stage: 2-means partitioning of the physical and spurious poles. 
        
        
        E. Neu et al.
        
        1. Identify mode candidates from a large number of system orders.
            -> OMA Algorithm with n_max sufficiently high, i.e. number of mathematical modes should exceed the number pf physical modes at n <= n_max
            
        2. Remove as many mathematical modes as possible.        
        (a) Remove certainly mathematical modes using hard validation criteria.
            Re(\lambda_n)>= 0 or Im(\lambda_n)==0-> remove conjugates in OMA algorithm
        (b) Split modes into consistent and non-consistent sets using k-means clustering.
            p_i = [d_lambda, d_f, d_zeta, 1-MAC, dMPD]
            power transformation eq 11
            h_Ti = ln(p_i)
            normalize: 
            h_Ni = (h_Ti - mean(h_Ti)) / std(h_Ti)
            initialize centroids with (+std(h_Ni), -std(h_Ni))
            
        
        3. Divide the remaining modes into homogeneous sets using hierarchical clustering.
        (a) Derive cutoff distance from the probability distribution of the consistent modes.
                np.percentile(a,95)
        (b) Cluster the mode candidates based on a complex distance measure.
                average linkage / single linkage 
        (c) Remove all but one mode from a single system order in one cluster.
                walk over each cluster and ensure each model order exists only once in the cluster, else remove the mode with a higher distance to the cluster center
                
        4. Remove the small sets, which typically consist of mathematical modes.
        (a) Reject sets that are smaller than a threshold derived from the largest set size.
            no recommendations given in paper (threshold 50 %)
        (b) Use outlier rejection to remove natural frequency and damping outliers.
            skip
        (c) Select a single mode representative from the remaining modes in each cluster.
            "multivariate" median
        
        """
    @staticmethod
    def decompress_flat_mask(compress_mask, flat_mask):
        # takes a flat mask generated on compressed data and restore it to its
        # decompressed form
        decompressed_mask = np.ma.copy(compress_mask.ravel())

        flat_index = 0
        for mask_index in range(decompressed_mask.shape[0]):
            if decompressed_mask[mask_index]:
                continue
            if flat_index >= len(flat_mask):
                decompressed_mask[mask_index] = True
            else:
                decompressed_mask[mask_index] = flat_mask[flat_index]
            flat_index += 1

        return decompressed_mask.reshape(compress_mask.shape)

    def plot_mask(self, mask, save_path=None):
        plot.figure(tight_layout=1)
        od_mask = np.copy(self.order_dummy.mask)
        mf_mask = np.copy(self.masked_frequencies.mask)
        self.order_dummy.mask = self.get_stabilization_mask('mask_pre')
        self.masked_frequencies.mask = self.get_stabilization_mask('mask_pre')
        plot.scatter(self.masked_frequencies.compressed(), self.order_dummy.compressed(
        ), marker='o', facecolors='none', edgecolors='grey', s=10)
        self.order_dummy.mask = mask
        self.masked_frequencies.mask = mask
        plot.scatter(self.masked_frequencies.compressed(), self.order_dummy.compressed(
        ), marker='o', facecolors='none', edgecolors='black', s=10)
        self.order_dummy.mask = od_mask
        self.masked_frequencies.mask = mf_mask
        plot.ylim((0, 200))
        plot.xlim((0, self.prep_data.sampling_rate / 2))
        plot.xlabel('Frequency [Hz]')
        plot.ylabel('Model Order ')
        plot.tight_layout()
        if save_path:
            plot.savefig(save_path + 'mask.pdf')
        else:
            plot.show()
            plot.pause(0.001)
            
    def automatic_clearing(self, num_iter=None):
        if self.state < 2:
            self.calculate_soft_critera_matrices()
        print('Clearing physical modes automatically...')
        # 2-means clustering of all poles by all available criteria
        # algorithm minimizes euclidian distances
        
        if num_iter is not None: 
            assert isinstance(num_iter, int)
            assert num_iter > 0
            self.num_iter = num_iter
        
        # represent all the vectors by their soft criteria : 
        # [index, i, d_lambda, d_f, d_xi, dMAC, dMPD]
        
        mask_pre = np.ma.array(self.get_stabilization_mask('mask_pre'))
        # in a second run mask_pre is itself masked        
 
        self.freq_diffs.mask = np.ma.nomask
        self.damp_diffs.mask = np.ma.nomask
        self.lambda_diffs.mask = np.ma.nomask
        if self.capabilities['msh']:
            self.MAC_diffs.mask = np.ma.nomask
            #self.MPD_diffs.mask = np.ma.nomask
            self.MP_diffs.mask = np.ma.nomask
 
        # assuming there are no frequencies equal within given precision
        mask_pre_3d = self.freq_diffs == 0
 
        soft_criteria_matrices = []
        for matrix in [self.lambda_diffs, self.freq_diffs, self.damp_diffs]:
            matrix.mask = mask_pre_3d
            soft_criteria_matrices.append(matrix.min(axis=2))
                
        if self.capabilities['msh']:
            self.MAC_diffs.mask = mask_pre_3d
            soft_criteria_matrices.append(self.MAC_diffs.min(axis=2))
            
            #self.MPD_diffs.mask = mask_pre_3d
            #soft_criteria_matrices.append(self.MPD_diffs.min(axis=2))
            
            self.MP_diffs.mask = mask_pre_3d
            soft_criteria_matrices.append(self.MP_diffs.min(axis=2))
         
        for matrix in soft_criteria_matrices:
            matrix.mask = mask_pre

        # flatten unmasked values and remove first two model orders
        compressed_matrices = [matrix[2:, :].compressed()
                               for matrix in soft_criteria_matrices]
        
        # dlambda, df, dd, dMAC, dMPD, stacked as list of size (order, num_modes)
        all_poles = np.vstack(compressed_matrices).T
        
        #transform distribution (weibull like) to logarithmic scale (resembles normal distribution)
        all_poles = np.log(all_poles)

        # whitening (scale to unit variance) significantly improves 
        # convergence rates of the kmeans algorithm
        mean_all_poles = np.mean(all_poles, axis=0)
        std_all_poles = np.std(all_poles, axis=0)
        all_poles -= mean_all_poles
        all_poles /= std_all_poles
        
        std_dev = np.std(all_poles, axis=0)
        ideal_physical_values = -std_dev
        ideal_spurious_values = std_dev

        # the k-means algorithm is sensitive to the initial starting 
        # values in order to converge to a solution
        # therefore two starting attempts are introduced
        ctr_init = np.array([ideal_physical_values,
                             ideal_spurious_values])
 
        # masked arrays are not supported by scipy's kmeans algorithm
        # all_poles an M by N array where the rows are observation vectors
        self.clear_ctr, idx = scipy.cluster.vq.kmeans2(
            all_poles, ctr_init, self.num_iter)


        print('Possibly physical poles 1st stage: {0}\nSpurious poles 1st stage: {1}'.format(
            collections.Counter(idx)[0], collections.Counter(idx)[1]))
        
        # add unmasked values of the first two model orders that were
        # previously not considered
        mask_pre.mask = np.ma.nomask
        new_idx = np.hstack(
            (np.ones(np.sum(np.logical_not(mask_pre[:2, :]))), idx))
        mask_autoclear = self.decompress_flat_mask(mask_pre, new_idx)
 
        # re-apply mask_pre, should not be necessary
        mask_autoclear = np.logical_or(mask_autoclear, mask_pre)
     
        # apply the hard validation criteria
#         mask_autoclear = np.logical_or(
#             mask_autoclear, self.modal_data.modal_damping < 0.001)
#         mask_autoclear = np.logical_or(
#             mask_autoclear, self.modal_data.modal_damping > 20)
        
        # compute the threshold as the 95th percentile of
        # P(threshold > d_lambda + d_MAC) = 0.95
        soft_criteria_matrices[0].mask = np.ma.nomask
        soft_criteria_matrices[3].mask = np.ma.nomask
        distance_mat = soft_criteria_matrices[0]+soft_criteria_matrices[3]
        distance_mat.mask = mask_autoclear
        self.threshold = np.percentile(distance_mat.compressed(),q=95)
        
        self.masks['mask_autoclear'] = mask_autoclear
        self.update_stabilization_masks()
        
        self.state = 3

    def automatic_classification(self, threshold=None, use_stabil=False):
        if self.state < 3 and not use_stabil:
            self.automatic_clearing()
        print('Classifying physical modes automatically...')

        if use_stabil:
            mask_autoclear = self.get_stabilization_mask('mask_stable')
            print(np.sum(mask_autoclear))
            #self.masks['mask_autoclear'] = mask_autoclear
            #self.update_stabilization_masks()
            #print(123)
        else:
            mask_autoclear = self.get_stabilization_mask('mask_autoclear')
        
        self.use_stabil = use_stabil
        
        if threshold is not None:
            assert isinstance(threshold, int)
            self.threshold = threshold
        if self.threshold is None:
            self.freq_diffs.mask = np.ma.nomask
            mask_pre_3d = self.freq_diffs == 0        
            self.lambda_diffs.mask = mask_pre_3d
            self.MAC_diffs.mask = mask_pre_3d
            distance_mat = self.lambda_diffs.min(axis=2)\
                           + self.MAC_diffs.min(axis=2)
            distance_mat.mask = mask_autoclear
            self.threshold = np.percentile(distance_mat.compressed(),q=95)
        #print(self.threshold)

        length_mat = np.product(mask_autoclear.shape) \
                     - np.sum(mask_autoclear)

        self.masked_lambda.mask = mask_autoclear
        lambda_compressed = self.masked_lambda.compressed()
        self.masked_lambda.mask = np.ma.nomask

        dim0, dim1 = mask_autoclear.shape
        mode_shapes_compressed = np.zeros(
            (self.modal_data.mode_shapes.shape[0], length_mat), 
            dtype=np.complex128
            )
        
        n = 0
        for i in range(dim0):
            for j in range(dim1):
                if mask_autoclear[i, j]:
                    continue
                this_msh = self.modal_data.mode_shapes[:, j, i]
                mode_shapes_compressed[:, n] = this_msh
                n += 1

        l = len(lambda_compressed)
        #print(l)
        div_lambda = np.maximum(
            np.repeat(np.expand_dims(np.abs(lambda_compressed), axis=1), 
                      lambda_compressed.shape[0], axis=1),
            np.repeat(np.expand_dims(np.abs(lambda_compressed), axis=0), 
                      lambda_compressed.shape[0], axis=0)
            )
        
        lambda_proximity_matrix = np.abs(
            lambda_compressed - lambda_compressed.reshape((l, 1)))/div_lambda
            
        mac_proximity_matrix = 1 - \
            self.calculateMAC(mode_shapes_compressed, mode_shapes_compressed)

        proximity_matrix = self.weight_lambda * lambda_proximity_matrix \
                        + self.weight_MAC * mac_proximity_matrix
        
        # correct round off errors
        proximity_matrix[
            proximity_matrix < np.finfo(proximity_matrix.dtype).eps] = 0
        
        self.proximity_matrix_sq = scipy.spatial.distance.squareform(
            proximity_matrix, checks=False)
        linkage_matrix = scipy.cluster.hierarchy.linkage(
            self.proximity_matrix_sq, method='average')
        self.cluster_assignments = scipy.cluster.hierarchy.fcluster(
            linkage_matrix, self.threshold, criterion='distance')

        for clusternr in range(1, max(self.cluster_assignments)+1):
            flat_poles_ind = self.cluster_assignments != clusternr + 1
            mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
            self.order_dummy.mask = mask
            for order in range(self.modal_data.max_model_order):
                if np.sum(self.order_dummy==order)>1:
                    print('Double Model Order: ', self.order_dummy[order,:])
        
        self.order_dummy.mask = np.ma.nomask
        
        print('Number of classified clusters: {}'.format(
            max(self.cluster_assignments)))
        self.state = 4


    def automatic_selection(self, number=0):    
            
        if self.state < 4:
            self.automatic_classification()
            
        # count clusters with more than a fraction of the number of elements of
        # the largest cluster and add that many zero size clusters
        poles = []

        for cluster_ in range(1, 1 + max(self.cluster_assignments)):
            poles.append(np.where(self.cluster_assignments == cluster_))

        nr_poles = [len(a[0]) for a in poles]
        max_nr = max(nr_poles)
        # enlarge list of poles to improve clustering
#         for nr in nr_poles:
#             if nr > max_nr * fraction:
#                 nr_poles.append(0)
#             elif nr < max_nr * fraction:
#                 continue
        nr_poles = np.array(nr_poles, dtype=np.float64)

        if number == 0:
            # 2-means clustering of the number-of-poles, return indices; 
            # split into two clusters
            _, select_clusters = scipy.cluster.vq.kmeans2(
                np.array(nr_poles, dtype=np.float64), 
                np.array([max_nr, 1e-12]), self.num_iter)
        else:
            meta_list = list(enumerate(nr_poles))
            
            sorted_meta_list = sorted(meta_list, key=itemgetter(1), 
                                      reverse=True)
            select_clusters = [1 for p in nr_poles]
            
            for i in range(number):
                ind = sorted_meta_list[i][0]
                select_clusters[ind]=0


        print('Number of physical modes: {0}'.format(
            collections.Counter(select_clusters)[0]))
        self.select_clusters = select_clusters
        self.nr_poles = nr_poles

        self.selection_cut_off = np.Infinity
        for i, b in zip(self.nr_poles, self.select_clusters):
            if not b:
                self.selection_cut_off = min(i - 1, self.selection_cut_off)
        print('Minimum number of elements in retained clusters: {}'.format(
            self.selection_cut_off))
        
        if self.use_stabil:
            mask_autoclear = self.get_stabilization_mask('mask_stable')
            #mask_autoclear = self.masks['mask_autoclear']
        else:
            mask_autoclear = self.masks['mask_autoclear']

        self.MAC_diffs.mask = self.MAC_diffs == 0
        MAC_diffs = self.MAC_diffs.min(axis=2)
        self.MAC_diffs.mask = np.ma.nomask
        if 'mask_autosel' not in self.masks:
            self.masks['mask_autosel'] = []
        for clusternr, inout in enumerate(select_clusters):

            if inout:
                continue

            flat_poles_ind = self.cluster_assignments != clusternr + 1

            mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
            self.masks['mask_autosel'].append(np.ma.copy(mask))

            # remove outermost values in all criteria, until only the 
            # "multi-variate median" is left this pole is selected as the 
            # representative solution for this cluster
            num_poles_left = np.product(mask.shape) - np.sum(mask)
            
            #print(clusternr, num_poles_left)
            
            while num_poles_left > 1:
                ind = []
                for matrix, target in zip(
                    [self.masked_frequencies, self.masked_damping,],
                    [np.ma.median,             np.ma.median,]):
                    
                    matrix.mask = mask

                    val = target(matrix)
                    min_ = np.min(matrix)
                    max_ = np.max(matrix)
                    
                    if val - min_ <= max_ - val:
                        ind.append(np.where(matrix == max_))
                    else:
                        ind.append(np.where(matrix == min_))

                for i in range(min(len(ind), num_poles_left - 1)):
                    this_ind = ind[i]
                    mask[this_ind] = True

                num_poles_left = np.product(mask.shape) - np.sum(mask)
                
                #print(clusternr, num_poles_left, len(ind))
            
            select_mode = np.where(np.logical_not(mask))
            self.select_modes.append((select_mode[0][0], select_mode[1][0]))
            if self.select_callback is not None:
#                 for matrix in [self.masked_frequencies, self.masked_damping, 
#                            MAC_diffs, self.MPC_matrix, self.MPD_matrix]:
#                 
#                     matrix.mask = np.ma.nomask
                self.select_callback(self.select_modes[-1])

        for matrix in [self.masked_frequencies, self.masked_damping, 
                       MAC_diffs, self.MPC_matrix, self.MPD_matrix]:
            
            matrix.mask = np.ma.nomask

        self.state = 5

    def plot_clearing(self, save_path=None):

        mask_autoclear = self.masks['mask_autoclear']
        mask_pre = self.get_stabilization_mask('mask_pre')
        self.plot_mask(mask_autoclear, save_path)

        self.freq_diffs.mask = self.freq_diffs == 0
        freq_diffs = self.freq_diffs.min(axis=2)
        self.freq_diffs.mask = np.ma.nomask

        self.MAC_diffs.mask = self.MAC_diffs == 0
        MAC_diffs = self.MAC_diffs.min(axis=2)
        self.MAC_diffs.mask = np.ma.nomask

        self.damp_diffs.mask = self.damp_diffs == 0
        damp_diffs = self.damp_diffs.min(axis=2)
        self.damp_diffs.mask = np.ma.nomask

        crits = [freq_diffs, damp_diffs, MAC_diffs,
                 self.MPC_matrix, self.MPD_matrix]
        labels = ['df', 'dd', 'MAC', 'MPC', 'MPD']

        new_crits = []

        for j, b in enumerate(crits):
            new_crits.append(b)
            for i, a in enumerate(new_crits):
                if a is b:
                    continue
                plot.figure(tight_layout=1)

                a.mask = mask_autoclear
                b.mask = mask_autoclear

                plot.plot(a.compressed(), b.compressed(), ls='', marker=',')
                plot.plot(
                    np.mean(a), np.mean(b), ls='', marker='d', color='black')

                a.mask = mask_pre
                b.mask = mask_pre
                plot.plot(
                    a.compressed(), b.compressed(), ls='', marker=',', color='grey')
                plot.plot(
                    np.mean(a), np.mean(b), ls='', marker='d', color='grey')

                labela = labels[i]
                labelb = labels[j]

                plot.xlabel(labela)
                plot.ylabel(labelb)

                plot.xlim((0, 1))
                plot.ylim((0, 1))

                if save_path is not None:
                    plot.savefig(
                        save_path + 'clear_{}_{}.pdf'.format(labela, labelb))
                else:
                    # print('show')
                    plot.show()
                    plot.pause(0.01)
        for crit in crits:
            crit.mask = np.ma.nomask

    def plot_classification(self, save_path=None):
        rel_matrix = scipy.cluster.hierarchy.linkage(
            self.proximity_matrix_sq, method='average')
        lvs = scipy.cluster.hierarchy.leaves_list(rel_matrix)

        def _llf(id):
            if len(lvs) > 500:
                if (np.where(id == lvs)[0][0] % 100 == 0):
                    return str(np.where(id == lvs)[0][0])
                else:
                    return str('')
            else:
                if (np.where(id == lvs)[0][0] % 10 == 0):
                    return str(np.where(id == lvs)[0][0])
                else:
                    return str('')
        fig = plot.figure(tight_layout=1)
        ax = fig.add_subplot(111)
        scipy.cluster.hierarchy.dendrogram(
            rel_matrix, leaf_label_func=_llf, color_threshold=self.threshold, leaf_font_size=16, leaf_rotation=40)
        ax = plot.gca()
        ax.set_xlabel('Mode number [-]')
        ax.set_ylabel('Distance [-]')
        ax.axhline(self.threshold, c='r', ls='--', linewidth=3)
        plot.tight_layout()
        if save_path is not None:
            plot.savefig(save_path + 'dendrogram.pdf')
        else:
            # print('show')
            plot.show()
            plot.pause(0.001)

    def plot_selection(self, save_path=None):
        """ Plot relevant results of the clustering."""

        plot.figure(tight_layout=1)

        in_poles = list(self.nr_poles[self.nr_poles >= self.selection_cut_off])
        in_poles.sort(reverse=True)
        out_poles = self.nr_poles[self.nr_poles < self.selection_cut_off]
        out_poles = list(out_poles[out_poles > 0])
        out_poles.sort(reverse=True)
        plot.bar(range(len(in_poles)), in_poles, facecolor='red',
                 edgecolor='none', align='center',)
        # print(list(range(len(in_poles),len(self.nr_poles))),out_poles)
        plot.bar(range(len(in_poles), len(in_poles) + len(out_poles)),
                 out_poles, facecolor='blue', edgecolor='none', align='center',)

        plot.xlim((0, len(self.nr_poles)))
        plot.tight_layout()

        if save_path is not None:
            plot.savefig(save_path + 'cluster_sizes.pdf')
        else:
            plot.show()
            plot.pause(0.001)

        fig = plot.figure(tight_layout=1)
        ax1 = fig.add_subplot(211)

        mask_autoclear = self.masks['mask_autoclear']
        mask_pre = self.get_stabilization_mask('mask_pre')
        mask_pre_ = np.logical_not(
            np.logical_and(np.logical_not(mask_pre), mask_autoclear))

        self.order_dummy.mask = mask_pre_
        self.masked_frequencies.mask = mask_pre_
        ax1.scatter(self.masked_frequencies.compressed(), self.order_dummy.compressed(
        ), marker='o', facecolors='none', edgecolors='grey', s=10, label='pole')

        self.order_dummy.mask = mask_autoclear
        self.masked_frequencies.mask = mask_autoclear
        ax1.scatter(self.masked_frequencies.compressed(), self.order_dummy.compressed(
        ), marker='o', facecolors='none', edgecolors='black', s=10, label='stable pole')

        self.order_dummy.mask = np.ma.nomask
        self.masked_frequencies.mask = np.ma.nomask

        ax1.autoscale_view(tight=True)
        #ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Model order [-]')
        ax1.set_title('Stabilization Diagram')
        ax1.set_ylim((0, 200))

        for mask in self.masks['mask_autosel']:
            self.masked_frequencies.mask = mask
            plot.axvspan(self.masked_frequencies.min(), self.masked_frequencies.max(
            ), facecolor='blue', alpha=.3, edgecolor='none')
            #print(self.masked_frequencies.min(), self.masked_frequencies.max(),np.ma.mean(self.masked_frequencies))

        self.masked_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask

        for mode in self.select_modes:
            f = self.modal_data.modal_frequencies[mode]
            n = self.order_dummy[mode]
            ax1.scatter(
                f, n, facecolors='none', marker='o', edgecolors='red', s=10)

        num_poles = []
        fpoles = []
        for clusternr in range(1, 1 + max(self.cluster_assignments)):

            flat_poles_ind = self.cluster_assignments != clusternr
            mask = self.decompress_flat_mask(mask_autoclear, flat_poles_ind)
            self.masked_frequencies.mask = mask
            num_poles.append(np.product(mask.shape) - np.sum(mask))
            # print(np.ma.mean(self.masked_frequencies))
            fpoles.append(np.ma.mean(self.masked_frequencies))

        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.bar(fpoles, num_poles, width=0.01,
                align='center', edgecolor='none')

        ax2.axhline(self.selection_cut_off, c='r', ls='--', linewidth=2)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Nr. of elements')
        ax2.set_title('Clusters')
        plot.tight_layout()
        plot.xlim((0, self.prep_data.sampling_rate / 2))
        #plot.savefig('Main_plot_clusters_' + self.timestamp + '.' + self.format_plot, format=self.format_plot)
        if save_path is not None:
            plot.savefig(save_path + 'select_clusters.pdf')
        else:
            # print('show')
            plot.show()
            plot.pause(0.001)
        # plot.show(block=False)

    def return_results(self):        
        
        all_f = []
        all_d = []
        all_n = []
        all_std_f = []
        all_std_d = []
        all_MPC = []
        all_MPD = []
        all_MP = []
        all_msh = []
        all_MC = []

        #for select_mode, mask in zip(self.select_modes, self.masks['mask_autosel']):
        for select_mode in self.select_modes:
            
            n,  f, stdf,  d, stdd, mpc, mp, mpd, dmp, dmpd, mtn, MC,ex_1,ex_2 = self.get_modal_values(select_mode)
            msh= self.get_mode_shape(select_mode)
            
            all_n.append(n)
            all_f.append(f)
            all_std_f.append(stdf)
            all_d.append(d)
            all_std_d.append(stdd)
            all_MPC.append(mpc)
            all_MP.append(mp)
            all_MPD.append(mpd)
            all_MC.append(MC)
            all_msh.append(msh)
            
            continue
            
#             self.masked_frequencies.mask = np.ma.nomask
#             select_f = self.masked_frequencies[select_mode]
# 
#             self.masked_damping.mask = np.ma.nomask
#             select_d = self.masked_damping[select_mode]
# 
#             select_order = self.order_dummy[select_mode]
# 
#             if self.capabilities['msh']:
#                 self.MPC_matrix.mask = np.ma.nomask
#                 select_MPC = self.MPC_matrix[select_mode]
#     
# #                 self.MP_matrix.mask = np.ma.nomask
# #                 select_MP = self.MP_matrix[select_mode]
#     
#                 self.MPD_matrix.mask = np.ma.nomask
#                 select_MPD = self.MPD_matrix[select_mode]
#     
#                 select_msh = self.modal_data.mode_shapes[
#                     :, select_mode[1], select_mode[0]]
# 
#                 # scaling of mode shape
#                 #select_msh /= select_msh[np.argmax(np.abs(select_msh))][0]
#             if self.capabilities['MC']:
#                 self.modal_data.modal_contributions[select_mode]
# 
#             all_f.append(float(select_f))
#             all_d.append(float(select_d))
#             all_n.append(float(select_order))
#             all_MPC.append(float(select_MPC))
# #             all_MP.append(float(select_MP))
#             all_MPD.append(float(select_MPD))
#             all_msh.append(select_msh)
            
            
        return np.array(all_n),np.array(all_f),np.array(all_std_f),np.array(all_d),np.array(all_std_d),np.array(all_MPC),np.array(all_MP),np.array(all_MPD),np.array(all_MC),np.array(all_msh),


class StabilPlot(object):

    def __init__(self, stabil_calc):
        '''
        stab_* in %
        '''
        super().__init__()

        assert isinstance(stabil_calc, StabilCalc)
        self.stabil_calc = stabil_calc

        self.fig = Figure(facecolor='white', dpi=100, figsize=(16, 12))
        self.fig.set_tight_layout(True)
        self.ax = self.fig.add_subplot(111)
        
#         self.ax2 = self.ax.twinx()

        canvas = FigureCanvasBase(self.fig)

        # if self.fig.canvas:
        if False:
            self.init_cursor()
        else:
            self.cursor = None
        marker_obj_1 = MarkerStyle('o')
        path_1 = marker_obj_1.get_path().transformed(
            marker_obj_1.get_transform())
        marker_obj_2 = MarkerStyle('+')
        path_2 = marker_obj_2.get_path().transformed(
            marker_obj_2.get_transform())
        path_stab = Path.make_compound_path(path_1, path_2)

        marker_obj_2 = MarkerStyle('x')
        path_2 = marker_obj_2.get_path().transformed(
            marker_obj_2.get_transform())
        path_auto = Path.make_compound_path(path_1, path_2)

        fp = FontProperties(family='monospace', weight=0, size='large')
        

        self.psd_plot = None
        
        self.stable_plot = {
            'plot_pre':   None,  
            #'plot_ad':    None,
            #'plot_df':    None,
            #'plot_dd':    None,
            'plot_stable':   None,
        }
        
        self.colors = {
            'plot_pre':   'grey', 
            #'plot_ad':    'grey',
            #'plot_df':    'black',
            #'plot_dd':    'black',
            'plot_stable':   'black',
        }



        self.markers = {
            'plot_pre':   'o',  
            #'plot_ad':    TextPath((-2, -4), '\u00b7 d', prop=fp, size=10),
            #'plot_df':    TextPath((-2, -4), '\u00b7 f', prop=fp, size=10),
            #'plot_dd':    TextPath((-2, -4), '\u00b7 d', prop=fp, size=10),
            'plot_stable':   path_stab,  
            #
        }

        self.labels = {
            'plot_pre':   'all poles',
            #'plot_ad':    'damping criterion', 
            #'plot_df':    'unstable in frequency', 
            #'plot_dd':    'unstable in damping',  
            'plot_stable':   'stable poles',  
            
        }
        
        if self.stabil_calc.capabilities['std']:
            self.stable_plot['plot_stdf'] = None  # uncertainty frequency
            self.stable_plot['plot_stdd'] = None  # uncertainty damping
            
            self.colors['plot_stdf'] = 'grey'
            self.colors['plot_stdd'] = 'grey'
            
            self.labels['plot_stdf'] = 'uncertainty bounds frequency criterion'
            self.labels['plot_stdd'] = 'uncertainty bounds damping criterion'
            
            self.markers['plot_stdf'] = 'd'
            self.markers['plot_stdd'] = 'd'
        if self.stabil_calc.capabilities['msh']:
            self.stable_plot['plot_mpc'] = None  # absolute modal phase collineratity
            self.stable_plot['plot_mpd'] = None  # absolute mean phase deviation
            self.stable_plot['plot_dmac'] = None  # difference mac
            
            #self.colors['plot_mpc'] = 'grey'
            #self.colors['plot_mpd']=  'grey'
            #self.colors['plot_dmac']=  'black'

            #self.labels['plot_mpc']=   'modal phase collinearity criterion'
            #self.labels['plot_mpd']=   'mean phase deviation criterion'
            #self.labels['plot_dmac']=  'unstable in mac'
            
            #self.markers['plot_mpc']=   TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)
            #self.markers['plot_mpd']=   TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)
            #self.markers['plot_dmac']=  TextPath((-2, -4), '\u00b7 v', prop=fp, size=10)
            
        if self.stabil_calc.capabilities['auto']:
            self.stable_plot['plot_autoclear']= None  # auto clearing by 2Means Algorithm
            self.stable_plot['plot_autosel']= None # autoselection by 2 stage hierarchical clustering
            
            self.colors['plot_autoclear']= 'black'
            self.colors['plot_autosel']= 'rainbow'
            
            self.labels['plot_autoclear']=   'autoclear poles'
            self.labels['plot_autosel']= 'autoselect poles'
            
            self.markers['plot_autoclear']=   path_auto
            self.markers['plot_autosel']= 'o'
            
        if self.stabil_calc.capabilities['MC']:
            self.stable_plot['plot_MC']=  None  # absolute modal error contribution
            self.colors['plot_MC']=  'grey'
            
            self.labels['plot_MC']=   'modal error contribution criterion'
            
            self.markers['plot_MC']=   'x'
            
        if self.stabil_calc.capabilities['mtn']:
            self.stable_plot['plot_dmtn']=  None  # difference modal transfer norm
            self.stable_plot['plot_mtn']=  None  # absolute modal transfer norm
            
            self.colors['plot_dmtn']=  'black'
            self.colors['plot_mtn']=  'grey'
            
            self.labels['plot_mtn']=   'modal transfer norm criterion'
            self.labels['plot_dmtn']=  'unstable in modal transfer norm'
            
            self.markers['plot_mtn']=   '>'
            self.markers['plot_dmtn']=  '>'
        if False:
            self.stable_plot['plot_dev']=   None  # difference eigenvalue
            
            self.colors['plot_dev']=   'grey'
            
            self.labels['plot_dev']=   'unstable in eigenvalue'

            self.markers['plot_dev']=   TextPath((-2, -4), '\u00b7 \u03bb', prop=fp, size=10),
    
        
        self.zorders = {key: key != 'plot_pre' for key in self.labels.keys()}
        self.zorders['plot_autosel'] = 2
        self.sizes = {key: 30 for key in self.labels.keys()}

        self.prepare_diagram()

    def init_cursor(self):
        #print(self.stabil_calc.select_modes, type(self.stabil_calc.select_modes))
        self.cursor = DataCursor(ax=self.ax, order_data=self.stabil_calc.order_dummy,
                                 f_data=self.stabil_calc.masked_frequencies, datalist=self.stabil_calc.select_modes,
                                 color='black')

        self.fig.canvas.mpl_connect('button_press_event', self.cursor.onmove)
        self.fig.canvas.mpl_connect('resize_event', self.cursor.fig_resized)
        # self.cursor.add_datapoints(self.stabil_calc.select_modes)
        self.stabil_calc.select_callback = self.cursor.add_datapoint

    def prepare_diagram(self):

        self.ax.set_ylim((0, self.stabil_calc.modal_data.max_model_order))
        self.ax.locator_params(
            'y', tight=True, nbins=self.stabil_calc.modal_data.max_model_order // 5)
        x_lims = (0, self.stabil_calc.get_max_f())
        self.ax.set_xlim(x_lims)
        self.ax.autoscale_view(tight=True)
        self.ax.set_xlabel('Frequency [Hz]')
        self.ax.set_ylabel('Model Order')

    def update_stabilization(self, **criteria):

        # print(criteria)
        # recalculate stabilization masks
        self.stabil_calc.update_stabilization_masks(**criteria)

        # update stabil plots if values have changed
        #if 'd_range' in criteria:
        #    self.plot_stabil('plot_ad')
        if 'stdf_max' in criteria and self.stabil_calc.capabilities['std']:
            self.plot_stabil('plot_stdf')
        if 'stdd_max' in criteria and self.stabil_calc.capabilities['std']:
            self.plot_stabil('plot_stdd')
        #if 'mpc_min' in criteria and self.stabil_calc.capabilities['msh']:
        #    self.plot_stabil('plot_mpc')
        #if 'mpd_max' in criteria and self.stabil_calc.capabilities['msh']:
        #    self.plot_stabil('plot_mpd')
        if 'mtn_min' in criteria and self.stabil_calc.capabilities['mtn']:
            self.plot_stabil('plot_mtn')
        #if 'df_max' in criteria:
        #    self.plot_stabil('plot_df')
        #if 'dd_max' in criteria:
        #    self.plot_stabil('plot_dd')
        #if 'dmac_max' in criteria and self.stabil_calc.capabilities['msh']:
        #    self.plot_stabil('plot_dmac')
        if 'dev_min' in criteria and False:
            self.plot_stabil('plot_dev')
        if 'dmtn_min' in criteria and self.stabil_calc.capabilities['mtn']:
            self.plot_stabil('plot_dmtn')
        if 'MC_min' in criteria and self.stabil_calc.capabilities['MC']:
            self.plot_stabil('plot_MC')
            
        self.plot_stabil('plot_pre')
        self.plot_stabil('plot_stable')

        if self.stabil_calc.capabilities['auto']:
            if self.stabil_calc.state >= 3 and not self.stabil_calc.use_stabil:
                self.plot_stabil('plot_autoclear')
            if self.stabil_calc.state >= 5:
                self.plot_stabil('plot_autosel')

        if self.stabil_calc.capabilities['std']:
            self.plot_stabil('plot_stdf')
        # update the cursors snap mask
        if self.cursor:
            cursor_name_mask = self.cursor.name_mask
            cursor_mask = self.stabil_calc.get_stabilization_mask(
                cursor_name_mask)
            self.cursor.set_mask(cursor_mask, cursor_name_mask)

    def plot_stabil(self, name):
        # print(name)
        color = self.colors[name]
        marker = self.markers[name]
        #print(marker, name)
        zorder = self.zorders[name]
        size = self.sizes[name]
        label = self.labels[name]

        #print(name, color)

        if name == 'plot_autosel':
            if self.stable_plot[name] is not None:
                for plot in self.stable_plot[name]:
                    plot.remove()

            visibility = True
            masks = self.stabil_calc.masks['mask_autosel']
            # @UndefinedVariable
            colors = list(
                matplotlib.cm.gist_rainbow(np.linspace(0, 1, len(masks))))#@UndefinedVariable
            shuffle(colors)
            self.stable_plot[name] = []
            for color, mask in zip(colors, masks):
                self.stabil_calc.masked_frequencies.mask = mask
                self.stabil_calc.order_dummy.mask = mask
                self.stable_plot[name].append(self.ax.scatter(self.stabil_calc.masked_frequencies.compressed(),
                                                              self.stabil_calc.order_dummy.compressed(), zorder=zorder,
                                                              facecolors=color, edgecolors='none',
                                                              marker=marker, alpha=0.4,
                                                              s=size, label=label, visible=visibility))

        elif name == 'plot_stdf':

            if self.stable_plot[name] is not None:
                try:
                    visibility = self.stable_plot[name][1][0].get_visible()
                    self.stable_plot[name][1][0].remove()
                    self.stable_plot[name][1][1].remove()
                    self.stable_plot[name][2][0].remove()
                except IndexError:
                    visibility = True

            else:
                visibility = True
            mask = self.stabil_calc.get_stabilization_mask('mask_stable')
            self.stabil_calc.masked_frequencies.mask = mask
            self.stabil_calc.order_dummy.mask = mask
            if self.stabil_calc.capabilities['std']:
                std_frequencies = np.ma.array(
                    self.stabil_calc.modal_data.std_frequencies)
                std_frequencies.mask = mask
                
                # standard error
                num_blocks = self.stabil_calc.modal_data.num_blocks
                std_error = std_frequencies.compressed()/np.sqrt(num_blocks)
                
                # 95 % confidence interval -> student t (tabulated percentage points) * std_error (approx 2* std_error)
                self.stable_plot[name] = self.ax.errorbar(self.stabil_calc.masked_frequencies.compressed(),
                                                          self.stabil_calc.order_dummy.compressed(), xerr=scipy.stats.t.ppf(0.975,num_blocks)*std_error, zorder=zorder,
                                                          fmt='none',  ecolor=color, label=label, visible=visibility)

        else:
            if self.stable_plot[name] is not None:
                visibility = self.stable_plot[name].get_visible()
                self.stable_plot[name].remove()
            else:
                visibility = True
            mask = self.stabil_calc.get_stabilization_mask(
                name.replace('plot', 'mask'))

            self.stabil_calc.masked_frequencies.mask = mask
            self.stabil_calc.order_dummy.mask = mask
            
            self.stable_plot[name] = self.ax.scatter(self.stabil_calc.masked_frequencies.compressed(),
                                                     self.stabil_calc.order_dummy.compressed(), zorder=zorder,
                                                     facecolors='none', edgecolors=color,
                                                     marker=marker,
                                                     s=size, label=label, visible=visibility)
        
        #self.ax.set_yticks([])
        
        mask_stable = self.stabil_calc.get_stabilization_mask('mask_pre')
        self.stabil_calc.masked_frequencies.mask = mask_stable
        self.stabil_calc.order_dummy.mask = mask_stable

        self.fig.canvas.draw_idle()
    
    def show_MC(self,b=False):

        if b:
            ylim = self.fig.axes[0].get_ylim()
            if len(self.fig.axes)<2:
                self.fig.add_subplot(1,2,2, sharey=self.fig.axes[0])
                gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=(6,1), wspace=0, hspace=0)
                self.fig.axes[0].set_subplotspec(gs[0])
                self.fig.axes[1].set_subplotspec(gs[1])
            ax = self.fig.axes[1]
            MCs=np.zeros((self.stabil_calc.modal_data.max_model_order))
            for order in range(self.stabil_calc.modal_data.max_model_order):
                MCs[order] = np.sum(self.stabil_calc.modal_data.modal_contributions[order,:])
            ax.plot(MCs, list(range(self.stabil_calc.modal_data.max_model_order)), marker='o',fillstyle='full', 
                    markerfacecolor='white',markeredgecolor='grey', color='darkgrey',
                    markersize=4)
            ax.grid(True)
            ax.set_ylim(ylim)
            ax.set_yticks([])
            ax.set_xlim([0,1])
            ax.set_xticks([0,0.25,0.5,0.75,1])
            ax.set_xticklabels(['0','0.25','0.5','0.75','1'])
        else:
            if len(self.fig.axes)<2:
                return
            ax = self.fig.axes[1]
            self.fig.delaxes(ax)
            gs = matplotlib.gridspec.GridSpec(1, 1, wspace=0, hspace=0)
            self.fig.axes[0].set_subplotspec(gs[0])
        
        self.fig.canvas.draw_idle()
        
    def plot_fft(self, b, NFFT=2048):
        '''
        Todo: - add GUI for choosing PSD parameters
        '''
        if self.psd_plot is not None:
            for line in self.psd_plot:
                line._visible = b
            self.fig.canvas.draw_idle()
            return

        if self.stabil_calc.prep_data is None:
            raise RuntimeError('Measurement Data was not provided!')
        if not b: 
            return
        ft_freq, sum_ft = self.stabil_calc.prep_data.get_fft(svd=True,NFFT=NFFT)

        #print('ft_freq = ', ft_freq)
        #print('sum_ft = ', sum_ft)
        #print('self.stabil_calc.modal_data.max_model_order = ',
        #      self.stabil_calc.modal_data.max_model_order)
        
        sum_ft = sum_ft - np.min(sum_ft)
        sum_ft = sum_ft / (np.max(sum_ft)) * 0.5 * \
            self.stabil_calc.modal_data.max_model_order
        for channel in range(sum_ft.shape[0]):
#             self.psd_plot = self.ax2.semilogy(
#                 ft_freq, sum_ft[channel,:], color='grey', linestyle='solid', visible=b)
            self.psd_plot = self.ax.plot(
                ft_freq, sum_ft[channel,:], color='grey', linestyle='solid', visible=b, zorder=-1)
#         ymin,ymax= self.ax2.get_ylim()
#         ymin = np.percentile(sum_ft[-1,:],5)
#         ymax = (ymax-ymin)*2+ymin
#         self.ax2.set_ylim((ymin,ymax))
#         self.ax2.set_yticks([],[])
        self.fig.canvas.draw_idle()

    def update_xlim(self, xlim):
        self.ax.set_xlim(xlim)
        self.fig.canvas.draw_idle()

    def update_ylim(self, ylim):
        self.ax.set_ylim(ylim)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    def snap_frequency(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_df')
            self.cursor.set_mask(mask, 'mask_df')

    #@pyqtSlot(bool)
    def snap_damping(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_dd')
            self.cursor.set_mask(mask, 'mask_dd')

    #@pyqtSlot(bool)
    def snap_vector(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_dmac')
            self.cursor.set_mask(mask, 'mask_dmac')

    #@pyqtSlot(bool)
    def snap_stable(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_stable')
            self.cursor.set_mask(mask, 'mask_stable')

    #@pyqtSlot(bool)
    def snap_all(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_pre')
            self.cursor.set_mask(mask, 'mask_pre')

    #@pyqtSlot(bool)
    def snap_clear(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_autoclear')
            self.cursor.set_mask(mask, 'mask_autoclear')

    #@pyqtSlot(bool)
    def snap_select(self, b=True):
        if b:
            mask = self.stabil_calc.get_stabilization_mask('mask_autoselect')
            self.cursor.set_mask(mask, 'mask_autoselect')

    #@pyqtSlot(int)
    def toggle_df(self, b):
        plot_obj = self.stable_plot['plot_df']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_stdf(self, b):
        plot_obj = self.stable_plot['plot_stdf']
        if plot_obj is None:
            return
        for obj in plot_obj:
            if obj is None: continue
            for nobj in obj:
                nobj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_stdd(self, b):
        plot_obj = self.stable_plot['plot_stdd']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_ad(self, b):
        plot_obj = self.stable_plot['plot_ad']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_dd(self, b):
        plot_obj = self.stable_plot['plot_dd']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_dmac(self, b):
        plot_obj = self.stable_plot['plot_dmac']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_mpc(self, b):
        plot_obj = self.stable_plot['plot_mpc']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_mpd(self, b):
        plot_obj = self.stable_plot['plot_dmac']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_mtn(self, b):
        plot_obj = self.stable_plot['plot_mtn']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_dev(self, b):
        plot_obj = self.stable_plot['plot_dev']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_dmtn(self, b):
        plot_obj = self.stable_plot['plot_dmtn']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_stable(self, b):
        # print('plot_stable',b)
        plot_obj = self.stable_plot['plot_stable']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_clear(self, b):
        # print('plot_autoclear',b)
        plot_obj = self.stable_plot['plot_autoclear']
        if plot_obj is None:
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_select(self, b):
        plot_obj = self.stable_plot['plot_autosel']
        if plot_obj is None:
            return
        for plot_obj_ in plot_obj:
            plot_obj_.set_visible(b)
        self.fig.canvas.draw_idle()

    #@pyqtSlot(bool)
    #@pyqtSlot(int)
    def toggle_all(self, b):
        plot_obj = self.stable_plot['plot_pre']
        if plot_obj is None:
            print('plot_pre not found')
            return
        plot_obj.set_visible(b)
        self.fig.canvas.draw_idle()

    def save_figure(self, fname=None):

        startpath = rcParams.get('savefig.directory', '')
        startpath = os.path.expanduser(startpath)
        start = os.path.join(startpath, self.fig.canvas.get_default_filename())

        if fname:
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                rcParams['savefig.directory'] = os.path.dirname(str(fname))
            try:
                scatter_objs = []
                ord_mask = self.stabil_calc.order_dummy.mask
                self.stabil_calc.order_dummy.mask = np.ma.nomask
                f_mask = self.stabil_calc.masked_frequencies.mask
                self.stabil_calc.masked_frequencies.mask = np.ma.nomask

                for mode in self.stabil_calc.select_modes:
                    mode = tuple(mode)
                    y, x = self.stabil_calc.order_dummy[
                        mode], self.stabil_calc.masked_frequencies[mode]
                    # print(x,y)
                    scatter_objs.append(
                        self.ax.scatter(x, y, facecolors='none', edgecolors='red', s=200, visible=True))

                self.stabil_calc.order_dummy.mask = ord_mask
                self.stabil_calc.masked_frequencies.mask = f_mask

                text = self.ax.annotate(
                    str(self.stabil_calc.start_time), xy=(0.85, 0.99), xycoords='figure fraction')

                self.fig.canvas.print_figure(str(fname))

                text.remove()

                for scatter_obj in scatter_objs:
                    scatter_obj.remove()
                del scatter_objs

            except Exception as e:
                import traceback
                traceback.print_exc()


class ComplexPlot(QMainWindow):

    def __init__(self):

        QMainWindow.__init__(self)
        self.setWindowTitle('Modeshapeplot in complex plane')
        self.setGeometry(300, 300, 1000, 600)
        main_frame = QWidget()
        vbox = QVBoxLayout()
        
        self.fig = Figure(facecolor='white', dpi=100, figsize=(4,4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        #self.canvas.setParent(main_frame)
        
        vbox.addWidget(self.canvas,10,Qt.AlignCenter)
        main_frame.setLayout(vbox)
        
        self.setCentralWidget(main_frame)
        #self.show()

    def scatter_this(self, msh, mp=None):

        self.ax.cla()
        self.ax.scatter(msh.real, msh.imag)

        if mp is not None:
            while mp < 0:
                mp += 180
            while mp > 360:
                mp -= 360
            mp = mp * np.pi / 180
            xmin, xmax = -1, 1
            ymin, ymax = -1, 1
            if mp <= np.pi / 2:
                x1 = max(xmin, ymin / np.tan(mp))
                x2 = min(xmax, ymax / np.tan(mp))
                y1 = max(ymin, xmin * np.tan(mp))
                y2 = min(ymax, xmax * np.tan(mp))
            elif mp <= np.pi:
                x1 = max(xmin, ymax / np.tan(mp))
                x2 = min(xmax, ymin / np.tan(mp))
                y2 = max(ymin, xmax * np.tan(mp))
                y1 = min(ymax, xmin * np.tan(mp))
            elif mp <= 3 * np.pi / 2:
                x1 = max(xmin, ymin / np.tan(mp))
                x2 = min(xmax, ymax / np.tan(mp))
                y1 = max(ymin, xmin * np.tan(mp))
                y2 = min(ymax, xmax * np.tan(mp))
            else:
                x1 = max(xmin, ymax / np.tan(mp))
                x2 = min(xmax, ymin / np.tan(mp))
                y2 = max(ymin, xmax * np.tan(mp))
                y1 = min(ymax, xmin * np.tan(mp))
            self.ax.plot([x1, x2], [y1, y2])
        lim = max(max(abs(msh.real)) * 1.1, max(abs(msh.imag)) * 1.1)
        self.ax.set_xlim((-lim, lim))
        self.ax.set_ylim((-lim, lim))
        self.ax.spines['left'].set_position(('data', 0))
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.spines['right'].set_position(('data', 0 - 1))
        self.ax.spines['top'].set_position(('data', 0 - 1))

        # Hide the line (but not ticks) for "extra" spines
        for side in ['right', 'top']:
            self.ax.spines[side].set_color('none')

        # On both the x and y axes...
        for axis, center in zip([self.ax.xaxis, self.ax.yaxis], [0, 0]):
            axis.set_minor_locator(ticker.NullLocator())
            axis.set_major_formatter(ticker.NullFormatter())

        self.fig.canvas.draw_idle()

    def plot_diagram(self):

        self.fig.set_tight_layout(True)
        self.ax = self.fig.add_subplot(111)
        self.ax.autoscale_view(tight=True)
        # Set the axis's spines to be centered at the given point
        # (Setting all 4 spines so that the tick marks go in both directions)
        self.ax.spines['left'].set_position(('data', 0))
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.spines['right'].set_position(('data', 0 - 1))
        self.ax.spines['top'].set_position(('data', 0 - 1))
        
        self.ax.xaxis.set_label('Re')
        self.ax.yaxis.set_label('Im')

        # Hide the line (but not ticks) for "extra" spines
        for side in ['right', 'top']:
            self.ax.spines[side].set_color('none')

        # On both the x and y axes...
        for axis, center in zip([self.ax.xaxis, self.ax.yaxis], [0, 0]):
            axis.set_minor_locator(ticker.NullLocator())
            axis.set_major_formatter(ticker.NullFormatter())
        self.fig.canvas.draw_idle()


class ModeShapePlot(object):

    def __init__(self, stabil_calc, modal_data, geometry_data, prep_data,**kwargs):
        
        #print(kwargs)
        super().__init__()
        #sys.path.append("/vegas/users/staff/womo1998/Projects/2016_Burscheid") 
        #from main_Schwabach_2019 import print_mode_info
        
        self.mode_shape_plot = PlotMSH.ModeShapePlot(
            stabil_calc=stabil_calc, 
            modal_data=modal_data,
            geometry_data=geometry_data, 
            prep_data=prep_data, 
            amplitude=20, 
            linewidth=0.5,
            #callback_fun=print_mode_info
            **kwargs)
        self.mode_shape_plot.show_axis = False
        # self.mode_shape_plot.draw_nodes()
        self.mode_shape_plot.draw_lines()
        # self.mode_shape_plot.draw_master_slaves()
        # self.mode_shape_plot.draw_chan_dofs()

        self.fig = self.mode_shape_plot.fig
        self.fig.set_size_inches((2, 2))
        self.canvas = self.fig.canvas.switch_backends(FigureCanvasQTAgg)
        self.mode_shape_plot.canvas = self.canvas
        self.fig.get_axes()[0].mouse_init()
        #self.canvas = self.mode_shape_plot.canvas

        for axis in [self.mode_shape_plot.subplot.xaxis, self.mode_shape_plot.subplot.yaxis, self.mode_shape_plot.subplot.zaxis]:
            axis.set_minor_locator(ticker.NullLocator())
            axis.set_major_formatter(ticker.NullFormatter())
        self.mode_shape_plot.animate()
        
        
        
    def plot_this(self, index):
        #self.mode_shape_plot.stop_ani()
        self.mode_shape_plot.change_mode(mode_index=index)
        #self.mode_shape_plot.animate()


class MyMplCanvas(FigureCanvasQTAgg):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=2.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(True)

        self.compute_initial_figure()

        #
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QSizePolicy.Expanding,
                                        QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class HistoPlot(QMainWindow):

    def __init__(self, all_data, stabil_data, title='', ranges=None, select_ranges=[None], select_callback=[None]):
        QMainWindow.__init__(self)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)

        self.main_widget = QWidget(self)

        l = QVBoxLayout(self.main_widget)
        sc = MyMplCanvas(self.main_widget, width=5, height=2.5, dpi=100)
        l.addWidget(sc)
        m = QHBoxLayout()
        m.addWidget(QLabel('min'))

        if ranges is None:
            ranges = (all_data.min(), all_data.max())

        step = (ranges[1] - ranges[0]) / 50
        self.lrange = DelayedDoubleSpinBox(decimals=8, singleStep=step)
        self.lrange.setValue(ranges[0])
        self.lrange.valueChangedDelayed.connect(self.update_range)
        m.addWidget(self.lrange)

        m.addWidget(QLabel('max'))
        self.urange = DelayedDoubleSpinBox(decimals=8, singleStep=step)
        self.urange.setValue(ranges[1])
        self.urange.valueChangedDelayed.connect(self.update_range)
        m.addWidget(self.urange)
        l.addLayout(m)

        self.axes = sc.axes
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.all_data = np.copy(all_data)
        self.stabil_data = np.copy(stabil_data)

        self.all_patches = None
        self.stabil_patches = None
        self.ranges = None
        self.visible=True
        self.select_ranges = select_ranges
        self.select_callback = select_callback
        self.selector_lines = []
        #print('here in hist')
        self.update_range()

    def update_range(self, *args):
        self.ranges = (self.lrange.value(), self.urange.value())
        # print(self.ranges)
        if self.ranges[0] >= self.ranges[1]:
            return
        if self.all_patches:
            for patch in self.all_patches:
                patch.remove()
        n, self.bins, self.all_patches = self.axes.hist(
            self.all_data, bins=50, color='blue', range=self.ranges)
        self.axes.set_xlim(self.ranges)
        self.axes.set_ylim((0, max(n) * 1.1))
        self.axes.set_yticks([])
        if self.select_callback[0] is not None and (self.select_ranges[0] is not None or self.select_ranges[1] is not None) and not self.selector_lines:
            for val in self.select_ranges:
                if val is None:
                    self.selector_lines.append(None)
                else:
                    line = self.axes.axvline(val, picker=5, color='red')
                    # print(line)
                    self.selector_lines.append(line)
            self.axes.figure.canvas.mpl_connect(
                'pick_event', self.on_pick_event)
            self.axes.figure.canvas.mpl_connect(
                "button_release_event", self.on_release_event)
            self.axes.figure.canvas.mpl_connect(
                "motion_notify_event", self.on_move_event)
            self.dragged = None

        self.update_histo(self.stabil_data)

    def update_histo(self, stabil_data, select_ranges=None):
        self.stabil_data = np.copy(stabil_data)

        if self.stabil_patches:
            for patch in self.stabil_patches:
                patch.remove()

        _, _, self.stabil_patches = self.axes.hist(
            stabil_data, bins=self.bins, color='orange')
        if self.selector_lines and select_ranges is not None:

            # self.axes.figure.canvas.mpl_disconnect(self.connect_cid)
            for val, line in zip(select_ranges, self.selector_lines):
                if line is None:
                    continue
                line.set_xdata(val)
        self.axes.figure.canvas.draw_idle()

    def closeEvent(self, e):
        self.visible = False
        e.ignore()
        self.hide()

    def on_pick_event(self, event):

        self.dragged = event.artist
        self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

        return True

    def on_release_event(self, event):
        " Update text position and redraw"

        if self.dragged is not None:
            xdata = event.xdata
            if not xdata:
                return False
            #old_pos = self.dragged.get_xdata()
            #new_pos = old_pos[0] + event.xdata - self.pick_pos[0]
            self.dragged.set_xdata(xdata)
            #print(self.dragged.get_xdata(), event.xdata)
            ind = self.selector_lines.index(self.dragged)
            self.select_callback[ind](xdata)

            if len(self.selector_lines) == 1:
                self.urange.setValue(xdata * 2)
                self.urange.delayed_emit()
            elif len(self.selector_lines) == 2:
                delta_x = (self.ranges[0 if ind == 1 else 1] - xdata) / 2
                [self.lrange, self.urange][ind].setValue(xdata - delta_x)
                [self.lrange, self.urange][ind].delayed_emit()

            self.dragged = None
            self.axes.figure.canvas.draw_idle()
        return True

    def on_move_event(self, event):
        " Update text position and redraw"

        if self.dragged is not None:
            #old_pos = self.dragged.get_xdata()
            #new_pos = old_pos[0] + event.xdata - self.pick_pos[0]
            self.dragged.set_xdata(event.xdata)
            #print(self.dragged.get_xdata(), event.xdata)
            #self.dragged = None
            self.axes.figure.canvas.draw_idle()
        return True


class DataCursor(Cursor, QObject):
    # create and edit an instance of the matplotlib default Cursor widget

    show_current_info = pyqtSignal(tuple)
    mode_selected = pyqtSignal(tuple)
    mode_deselected = pyqtSignal(tuple)

    def __init__(self, ax, order_data, f_data, mask=None,  useblit=True, datalist=[], **lineprops):

        Cursor.__init__(self, ax, useblit=useblit, **lineprops)
        QObject.__init__(self)
        self.ax = ax

        self.y = order_data
        self.y.mask = np.ma.nomask

        self.x = f_data
        self.x.mask = np.ma.nomask

        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.ma.nomask

        self.name_mask = 'mask_stable'
        self.i = None

        # that list should eventually be replaced by a matplotlib.collections
        # collection
        self.scatter_objs = []

        self.datalist = datalist
        if datalist:
            self.add_datapoints(datalist)

        self.fig_resized()

    def add_datapoint(self, datapoint):
        datapoint = tuple(datapoint)
        if datapoint not in self.datalist:
            self.datalist.append(datapoint)
        x, y = self.x[datapoint], self.y[datapoint]
        #print(x,y)
        self.scatter_objs.append(self.ax.scatter(
            x, y, facecolors='none', edgecolors='red', s=200, visible=False))
        self.mode_selected.emit(datapoint)

    def add_datapoints(self, datalist):
        # convenience function for add_datapoint
        for datapoint in datalist:
            self.add_datapoint(datapoint)

    def remove_datapoint(self, datapoint):
        datapoint = tuple(datapoint)
        if datapoint in self.datalist:
            ind = self.datalist.index(datapoint)
            self.scatter_objs[ind].remove()
            del self.scatter_objs[ind]
            self.datalist.remove(datapoint)
            self.mode_deselected.emit(datapoint)
        else:
            print(datapoint, 'not in self.datalist')

    def remove_datapoints(self, datalist):
        # convenience function for remove_datapoint
        for datapoint in datalist:
            self.remove_datapoint(datapoint)

    def set_mask(self, mask, name):
        self.mask = mask
        self.fig_resized()
        self.name_mask = name

    def fig_resized(self, event=None):
        #self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)

        if event is not None:
            self.width, self.height = event.width, event.height
        else:
            self.width, self.height = self.ax.get_figure(
            ).canvas.get_width_height()

        self.xpix, self.ypix = self.ax.transData.transform(
            np.vstack([self.x.flatten(), self.y.flatten()]).T).T

        self.xpix.shape = self.x.shape
        self.xpix.mask = self.mask

        self.ypix.shape = self.y.shape
        self.ypix.mask = self.mask

    def onmove(self, event):
        if self.ignore(event):
            return
        '''
        1. Override event.data to force it to snap-to nearest data item
        2. On a mouse-click, select the data item and append it to a list of selected items
        3. The second mouse-click on a previously selected item, removes it from the list
        '''
        if (self.xpix.mask == True).all():  # i.e. no stable poles
            return

        if event.name == "motion_notify_event":

            # get cursor coordinates
            xdata = event.xdata
            ydata = event.ydata

            if xdata is None or ydata is None:
                return

            xData_yData_pixels = self.ax.transData.transform(
                np.vstack([xdata, ydata]).T)

            xdata_pix, ydata_pix = xData_yData_pixels.T

            self.fig_resized()

            self.i = self.findIndexNearestXY(xdata_pix[0], ydata_pix[0])
            xnew, ynew = self.x[self.i], self.y[self.i]

            if xdata == xnew and ydata == ynew:
                return

            # set the cursor and draw
            event.xdata = xnew
            event.ydata = ynew

            self.show_current_info.emit(self.i)

        # select item by mouse-click only if the cursor is active and in the
        # main plot
        if event.name == "button_press_event" and event.inaxes == self.ax and self.i is not None:

            if not self.i in self.datalist:
                # self.linev.set_visible(False)
                # self.lineh.set_visible(False)
                #self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)
                self.datalist.append(self.i)
                # self.ax.hold(True) # overlay plots
                # plot a circle where clicked
                self.scatter_objs.append(self.ax.scatter(self.x[self.i], self.y[
                                         self.i], facecolors='none', edgecolors='red', s=200, visible=False))
                self.mode_selected.emit(self.i)
                # self.ax.draw_artist(self.scatter_objs[-1])

            else:
                ind = self.datalist.index(self.i)
                self.scatter_objs[ind].remove()
                del self.scatter_objs[ind]
                self.datalist.remove(self.i)
                self.mode_deselected.emit(self.i)

            # self.ax.figure.canvas.restore_region(self.background)
            # self.ax.figure.canvas.blit(self.ax.figure.bbox)

            self.i = None

        Cursor.onmove(self, event)
        #for scatter in self.scatter_objs: scatter.set_visible(False)

    def _update(self):

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            for scatter in self.scatter_objs:
                scatter.set_visible(True)
                self.ax.draw_artist(scatter)
                scatter.set_visible(False)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:

            self.canvas.draw_idle()

        return False

    def findIndexNearestXY(self, x_point, y_point):

        distance = np.square(
            self.ypix - y_point) + np.square(self.xpix - x_point)
        index = np.argmin(distance)
        index = np.unravel_index(index, distance.shape)
        return index


def nearly_equal(a, b, sig_fig=5):
    return (a == b or
            int(a * 10**sig_fig) == int(b * 10**sig_fig)
            )


def start_stabil_gui(stabil_plot, modal_data, geometry_data=None, prep_data=None, select_modes=[],**kwargs):
    #print(kwargs)
    def handler(msg_type, msg_string):
        pass

    if not 'app' in globals().keys():
        global app
        app = QApplication(sys.argv)
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
        
    assert isinstance(stabil_plot, StabilPlot)
    cmpl_plot = ComplexPlot()
    if geometry_data is not None:# and prep_data is not None:
        
        import PlotMSH
#         msh_plot = ModeShapePlot(
#             stabil_plot.stabil_calc, modal_data, geometry_data, prep_data,**kwargs)
        mode_shape_plot = PlotMSH.ModeShapePlot(stabil_calc=stabil_plot.stabil_calc,
                                                modal_data=modal_data, 
                                                geometry_data=geometry_data, 
                                                prep_data=prep_data,
                                                **kwargs)
        
        msh_plot = PlotMSH.ModeShapeGUI(mode_shape_plot, reduced_gui=True)
        msh_plot.setGeometry(1000, 0, 800, 600)
        msh_plot.reset_view()
        msh_plot.hide()
        
        
    else:
        msh_plot = None


    # qInstallMessageHandler(handler) #suppress unimportant error msg

    stabil_gui = StabilGUI(stabil_plot, cmpl_plot, msh_plot)
    stabil_plot.cursor.add_datapoints(select_modes)
    loop = QEventLoop()
    stabil_gui.destroyed.connect(loop.quit)
    loop.exec_()
    print('Exiting GUI')
    canvas = FigureCanvasBase(stabil_plot.fig)
    return


class FigureCanvasQTAgg_(FigureCanvasQTAgg):

    def change_backend(self, figure):
        FigureCanvasQT.__init__(self, figure)
        self._drawRect = None
        self.blitbox = None
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        if sys.platform.startswith('win'):
            self._priv_update = self.repaint
        else:
            self._priv_update = self.update


class DelayedDoubleSpinBox(QDoubleSpinBox):
    '''
    reimplementation of QDoubleSpinBox to delay the emit of the 
    valueChanged signal by 1.5 seconds after the last change of the value
    this allows for a function to be directly connected to the signal
    without the need to check for further changes of the value
    else when the user clicks through the values it would emit a
    lot of signals and the connected funtion would run this many times
    note that you have to connect to valueChangedDelayed signal if 
    you want to make use of this functionality
    valueChanged signal works as in QDoubleSpinBox
    '''
    # define custom signals
    valueChangedDelayed = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        '''
        inherit from QDoubleSpinBox
        instantiate a timer and set its default timeout value (1500 ms)
        connect the valueChanged signal of QDoubleSpinBox to the 
        start () slot of QTimer
        connect the timeout () signal of QTimer to delayed emit
        '''
        super(DelayedDoubleSpinBox, self).__init__(*args, **kwargs)
        self.timer = QTimer()
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.delayed_emit)
        self.valueChanged[float].connect(self.timer.start)

    #@pyqtSlot()
    def delayed_emit(self):
        '''
        stop the timer and send the current value of the QDoubleSpinBox
        '''
        self.timer.stop()
        self.valueChangedDelayed.emit(self.value())

    def set_timeout(self, timeout):
        '''
        set the timeout of the timer to a custom value
        '''
        assert isinstance(timeout, (int, float))
        self.timer.setInterval(timeout)


if __name__ == '__main__':
    
#     min=QSliderL.value()
#     max=QSliderU.value()
#     QScrollBar.setRange(min,max)
#     
#     min = QScrollBar.minimum
#     max = QScrollBar.maximum
#     QSliderL.setValue(min)
#     QSliderU.setValue(max)
#
    nbins=150
#     x=np.arange(0,5,5/nbins)
    import scipy .stats
    aa = np.random.rand(25)/2+0.5
#     q5fit=[]
#     q50fit=[]
#     q95fit=[]
#     q5num=[]
#     q50num=[]
#     q95num=[]
#     q5real=[]
#     q50real=[]
#     q95real=[]
    for a in aa:
        print(a)
        scale=2
        loc=0
        random_numbers=np.random.weibull(a=a, size=(30000))*scale+loc
#         a_,loc_,scale_=scipy.stats.weibull_min.fit(random_numbers)#,floc=loc,scale=scale)
        log_rand=np.log(random_numbers)
        mean_log_rand=np.mean(log_rand)
        std_log_rand= np.std(log_rand)
        log_rand -= mean_log_rand
        log_rand /=std_log_rand
        n,bins,patches= plot.hist(log_rand, bins=nbins,range=(-5,5),alpha=.5, normed=True,label='log {:1.2f}, {:1.2f}, {:1.2f}, {:1.2f}'.format(a, a,loc,scale))
        n,bins,patches= plot.hist(random_numbers, bins=nbins,range=(0,5),alpha=.5, normed=True,label='normal {:1.2f}, {:1.2f}, {:1.2f}, {:1.2f}'.format(a, a,loc,scale))
        plot.legend()
        plot.show()
#         #y_=(a_/scale_)*(((x-loc_)/scale_)**(a_-1))*np.exp(-(((x-loc_)/scale_)**a_))
#         #plot.plot(x,y_,color='grey')
#         #y=(a/scale)*(((x-loc)/scale)**(a-1))*np.exp(-(((x-loc)/scale)**a))
#         #plot.plot(x,y,color='black')
#         num = np.percentile(a=random_numbers, q=[5,50,95])
#         q5num.append(num[0])
#         q50num.append(num[1])
#         q95num.append(num[2])
#         fit = scipy.stats.weibull_min.ppf(q=[0.05,0.5,0.95],c=a_,loc=loc_, scale=scale_)
#         q5fit.append(fit[0])
#         q50fit.append(fit[1])
#         q95fit.append(fit[2])
#         
#         real = scipy.stats.weibull_min.ppf(q=[0.05,0.5,0.95],c=a,loc=loc, scale=scale)
#         q5real.append(real[0])
#         q50real.append(real[1])
#         q95real.append(real[2])
#         
#         #print('numeric: {:1.2f}, {:1.2f}, {:1.2f}'.format(*num))
#         #print('fit: {:1.2f}, {:1.2f}, {:1.2f}'.format(*fit))
#         #print('real: {:1.2f}, {:1.2f}, {:1.2f}'.format(*real))
#         #print('')
#         
#     plot.figure(figsize=(5,5))
#     plot.scatter(q5real,q5fit,label='q5fit')
#     plot.scatter(q5real,q5num,label='q5num')
#     minval = min(q5real+q5fit+q5num)
#     maxval = max(q5real+q5fit+q5num)
#     plot.plot([minval,maxval],[minval,maxval],)
#     plot.xlim((minval,maxval))
#     plot.ylim((minval,maxval))
#     plot.xlabel('real')
#     plot.ylabel('fit/num')
#     plot.legend()
#     
#     
#     plot.figure(figsize=(5,5))
#     plot.scatter(q50real,q50fit,label='q50fit')
#     plot.scatter(q50real,q50num,label='q50num')
#     minval = min(q50real+q50fit+q50num)
#     maxval = max(q50real+q50fit+q50num)
#     plot.plot([minval,maxval],[minval,maxval],)
#     plot.xlim((minval,maxval))
#     plot.ylim((minval,maxval))
#     plot.xlabel('real')
#     plot.ylabel('fit/num')
#     plot.legend()
#     
#     plot.figure(figsize=(5,5))
#     plot.scatter(q95real,q95fit,label='q95fit')
#     plot.scatter(q95real,q95num,label='q95num')
#     minval = min(q95real+q95fit+q95num)
#     maxval = max(q95real+q95fit+q95num)
#     plot.plot([minval,maxval],[minval,maxval],)
#     plot.xlim((minval,maxval))
#     plot.ylim((minval,maxval))
#     plot.xlabel('real')
#     plot.ylabel('fit/num')
#     plot.legend()
#     #plot.legend()
#     plot.show()