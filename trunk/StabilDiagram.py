# -*- coding: utf-8 -*-
'''
Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''

import numpy as np
import sys
import os

from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.text import TextPath, FontProperties
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.widgets import Cursor

from PyQt4.QtGui import QMainWindow, QWidget, QHBoxLayout, QPushButton,\
    QCheckBox, QButtonGroup, QLabel, QToolButton, QComboBox, QStyle,\
    QTextEdit, QGridLayout, QFrame, QVBoxLayout, QAction, QIcon,\
    QFileDialog, QInputDialog, QMessageBox, QDoubleSpinBox, QTableWidget,\
    QSpinBox, QAbstractItemView, QTableWidgetItem, QApplication, QRadioButton,\
    QLineEdit, QPalette, QSizePolicy
from PyQt4.QtCore import pyqtSignal, Qt, pyqtSlot, QTimer, QObject, qInstallMsgHandler, QEventLoop

'''
TODO:
scale markers right on every platform
frequency range as argument or from ssi params, sampling freq
ask if user wants to save results or just pass them to caller
add switch to choose between "unstable only in ..." or "stable in ..."
(select and merge several poles with a rectangular mouse selection)
make possible missing model orders eg use only every 2nd model order

'''

class StabilGUI(QMainWindow):
    
    def __init__(self, modal_frequencies,modal_damping, mode_shapes, 
                 num_analised_channels=None, max_model_order=None, 
                 stab_frequency=1, stab_damping=5, stab_MAC=2,
                 measurement=None, sampling_rate=None,
                 result_folder=None, num_block_columns=0, num_block_rows=0, select_modes=[]):
        
        QMainWindow.__init__(self)
        self.setWindowTitle('Stabilization Diagram')
        self.create_menu()
        self.create_main_frame(modal_frequencies,modal_damping, mode_shapes, 
                             num_analised_channels, max_model_order, 
                             stab_frequency, stab_damping, stab_MAC,
                             measurement, sampling_rate,
                             result_folder, num_block_columns, num_block_rows, select_modes)
        self.setGeometry(300, 300, 1000, 600)
        self.setWindowModality(Qt.ApplicationModal)
        self.showMaximized()
        
        

        for widg in [self.mode_val_view,self.current_value_view]:
            widg.setText('\n \n \n \n \n \n \n')
            height=widg.document().size().toSize().height()+3
            widg.setFixedHeight(height)   
        self.plot_selector_msh.setChecked(True)
        self.update_stabil_view()
        self.stabil_plot.cursor.add_datapoints(select_modes)
    
    def create_main_frame(self,modal_frequencies,modal_damping, mode_shapes, 
                         num_analised_channels, max_model_order, 
                         stab_frequency, stab_damping, stab_MAC,
                         measurement, sampling_rate,
                         result_folder, num_block_columns, num_block_rows, select_modes):
        '''
        set up all the widgets and other elements to draw the GUI
        '''
        self.result_folder=result_folder
        main_frame = QWidget()

        # Create the mpl Figure and FigCanvas objects.
        self.fig = Figure(facecolor='white', dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        
        self.stabil_plot = StabilPlot(modal_frequencies,modal_damping, mode_shapes, 
                                     num_analised_channels, max_model_order, 
                                     stab_frequency, stab_damping, stab_MAC,
                                     measurement, sampling_rate,
                                     result_folder, num_block_columns, num_block_rows, self.fig)
        
        self.stabil_plot.calculate_stabilization_values()
        self.stabil_plot.plot_diagram()
        self.stabil_plot.cursor.show_current_info.connect(self.update_value_view)
        self.stabil_plot.cursor.mode_selected.connect(self.mode_selector_add)
        self.stabil_plot.cursor.mode_deselected.connect(self.mode_selector_take)
        
        
        
        self.canvas.setParent(main_frame)
        
        main_layout = QHBoxLayout()
        
        left_pane_layout = QVBoxLayout()
        left_pane_layout.addStretch(1)
        palette = QPalette();
        palette.setColor(QPalette.Base,Qt.transparent)
        
        self.current_value_view = QTextEdit()
        self.current_value_view.setFrameShape(QFrame.Box)
        self.current_value_view.setPalette(palette)
        
        self.diag_val_widget = QWidget()
        
        fra_1 = QFrame()
        fra_1.setFrameShape(QFrame.Panel)
        fra_1.setLayout(self.create_stab_val_widget(df_max=stab_frequency, dd_max=stab_damping, d_mac=stab_MAC, 
                               ))
        left_pane_layout.addWidget(fra_1)
        
        left_pane_layout.addStretch(2)
        
        fra_2 = QFrame()
        fra_2.setFrameShape(QFrame.Panel)
        fra_2.setLayout(self.create_diag_val_widget())
        left_pane_layout.addWidget(fra_2)
        
        left_pane_layout.addStretch(2)
        left_pane_layout.addWidget(self.current_value_view)
        left_pane_layout.addStretch(1)
        
        right_pane_layout = QVBoxLayout()
        
        self.plot_selector_c = QRadioButton('Mode Shape in Complex Plane')
        self.plot_selector_c.toggled.connect(self.toggle_cpl_plot)
        self.plot_selector_msh = QRadioButton('Mode Shape in Spatial Model')
        self.plot_selector_msh.toggled.connect(self.toggle_msh_plot)
        
        
        self.group=QButtonGroup()
        self.group.addButton(self.plot_selector_c)
        self.group.addButton(self.plot_selector_msh)
        
        self.mode_selector = QComboBox()
        self.mode_selector.currentIndexChanged[int].connect(self.update_mode_val_view)
        
        self.mode_plot_widget = QWidget()
        self.cmplx_plot_widget=QWidget()
        fig = Figure(facecolor='white', dpi=100, figsize=(2,2))
        canvas = FigureCanvasQTAgg(fig)
        canvas.setParent(self.cmplx_plot_widget)

        self.cmpl_plot = ModeShapePlot(fig)
        lay=QHBoxLayout()
        lay.addWidget(canvas)
        self.cmplx_plot_widget.setLayout(lay)
        self.cmpl_plot.plot_diagram()
        #self.msh_plot_widget=self.cmplx_plot_widget
        self.msh_plot_widget=QWidget()
        
        self.mode_val_view = QTextEdit()
        self.mode_val_view.setFrameShape(QFrame.Box)
        
        
        self.mode_val_view.setPalette(palette)
        right_pane_layout.addStretch(1)
        right_pane_layout.addWidget(self.mode_selector)
        
        right_pane_layout.addWidget(self.plot_selector_c)
        right_pane_layout.addWidget(self.plot_selector_msh)
        right_pane_layout.addStretch(2)
        self.mode_plot_layout=QVBoxLayout()
        self.mode_plot_layout.addWidget(self.mode_plot_widget)
        right_pane_layout.addLayout(self.mode_plot_layout)
        right_pane_layout.addStretch(2)
        right_pane_layout.addWidget(self.mode_val_view)
        right_pane_layout.addStretch(1)
        
        main_layout.addLayout(left_pane_layout)
        main_layout.addWidget(self.canvas)
        main_layout.setStretchFactor(self.canvas, 1)
        main_layout.addLayout(right_pane_layout)
        vbox=QVBoxLayout()
        vbox.addLayout(main_layout)
        vbox.addLayout(self.create_buttons())
        main_frame.setLayout(vbox)
        color=main_frame.palette().color(QPalette.Base).name()
        self.stabil_plot.fig.set_facecolor('none')
        #print(color)
        self.setCentralWidget(main_frame)
        self.current_mode=(0,0)
        return
    
    def create_buttons(self):
        b0=QPushButton('Apply')
        b0.released.connect(self.update_stabil_view)
        b1 = QPushButton('Save Figure')
        b1.released.connect(self.save_figure)
        b2 = QPushButton('OK')
        b2.released.connect(self.save_results)
        b3 = QPushButton('Cancel')
        b3.released.connect(self.save_figure)
        b3.released.connect(self.close)
        b3.released.connect(self.deleteLater)
        
        lay=QHBoxLayout()
        
        lay.addWidget(b0)
        lay.addWidget(b1)
        lay.addWidget(b2)
        lay.addWidget(b3)
        lay.addStretch()
        
        return lay
        
    def update_mode_val_view(self,index):
        #display information about currently selected mode
        #print(self.mode_selector.itemText(index))
        i=self.stabil_plot.cursor.datalist[index]
        n,f,d,mpc, mp, mpd = self.stabil_plot.get_modal_values(i)
        #print('{} - {:2.3f}'.format(index, f))
        self.current_mode=i
        s='Frequency=%1.3fHz, \n Order=%1.0f, \n Damping=%1.3f%%, \n MPC=%1.5f, \n MP=%1.3f\u00b0, \n MPD=%1.5f\u00b0'%(f,n,d,mpc,mp,mpd)        
        self.mode_val_view.setText(s)
        height=self.mode_val_view.document().size().toSize().height()+3
        self.mode_val_view.setFixedHeight(height)   
        self.update_mode_plot(i, mp)
        
    
    @pyqtSlot(tuple)
    def mode_selector_add(self,i):
        #add mode tomode_selector and select it
        n,f,d,mpc, mp, mpd = self.stabil_plot.get_modal_values(i)
        index = self.stabil_plot.cursor.datalist.index(i)
        text = '{} - {:2.3f}'.format(index, f)
        self.mode_selector.currentIndexChanged[int].disconnect(self.update_mode_val_view)
        self.mode_selector.addItem(text)
        found=False        
        for index in range(self.mode_selector.count()):
            if text == self.mode_selector.itemText(index):
                found=True
        self.mode_selector.setCurrentIndex(index)
        self.update_mode_val_view(index)
        self.mode_selector.currentIndexChanged[int].connect(self.update_mode_val_view)

    @pyqtSlot(tuple)
    def mode_selector_take(self,i_):
        if self.current_mode == i_:
            if self.stabil_plot.cursor.datalist:
                self.current_mode = self.stabil_plot.cursor.datalist[0]
            else:
                self.current_mode = (0,0)
        self.mode_selector.currentIndexChanged[int].disconnect(self.update_mode_val_view)
        self.mode_selector.clear()

        for index,i in enumerate(self.stabil_plot.cursor.datalist):
            n,f,d,mpc, mp, mpd = self.stabil_plot.get_modal_values(i)
            text = '{} - {:2.3f}'.format(index, f)
            self.mode_selector.addItem(text)
            if self.current_mode == i:        
                for ind in range(self.mode_selector.count()):
                    if text == self.mode_selector.itemText(ind):
                        break
        if self.mode_selector.count():
            self.mode_selector.setCurrentIndex(ind)
            self.update_mode_val_view(ind)
        self.mode_selector.currentIndexChanged[int].connect(self.update_mode_val_view)
        
    def update_mode_plot(self, i, mpd=None):
        #update the plot of the currently selected mode
        msh=self.stabil_plot.get_mode_shape(i)
        self.cmpl_plot.scatter_this(msh, mpd)
        
    
    def toggle_msh_plot(self,b):
        #change the type of mode plot
        
        if b:
            index = self.mode_plot_layout.indexOf(self.cmplx_plot_widget)
            self.mode_plot_layout.takeAt(index)
            self.cmplx_plot_widget.hide()
            self.mode_plot_layout.insertWidget(index, self.msh_plot_widget)
            self.msh_plot_widget.show()
                    
    
    def toggle_cpl_plot(self,b ):
        #change the type of mode plot
        
        if b:
            index = self.mode_plot_layout.indexOf(self.msh_plot_widget)
            self.mode_plot_layout.takeAt(index)
            self.msh_plot_widget.hide()
            self.mode_plot_layout.insertWidget(index, self.cmplx_plot_widget)
            self.cmplx_plot_widget.show()
    
    def update_value_view(self,i):
        n,f,d,mpc,mp,mpd = self.stabil_plot.get_modal_values(i)
        s='Frequency=%1.3fHz, \n Order=%1.0f, \n Damping=%1.3f%%,  \n MPC=%1.5f, \n MP=%1.3f\u00b0, \n MPD=%1.5f\u00b0'%(f,n,d,mpc,mp,mpd)        
        self.current_value_view.setText(s)
        height=self.current_value_view.document().size().toSize().height()+3
        self.current_value_view.setFixedHeight(height)
        
    def update_stabil_view(self):
        df_max = float(self.df_edit.text())/100
        dd_max = float(self.dd_edit.text())/100
        d_mac = float(self.mac_edit.text())/100
        d_range = (float(self.d_min_edit.text()) , float(self.d_max_edit.text()))
        mpc_min = float(self.mpc_edit.text())
        mp_max = float(self.mp_edit.text())
        mpd_max = float(self.mpd_edit.text())
        f_range = (float(self.freq_low.text()) , float(self.freq_high.text()))
        n_range = (int(self.n_low.text()), int(self.n_step.text()), int(self.n_high.text()))
        
        self.stabil_plot.update_stabilization(df_max, dd_max, d_mac, d_range, mpc_min, mp_max, mpd_max, n_range)
        self.stabil_plot.update_xlim(f_range)
        self.stabil_plot.update_ylim((n_range[0],n_range[2]))
        
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
                               d_range=(0,5), mpc_min=0.9, mp_max=360, mpd_max=15):
        layout= QGridLayout()
        layout.addWidget(QLabel('Stabilization Criteria'), 1,1,1,3)
        
        layout.setColumnStretch(2,1)
        
        layout.addWidget(QLabel('Frequency [%]'), 2,1)
        self.df_edit=QLineEdit(str(df_max))
        self.df_edit.setMaxLength(3)
        self.df_edit.setFixedWidth(60)
        layout.addWidget(self.df_edit, 2,3)
        
        layout.addWidget(QLabel('Damping[%]'), 3,1)
        self.dd_edit=QLineEdit(str(dd_max))
        self.dd_edit.setMaxLength(3)
        self.dd_edit.setFixedWidth(60)
        layout.addWidget(self.dd_edit, 3,3)
        
        layout.addWidget(QLabel('MAC [%]'), 4,1)
        self.mac_edit = QLineEdit(str(d_mac))
        self.mac_edit.setMaxLength(3)
        self.mac_edit.setFixedWidth(60)
        layout.addWidget(self.mac_edit, 4,3)
        
        layout.addWidget(QLabel('Damping range [%]'), 5,1)
        self.d_min_edit = QLineEdit(str(d_range[0]))
        self.d_min_edit.setMaxLength(3)
        self.d_min_edit.setFixedWidth(60)
        self.d_max_edit = QLineEdit(str(d_range[1]))
        self.d_max_edit.setMaxLength(3)
        self.d_max_edit.setFixedWidth(60)
        lay=QHBoxLayout()
        lay.addStretch()
        lay.addWidget(self.d_min_edit)
        lay.addWidget(QLabel('to'))
        lay.addWidget(self.d_max_edit)
        layout.addLayout(lay, 5,2,1,2)
        
        layout.setRowStretch(6, 2)
        
        layout.addWidget(QLabel('MPC_min '), 7,1)
        self.mpc_edit=QLineEdit(str(mpc_min))
        self.mpc_edit.setMaxLength(6)
        self.mpc_edit.setFixedWidth(60)
        layout.addWidget(self.mpc_edit, 7,3)
        
        layout.addWidget(QLabel('MP_max [°]'), 8,1)
        self.mp_edit=QLineEdit(str(mp_max))
        self.mp_edit.setMaxLength(3)
        self.mp_edit.setFixedWidth(60)
        layout.addWidget(self.mp_edit, 8,3)
        
        layout.addWidget(QLabel('MPD_max [°]'), 9,1)
        self.mpd_edit = QLineEdit(str(mpd_max))
        self.mpd_edit.setMaxLength(3)
        self.mpd_edit.setFixedWidth(60)
        layout.addWidget(self.mpd_edit, 9,3)
        
        return layout

    def create_diag_val_widget(self, show_sf=True, show_sd=True, 
                               show_sv=True, show_sa = True, show_all=True, 
                               show_psd=False, snap_to='sa', f_range=(0,30), n_range=(0,1,0)):
        layout=QGridLayout()
        
        layout.addWidget(QLabel('View Settings'),1,1,1,2)
        
        check_sf=QCheckBox('Unstable in Frequency only')
        check_sf.setChecked(show_sf)
        self.stabil_plot.toggle_frequency(show_sf)
        check_sf.stateChanged.connect(self.stabil_plot.toggle_frequency)
        snap_sf = QRadioButton()
        snap_sf.toggled.connect(self.stabil_plot.snap_frequency)
        snap_sf.setChecked(snap_to=='sf')
        layout.addWidget(check_sf,2,1)
        layout.addWidget(snap_sf,2,2)
        
        check_sd=QCheckBox('Unstable in Damping only')
        check_sd.setChecked(show_sd)
        self.stabil_plot.toggle_damping(show_sd)
        check_sd.stateChanged.connect(self.stabil_plot.toggle_damping)
        snap_sd = QRadioButton()
        snap_sd.toggled.connect(self.stabil_plot.snap_damping)
        snap_sd.setChecked(snap_to=='sd')
        layout.addWidget(check_sd,3,1) 
        layout.addWidget(snap_sd,3,2)
        
        check_sv=QCheckBox('Unstable in MAC only')
        check_sv.setChecked(show_sv)
        self.stabil_plot.toggle_vector(show_sv)
        check_sv.stateChanged.connect(self.stabil_plot.toggle_vector)
        snap_sv = QRadioButton()
        snap_sv.toggled.connect(self.stabil_plot.snap_vector)
        snap_sv.setChecked(snap_to=='sv')
        layout.addWidget(check_sv,4,1) 
        layout.addWidget(snap_sv,4,2)
        
        check_sa=QCheckBox('Stable Pole')
        check_sa.setChecked(show_sa)
        self.stabil_plot.toggle_stable(show_sa)
        check_sa.stateChanged.connect(self.stabil_plot.toggle_stable)
        snap_sa = QRadioButton()
        snap_sa.toggled.connect(self.stabil_plot.snap_stable)
        snap_sa.setChecked(snap_to=='sa')
        layout.addWidget(check_sa,5,1)  
        layout.addWidget(snap_sa,5,2)
        
        check_all=QCheckBox('All Poles')
        check_all.setChecked(show_all)
        self.stabil_plot.toggle_all(show_all)
        check_all.stateChanged.connect(self.stabil_plot.toggle_all)
        snap_all = QRadioButton()
        snap_all.toggled.connect(self.stabil_plot.snap_all)
        snap_all.setChecked(snap_to=='all')
        layout.addWidget(check_all,6,1)  
        layout.addWidget(snap_all,6,2)
        
        psd_check=QCheckBox('Show PSD')
        psd_check.toggled.connect(self.stabil_plot.prepare_fft)
        layout.addWidget(psd_check,7,1,1,2) 
        
        lay = QHBoxLayout()
        lay.addWidget(QLabel('Freq. range:'))
        if f_range[1] == 0:
            f_range=(f_range[0],self.stabil_plot.get_max_f())
        self.freq_low = QLineEdit('{:2.3f}'.format(f_range[0]))
        self.freq_low.setFixedWidth(60)
        self.freq_high = QLineEdit('{:2.3f}'.format(f_range[1]*1.05))
        self.freq_high.setFixedWidth(60)
        lay.addWidget(self.freq_low)
        lay.addWidget(QLabel('to'))
        lay.addWidget(self.freq_high)
        lay.addWidget(QLabel('[Hz]'))
        layout.addLayout(lay, 8,1,1,2)
        
        lay = QHBoxLayout()
        lay.addWidget(QLabel('Order. range (low:step:high):'))
        if n_range[2] == 0:
            n_range=(n_range[0],n_range[1],self.stabil_plot.max_model_order)
        self.n_low = QLineEdit('{:2d}'.format(n_range[0]))
        self.n_low.setFixedWidth(60)
        self.n_step = QLineEdit('{:2d}'.format(n_range[1]))
        self.n_step.setFixedWidth(60)
        self.n_high = QLineEdit('{:2d}'.format(n_range[2]))
        self.n_high.setFixedWidth(60)
        lay.addWidget(self.n_low) 
        lay.addWidget(self.n_step)       
        lay.addWidget(self.n_high)
        layout.addLayout(lay, 9,1,1,2)
        
        return layout
    
    def save_figure(self, fname=None):
        # copied and modified from matplotlib.backends.backend_qt4.NavigationToolbar2QT
        canvas=self.stabil_plot.ax.figure.canvas
        
        filetypes = canvas.get_supported_filetypes_grouped()
        sorted_filetypes = list(filetypes.items())
        sorted_filetypes.sort()
        default_filetype = canvas.get_default_filetype()
        
        startpath = self.result_folder
        #startpath = os.path.expanduser(startpath)
        start = os.path.join(startpath, 'stabil_diag.pdf')
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)
        
        #fname = QFileDialog.getSaveFileName(self, caption="Choose a filename to save to",
        #                                directory=start, filter=filters)
        fname=start
        if fname:
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                rcParams['savefig.directory'] = os.path.dirname(str(fname))
            try:
                canvas.print_figure( str(fname) )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error saving file", str(e),
                    QMessageBox.Ok, QMessageBox.NoButton)   
                  
    def save_results(self):
        
        dname = QFileDialog.getSaveFileName(self, caption="Choose a directory to save to",
                                        directory=os.getcwd(), options=QFileDialog.ShowDirsOnly)
        self.stabil_plot.save_results(dname)
        self.close()
        self.deleteLater()
                 
class StabilPlot(object): 
    
    def __init__(self, modal_frequencies,modal_damping, mode_shapes, 
                 num_analised_channels=None, max_model_order=None, 
                 stab_frequency=1, stab_damping=5, stab_MAC=2,
                 measurement=None, sampling_rate=None,
                 result_folder=None, num_block_columns=0, num_block_rows=0,
                 figure=None):
        
        super().__init__()

        self.measurement=measurement
        self.modal_frequencies = np.ma.array(modal_frequencies)
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
        if num_analised_channels is None:
            num_analised_channels = self.mode_shapes.shape[0]
        self.num_analised_channels = num_analised_channels
        if result_folder is None:
            result_folder = os.getcwd()
        self.result_folder = result_folder
        self.sampling_rate=sampling_rate
        if max_model_order is None:
            max_model_order = self.mode_shapes.shape[1]
        self.max_model_order = max_model_order
        self.num_block_columns = num_block_columns
        self.num_block_rows = num_block_rows
        
        self.stab_frequency = stab_frequency/100 if stab_frequency > 1 else stab_frequency
        self.stab_damping = stab_damping/100 if stab_damping > 1 else stab_damping
        self.stab_MAC = stab_MAC/100 if stab_MAC > 1 else stab_MAC
        
        self.order_dummy = np.ma.array([[order]*self.max_model_order for order in range(self.max_model_order)])
        if figure is None:
            import matplotlib.pyplot as plot
            figure = plot.figure()
        self.fig = figure
        self.stable_plot = [None for i in range(6)]
        self.masks = [None for i in range(6)]
        self.select_modes = []
                
    def calculate_stabilization_values(self):
        print('Checking stabilisation criteria...')       

        # Richtung 1: model order, Richtung 2: current pole, Richtung 3: previous pole:
        self.freq_diffs = np.zeros((self.max_model_order, self.max_model_order, self.max_model_order))
        self.damp_diffs = np.zeros((self.max_model_order, self.max_model_order, self.max_model_order))
        self.MAC_diffs = np.zeros((self.max_model_order, self.max_model_order, self.max_model_order))
        
        self.MPC_matrix = np.zeros((self.max_model_order, self.max_model_order))
        self.MP_matrix = np.zeros((self.max_model_order, self.max_model_order))
        self.MPD_matrix = np.zeros((self.max_model_order, self.max_model_order))
        
        previous_freq_row = self.modal_frequencies[0,:]
        previous_damp_row = self.modal_damping[0,:]
        previous_mode_shapes_row = self.mode_shapes[:,:,0]
        
        previous_non_zero_entries  = np.nonzero(previous_freq_row) # tuple with array of indizes of non-zero frequencies
        previous_length = len(previous_non_zero_entries[0])
 
        previous_freq = previous_freq_row[previous_non_zero_entries]
        previous_damp = previous_damp_row[previous_non_zero_entries]
        previous_mode_shapes = previous_mode_shapes_row[:,previous_non_zero_entries[0]]
        
        for current_order in range(1,self.max_model_order):      
     
            current_freq_row = self.modal_frequencies[(current_order),:]
            current_damp_row = self.modal_damping[(current_order),:]
            current_mode_shapes_row = self.mode_shapes[:,:,current_order]
    
            current_non_zero_entries = np.nonzero(current_freq_row)            
            current_length = len(current_non_zero_entries[0])
            
            if not current_length: continue
            
            current_freq = current_freq_row[current_non_zero_entries]
            current_damp = current_damp_row[current_non_zero_entries]
            current_mode_shapes = current_mode_shapes_row[:,current_non_zero_entries[0]] 
                
            self.MAC_diffs[current_order, :current_length,:previous_length] = np.transpose(1 - self.calculateMAC(previous_mode_shapes[:,:previous_length], current_mode_shapes[:,:current_length]))
            
            self.MPC_matrix[current_order, :current_length] = self.calculateMPC(current_mode_shapes[:,:current_length])
            
            self.freq_diffs[current_order, :current_length, :previous_length] = np.abs((np.repeat(np.expand_dims(previous_freq, axis=1), current_freq.shape[0], axis = 1)-current_freq) / current_freq).T
            
            self.damp_diffs[current_order, :current_length, :previous_length] = np.abs((np.repeat(np.expand_dims(previous_damp, axis=1), current_damp.shape[0], axis = 1)-current_damp) / current_damp).T
            
            self.MPD_matrix[current_order, :current_length], self.MP_matrix[current_order, :current_length] =  self.calculateMPD(current_mode_shapes[:,:current_length])
                
            previous_freq=current_freq
            previous_damp=current_damp
            previous_mode_shapes=current_mode_shapes
            previous_length=current_length
            
    def prepare_fft(self,b):  
        if self.stable_plot[5]:
            for line in self.stable_plot[5]:
                line._visible=b  
            self.fig.canvas.draw_idle()
            
            return
        print("Calculating FFT's")
        
        if self.measurement is None:
            print(RuntimeWarning('Measurement File was not provided. Cant compute FFTs!'))
            return
        
        if self.sampling_rate is None:
            self.sampling_rate =int(input('Please Enter the Sampling Rate for Computing FFTs!'))
            
        sum_ft=None
        for column in range(self.measurement.shape[1]):
            sample_signal = self.measurement[:,column]  
   
            # one-dimensional averaged discrete Fourier Transform for real input
            section_length = 2048
            overlap = 0.5 * section_length
            increment = int(section_length - overlap)
            num_average = (len(sample_signal) - section_length) // increment
                
            for iii in range(num_average):
                this_signal = sample_signal[(iii * increment):(iii * increment + section_length)]
                
                ft = np.fft.rfft(this_signal * np.hanning(len(this_signal)))
                ft = abs(ft)
                if iii == 0:
                    average_ft = np.zeros(len(ft))
                average_ft = average_ft + ft
        
            average_ft = average_ft / num_average
            
            if column == 0:
                sum_ft = np.zeros(len(ft))
            
            sum_ft = sum_ft + average_ft
    
            ft_freq = np.fft.rfftfreq(section_length, d = (1/self.sampling_rate))
    
        sum_ft = sum_ft / (np.amax(sum_ft)) * 0.5 * self.max_model_order
                      
        self.stable_plot[5]=self.ax.plot(ft_freq, sum_ft, color='grey', linestyle='solid', visible=b)
        self.fig.canvas.draw_idle()
        
        return

    def get_stabilization_mask(self, df_max=None, dd_max=None, d_mac=None, \
                        d_range=None, mpc_min=None, mp_max=None, mpd_max=None, order_range=None):
        self.modal_frequencies.mask = np.ma.nomask
        self.order_dummy.mask = np.ma.nomask
        
        mask_sf = self.modal_frequencies != 0         # stable in frequency
        mask_sd = self.modal_frequencies != 0         # stable in damping
        mask_sv = self.modal_frequencies != 0         # stable in mode shape (MAC)
        mask_sa = self.modal_frequencies != 0         # stable in all criteria
        mask_other = self.modal_frequencies != 0
        
        if d_range is not None:
            assert isinstance(d_range, (tuple,list))
            assert len(d_range)==2
            mask_other = np.logical_and(mask_other, self.modal_damping >= d_range[0])
            mask_other = np.logical_and(mask_other, self.modal_damping <= d_range[1])
        if mpc_min is not None:
            mask_other = np.logical_and(mask_other, self.MPC_matrix >= mpc_min)
        if mp_max is not None:
            mask_other = np.logical_and(mask_other, self.MP_matrix <= mp_max)
        if mpd_max is not None:
            mask_other = np.logical_and(mask_other, self.MPD_matrix <= mpd_max) 
        if order_range is not None:
            start,step,stop = order_range
            start=max(0,start)
            stop = min(stop, self.max_model_order)
            mask_order = np.zeros_like(mask_other)
            for order in range(start,stop,step):
                mask_order = np.logical_or(mask_order, self.order_dummy == order)
            mask_other = np.logical_and(mask_other, mask_order) 
        mask_sa = np.logical_and(mask_sa, mask_other)
        mask_sf = np.logical_and(mask_sf, mask_other)
        mask_sd = np.logical_and(mask_sd, mask_other)
        mask_sv = np.logical_and(mask_sv, mask_other)
        
        if df_max is not None:
            # rel freq diffs for each pole with all previous poles, 
            # for all poles and orders results in 3d array
            # compare those rel freq diffs with df_max
            # and reduce 3d array to 2d array, by applying logical_or 
            # along each poles axis (diff with all previous)
            mask_sf_all = np.logical_and(self.freq_diffs!=0, self.freq_diffs<=df_max)
            mask_sf_red = np.any(mask_sf_all, axis=2)
            mask_sf = np.logical_and(mask_sf, mask_sf_red)
        if dd_max is not None:
            mask_sd_all = np.logical_and(self.damp_diffs!=0, self.damp_diffs<=dd_max)
            mask_sd_red = np.any(mask_sd_all, axis=2)
            mask_sd = np.logical_and(mask_sd, mask_sd_red)
        if d_mac is not None:
            mask_sv_all = np.logical_and(self.MAC_diffs!=0, self.MAC_diffs<=d_mac)
            mask_sv_red = np.any(mask_sv_all, axis=2)
            mask_sv = np.logical_and(mask_sv, mask_sv_red)        
       
        #check if all stability criteria are satisfied for all current poles
        if df_max is not None and dd_max is not None and d_mac is not None:
            mask_sa_all = np.logical_and(mask_sf_all , np.logical_and(mask_sd_all, mask_sv_all))
            mask_sa_all_red = np.any(mask_sa_all, axis=2)
            mask_sa = np.logical_and(mask_sa, mask_sa_all_red)
        #                        (              (             unstable) and (          stable and stable)) and (           unstable)
        mask_of = np.logical_and(np.logical_and(np.logical_not(mask_sa), np.logical_and(mask_sd, mask_sv)), np.logical_not(mask_sf))
        mask_od = np.logical_and(np.logical_and(np.logical_not(mask_sa), np.logical_and(mask_sf, mask_sv)), np.logical_not(mask_sd))
        mask_ov = np.logical_and(np.logical_and(np.logical_not(mask_sa), np.logical_and(mask_sf, mask_sd)), np.logical_not(mask_sv))
        
        # where mask is True the elements are hidden
        # unstable poles should be hidden, which are currently false
        # therefore invert all the masks
        mask_of = np.logical_not(mask_of)
        mask_od = np.logical_not(mask_od)
        mask_ov = np.logical_not(mask_ov)
        mask_sd = np.logical_not(mask_sd)
        mask_sa = np.logical_not(mask_sa)
        mask_sf = np.logical_not(mask_sf)
        mask_sv = np.logical_not(mask_sv)
        
        self.modal_frequencies.mask = np.ma.nomask
        zf_mask = self.modal_frequencies == 0 
        
        return mask_of, mask_od, mask_ov, mask_sa, zf_mask #unstable only in f,d,v...
        #return mask_sf, mask_sd, mask_sv, mask_sa, zf_mask # stable in f,d,v
    
    def update_xlim(self, xlim):
        self.ax.set_xlim(xlim)
        self.fig.canvas.draw_idle()
        
        
    def update_ylim(self, ylim):
        self.ax.set_ylim(ylim)
        self.fig.canvas.draw_idle()  
        
                
    def get_max_f(self):
        return float(np.amax(self.modal_frequencies))
    
    def get_modal_values(self,i):
        assert isinstance(i, (list, tuple))
        assert len(i)==2
        assert i[0] <= self.max_model_order
        assert i[1] <= self.max_model_order
        
        n = self.order_dummy[i]
        f = self.modal_frequencies[i]
        d = self.modal_damping[i]
        mpc = self.MPC_matrix[i]
        mp = self.MP_matrix[i]
        mpd = self.MPD_matrix[i]
        '''
        freq_diffs = self.freq_diffs[i[0],i[1],:]
        damp_diffs = self.damp_diffs[i[0],i[1],:]
        mac_diffs = self.MAC_diffs[i[0],i[1],:]
        n_ = np.ceil(n/2)
        print('f',i[0],i[1], freq_diffs[0:n_+1], self.stab_frequency, np.any(freq_diffs[0:n_] <= self.stab_frequency))
        print('d',i[0],i[1], damp_diffs[0:n_+1],  self.stab_damping, np.any(damp_diffs[0:n_] <= self.stab_damping))
        print('v',i[0],i[1], mac_diffs[0:n_+1],  self.stab_MAC, np.any(mac_diffs[0:n_] <= self.stab_MAC))
        '''
        return n,f,d,mpc, mp, mpd 
    
    def get_mode_shape(self,i):
        assert isinstance(i, (list, tuple))
        assert len(i)==2
        assert i[0] <= self.max_model_order
        assert i[1] <= self.max_model_order
        return self.mode_shapes[:, i[1], i[0]]
    
    @pyqtSlot(bool)  
    def snap_frequency(self,b=True):
        if b:
            mask=self.masks[0]                   
            self.cursor.set_mask(mask,0)
            
    @pyqtSlot(bool)          
    def snap_damping(self,b=True):
        if b:
            mask=self.masks[1]                   
            self.cursor.set_mask(mask,1)
            
    @pyqtSlot(bool)  
    def snap_vector(self, b=True):
        if b:
            mask=self.masks[2]                   
            self.cursor.set_mask(mask,2)
    @pyqtSlot(bool)         
    def snap_stable(self, b=True):
        if b:
            mask=self.masks[3]                   
            self.cursor.set_mask(mask,3)
            
    @pyqtSlot(bool)          
    def snap_all(self, b=True):
        if b:
            mask=self.masks[4]                   
            self.cursor.set_mask(mask,4)
            
    @pyqtSlot(bool)          
    def toggle_frequency(self, b):
        if len(self.stable_plot)>=1:
            self.stable_plot[0].set_visible(b)           
        self.fig.canvas.draw_idle()
        
         
    @pyqtSlot(bool)  
    def toggle_damping(self, b):
        if len(self.stable_plot)>=2:
            self.stable_plot[1].set_visible(b)       
        self.fig.canvas.draw_idle()
        
         
    @pyqtSlot(bool)      
    def toggle_vector(self, b):
        if len(self.stable_plot)>=3:
            self.stable_plot[2].set_visible(b)          
        self.fig.canvas.draw_idle()
        
         
    @pyqtSlot(bool)      
    def toggle_stable(self, b):
        if len(self.stable_plot)>=4:
            self.stable_plot[3].set_visible(b)          
        self.fig.canvas.draw_idle()
        
         
    @pyqtSlot(bool)  
    def toggle_all(self,b):        
        if len(self.stable_plot)>=5:
            self.stable_plot[4].set_visible(b)
        self.fig.canvas.draw_idle()  
               

    def update_stabilization(self, df_max=None, dd_max=None, d_mac=None, d_range=None, mpc_min=None, mp_max=None, mpd_max=None, order_range = None):
        
        self.stab_frequency = df_max if df_max is not None else self.stab_frequency
        self.stab_damping = dd_max if dd_max is not None else self.stab_damping
        self.stab_MAC = d_mac if d_mac is not None else self.stab_MAC

        self.masks = self.get_stabilization_mask(self.stab_frequency, self.stab_damping, self.stab_MAC, d_range, mpc_min, mp_max, mpd_max, order_range)

        if df_max:
            self.scatter_this(0)
        if dd_max:
            self.scatter_this(1)
        if d_mac:
            self.scatter_this(2)
        if df_max or dd_max or d_mac or d_range or mpc_min or mp_max or mpd_max:
            self.scatter_this(3)
        cursor_num_mask=self.cursor.num_mask
        self.cursor.set_mask(self.masks[cursor_num_mask],cursor_num_mask)
            
    def scatter_this(self, i):
        '''
        i    
        0    stable frequency
        1    stable damping
        2    stable mode shape
        3    stable all
        4    all
        5    psd
        
        '''
        assert i <= 5
        #colors=['b','m','g','r','grey', 'grey']
        colors=['black', 'black', 'black', 'black', 'grey', 'grey']
        #markers = ['x','d','v','o','o',None]
        
        marker_obj_1 = MarkerStyle('o')
        path_1 = marker_obj_1.get_path().transformed(
            marker_obj_1.get_transform())
        marker_obj_2 = MarkerStyle('+')
        path_2 = marker_obj_2.get_path().transformed(
            marker_obj_2.get_transform())
        path=Path.make_compound_path(path_1, path_2)


        fp=FontProperties(family='monospace', weight=0, size='large')
        markers = [TextPath((-2, -4), '\u00b7 f', prop=fp, size=10),TextPath((-2, -4), '\u00b7 d', prop=fp, size=10),TextPath((-2, -4), '\u00b7 v', prop=fp, size=10),path,'o',None]
        labels=['unstable in frequency','unstable in damping','unstable in mode shape','stable pole','spurious pole', 'psd']
        zorders=[1,1,1,2,0,0]
        sizes=[80, 80, 80, 20, 80, 80 ]
        self.modal_frequencies.mask = self.masks[i]
        self.order_dummy.mask = self.masks[i]    
        if self.stable_plot[i] is not None:
            visibility=self.stable_plot[i].get_visible()
            self.stable_plot[i].remove()
        else:
            visibility=True
        if i < 5:
            self.stable_plot[i]=self.ax.scatter(self.modal_frequencies.compressed(), 
                             self.order_dummy.compressed(), zorder=zorders[i], 
                             facecolors='none', edgecolors=colors[i],
                             marker=markers[i],
                             s=sizes[i], label=labels[i], visible=visibility)

        
        self.modal_frequencies.mask = self.masks[4]
        self.order_dummy.mask =self.masks[4]
        self.fig.canvas.draw_idle()
        
    def plot_diagram(self):

        self.fig.set_tight_layout(True)
        
        self.ax = self.fig.add_subplot(111)
        self.masks=self.get_stabilization_mask(self.stab_frequency, self.stab_damping, self.stab_MAC, d_range=(0,50))
        
        self.cursor = DataCursor(ax = self.ax, order_data = self.order_dummy, 
                                 f_data=self.modal_frequencies,
                                 color='black')

        self.fig.canvas.mpl_connect('button_press_event', self.cursor.onmove)
        self.fig.canvas.mpl_connect('resize_event', self.cursor.fig_resized)
        
        for i in range(5):
            self.scatter_this(i)
        self.stable_plot.append(False)        
        
        self.ax.set_ylim(0,self.max_model_order)
        self.ax.locator_params('y',tight=True, nbins=self.max_model_order//5)
        x_lims=self.ax.get_xlim()
        self.ax.set_xlim((0,x_lims[1]))
        
        self.ax.autoscale_view(tight=True)

        self.ax.set_xlabel('Frequency [Hz]')
        self.ax.set_ylabel('Model Order')
    
    def save_results(self, dname=None):
        if dname == '': dname = None
        selected_indices=self.cursor.datalist   
                
        if selected_indices:
            self.modal_frequencies.mask = np.ma.nomask
            self.order_dummy.mask = np.ma.nomask

            selected_freq = [self.modal_frequencies[index] for index in selected_indices]
            selected_damp = [self.modal_damping[index] for index in selected_indices]
            selected_order = [self.order_dummy[index] for index in selected_indices]
            selected_MPC = [self.MPC_matrix[index] for index in selected_indices]
            selected_MP = [self.MP_matrix[index] for index in selected_indices]
            selected_MPD = [self.MPD_matrix[index] for index in selected_indices]
            
            selected_modes = np.zeros((self.num_analised_channels, len(selected_indices)), dtype = complex)
             
            for num,ind in enumerate(selected_indices):
                row_index = ind[0]
                col_index = ind[1]
                mode_tmp = self.mode_shapes[:, col_index, row_index]
                
                #scaling of mode shape
                abs_mode_tmp = np.abs(mode_tmp)
                index_max = np.argmax(abs_mode_tmp)
                this_max = mode_tmp[index_max]
                mode_tmp = mode_tmp / this_max
                
                selected_modes[:,num] = mode_tmp
        
        freq_str = ''
        damp_str = ''
        ord_str = ''
        msh_str = ''
        mpc_str = ''
        mp_str = ''
        mpd_str = ''
        for col in range(len(selected_indices)):
            freq_str += '{:3.3f} \t\t'.format(selected_freq[col])
            damp_str += '{:3.3f} \t\t'.format(selected_damp[col])
            ord_str += '{:3d} \t\t'.format(selected_order[col])
            mpc_str += '{:3.3f}\t \t'.format(selected_MPC[col])
            mp_str += '{:3.2f} \t\t'.format(selected_MP[col])
            mpd_str += '{:3.2f} \t\t'.format(selected_MPD[col])
            
        for row in range(selected_modes.shape[0]):
            msh_str+='\n           \t\t'
            for col in range(selected_modes.shape[1]):
                msh_str+='{:+3.4f} \t'.format(selected_modes[row,col])       
        
        export_modes = 'MANUAL MODAL ANALYSIS\n'\
                      + '=======================\n'\
                      + 'Frequencies [Hz]:\t'         + freq_str       + '\n'\
                      + 'Damping [%]:\t\t'            + damp_str       + '\n'\
                      + 'Mode shapes:\t\t'            + msh_str        + '\n'\
                      + 'Model order:\t\t'            + ord_str        + '\n'\
                      + 'MPC [-]:\t\t'                + mpc_str        + '\n'\
                      + 'MP  [\u00b0]:\t\t'           + mp_str         + '\n'\
                      + 'MPD [-]:\t\t'                + mpd_str        + '\n\n'\
                      + 'SSI parameters\n'\
                      + '=======================\n'\
                      + 'Maximum order :\t\t'     + str(self.max_model_order) + '\n'\
                      + 'Block rows :\t\t'        + str(self.num_block_rows)     + '\n'\
                      + 'Block columns :\t\t'     + str(self.num_block_columns)  + '\n'
        #              + 'Decimation :\t\t'        + str(dec_fact)       + '\n'\
        #              + 'Filtering :\t\t'         + str(filt_w)
        
        # define and create a folder with all results
        if dname is not None:
            self.result_folder = dname
        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder)
        if not self.result_folder.endswith('/'):
            self.result_folder+='/'
            
        select_frequency_file = self.result_folder + 'select_frequencies.npy'
        select_damping_file = self.result_folder + 'select_damping.npy'
        select_mode_shape_file = self.result_folder + 'select_mode_shapes.npy'
        select_modes_text_file = self.result_folder + 'select_modal_info.txt'
        
        f = open(select_modes_text_file, 'w')          
        f.write(export_modes)
        f.close()
        
        np.save(select_frequency_file, selected_freq)
        np.save(select_damping_file, selected_damp)
        np.save(select_mode_shape_file, selected_modes)
        
        self.select_modes =selected_indices  
        
    @staticmethod
    def calculateMAC_sing(v1,v2):
        v1_=np.array(v1)
        v2_=np.array(v2)   
        MAC = np.abs(np.dot(v1.T, np.conj(v2))) ** 2 \
              / (np.dot(v1.T, np.conj(v1)) * 
                 np.dot(v2.T, np.conj(v2)))
        return MAC.real   
      
    @staticmethod
    def calculateMAC(v1,v2):
        
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
    def calculateMPC_sing(v):
        
        MPC_= StabilPlot.calculateMAC_sing(v, np.conjugate(v))  

        return MPC_
    
    @staticmethod
    def calculateMPC(v):
        
        MPC_ = np.abs(np.sum(v**2, axis = 0))**2 \
        / np.abs(np.einsum('ij,ij->j', v, v.conj()))**2
        
        return MPC_
    
    @staticmethod
    def calculateMPD(v, weighted=True,regression=True):
        if regression:
            #orthogonal regression through origin
            #http://mathforum.org/library/drmath/view/68362.html
            real_= np.real(v).copy()
            imag_= np.imag(v).copy()
            
            #real_-=np.mean(real_,axis=0)
            #imag_-=np.mean(imag_,axis=0)
            #MP = np.arctan(np.sum(2*real_*imag_/(real_**2-imag_**2), axis=0))
            
            ssxy = np.einsum('ij,ij->j', real_,imag_)
            ssxx = np.einsum('ij,ij->j', real_,real_)
            ssyy = np.einsum('ij,ij->j', imag_,imag_)
            
            #b = 2*ssxy / (ssxx-ssyy)
            #MP = np.arctan(b)/2     
            MP = np.arctan2(2*ssxy , (ssxx-ssyy))/2    
            
            # rotates complex plane by angle MP
            v_r = v*(np.cos(-MP)+1j*np.sin(-MP))
            #calculates phase in range -180 and 180
            phase = np.angle(v_r,True)
            # rotates into 1st and 4th quadrant
            phase[phase>90]-=180
            phase[phase<-90]+=180
            # calculates standard deviation
            
            if not weighted:
                MPD = np.std(phase, axis = 0)
            else:
                MPD = np.sqrt(np.average(np.power(phase,2), weights=np.absolute(v_r), axis = 0))
            
            MP*=180/np.pi
            MP[MP<0]+=180
            MPD[MPD<0]*=-1
            
            return MPD, MP
        else:
            phase = np.angle(v,True)
            
            phase[phase < 0] += 180
            
            if not weighted:        
                MP = np.mean(phase, axis = 0)
            else:
                MP = np.average(phase, weights= np.absolute(v), axis = 0)
            
            if not weighted:
                MPD = np.std(phase, axis = 0)
            else:
                MPD = np.sqrt(np.average(np.power(phase-MP,2), weights=np.absolute(v), axis = 0))
                
            return MPD, MP
    
class ModeShapePlot(object):
    
    def __init__(self, figure=None):
        
        super().__init__()  
        self.fig = figure
        
    def scatter_this(self, msh, mp=None):
    
        self.ax.cla()
        self.ax.scatter(msh.real, msh.imag)
        #abs_ = np.absolute(msh)
        #print(mp)
        #ang_ = np.angle(msh, True)
        #print(list(zip(abs_,ang_)))
        if mp is not None:
            while mp < 0: mp += 180
            while mp > 360:    mp -= 360
            mp = mp*np.pi/180
            xmin,xmax = -1,1
            ymin,ymax = -1,1
            if mp <= np.pi/2:
                x1=max(xmin, ymin/np.tan(mp))
                x2=min(xmax, ymax/np.tan(mp))
                y1=max(ymin, xmin*np.tan(mp))
                y2=min(ymax, xmax*np.tan(mp))
            elif mp <= np.pi:
                x1=max(xmin, ymax/np.tan(mp))
                x2=min(xmax, ymin/np.tan(mp))
                y2=max(ymin, xmax*np.tan(mp))
                y1=min(ymax, xmin*np.tan(mp))
            elif mp <= 3*np.pi/2:
                x1=max(xmin, ymin/np.tan(mp))
                x2=min(xmax, ymax/np.tan(mp))
                y1=max(ymin, xmin*np.tan(mp))
                y2=min(ymax, xmax*np.tan(mp))    
            else:
                x1=max(xmin, ymax/np.tan(mp))
                x2=min(xmax, ymin/np.tan(mp))
                y2=max(ymin, xmax*np.tan(mp))
                y1=min(ymax, xmin*np.tan(mp))
            self.ax.plot([x1,x2],[y1,y2])
        lim=max(max(abs(msh.real))*1.1,max(abs(msh.imag))*1.1)    
        self.ax.set_xlim((-lim,lim))
        self.ax.set_ylim((-lim,lim))
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

        if self.fig is None:
            self.fig = plt.figure(figsize = [20, 10.])# main plot items
            
        self.fig.set_tight_layout(True)        
        self.ax = self.fig.add_subplot(111)        
        self.ax.autoscale_view(tight=True)
        # Set the axis's spines to be centered at the given point
        # (Setting all 4 spines so that the tick marks go in both directions)
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
        
class DataCursor(Cursor, QObject): 
    # create and edit an instance of the matplotlib default Cursor widget
    
    show_current_info = pyqtSignal(tuple)
    mode_selected = pyqtSignal(tuple)
    mode_deselected = pyqtSignal(tuple)
    
    def __init__(self, ax, order_data, f_data, mask=None,  useblit=True, **lineprops):
        
        Cursor.__init__(self, ax, useblit=True, **lineprops)
        QObject.__init__(self)
        self.ax = ax
          
        self.y = order_data
        self.y.mask = np.ma.nomask
        
        self.x = f_data 
        self.x.mask = np.ma.nomask
        
        if mask is not None:
            self.mask=mask
        else:
            self.mask = np.ma.nomask
            
        self.num_mask=4
        self.i = None
    
        self.datalist = []
        self.scatter_objs=[] # that list should eventually be replaced by a matplotlib.collections collection 

        self.fig_resized()
        
    def add_datapoints(self, datalist):
        for datapoint in datalist:
            datapoint=tuple(datapoint)
            self.datalist.append(datapoint)
            #print(datapoint)
            x,y=self.x[datapoint],self.y[datapoint]
            #print(x,y)
            self.scatter_objs.append(self.ax.scatter(x,y,facecolors='none',edgecolors='red',s=200, visible=False))
            self.mode_selected.emit(datapoint) 
            
    def set_mask(self,mask, num):
        self.mask=mask
        self.fig_resized()
        self.num_mask=num
        return
        
    def fig_resized(self,event=None):
        #self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)
            
        if event is not None:
            self.width, self.height=event.width, event.height
        else:
            self.width, self.height = self.ax.get_figure().canvas.get_width_height()

        self.xpix, self.ypix = self.ax.transData.transform(np.vstack([self.x.flatten(),self.y.flatten()]).T).T       
        
        self.xpix.shape=self.x.shape
        self.xpix.mask=self.mask
        
        self.ypix.shape=self.y.shape
        self.ypix.mask=self.mask
  
    def onmove(self, event):
        if self.ignore(event):
            return
        '''
        1. Override event.data to force it to snap-to nearest data item
        2. On a mouse-click, select the data item and append it to a list of selected items
        3. The second mouse-click on a previously selected item, removes it from the list
        '''
        if (self.xpix.mask == True).all():#i.e. no stable poles
            return
            
        if event.name=="motion_notify_event":                
            
            # get cursor coordinates
            xdata  = event.xdata 
            ydata  = event.ydata
            
            if xdata is None or ydata is None:
                return
            
            xData_yData_pixels = self.ax.transData.transform(np.vstack([xdata,ydata]).T) 
                       
            xdata_pix, ydata_pix = xData_yData_pixels.T    

            self.fig_resized()
            
            self.i = self.findIndexNearestXY(xdata_pix[0],ydata_pix[0])
            xnew, ynew = self.x[self.i], self.y[self.i]
            
            if xdata==xnew and ydata==ynew:
                return

            # set the cursor and draw                
            event.xdata = xnew
            event.ydata = ynew

            self.show_current_info.emit(self.i)
            
        # select item by mouse-click only if the cursor is active and in the main plot        
        if event.name=="button_press_event" and event.inaxes==self.ax and self.i is not None:                             
            
            if not self.i in self.datalist:  
                #self.linev.set_visible(False)
                #self.lineh.set_visible(False)
                #self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)                  
                self.datalist.append(self.i)                 
                #self.ax.hold(True) # overlay plots
                # plot a circle where clicked
                self.scatter_objs.append(self.ax.scatter(self.x[self.i],self.y[self.i],facecolors='none',edgecolors='red',s=200, visible=False))
                self.mode_selected.emit(self.i)   
                #self.ax.draw_artist(self.scatter_objs[-1])

            else:
                ind=self.datalist.index(self.i)
                self.scatter_objs[ind].remove()
                del self.scatter_objs[ind]          
                self.datalist.remove(self.i)                    
                self.mode_deselected.emit(self.i)
            
            #self.ax.figure.canvas.restore_region(self.background)
            #self.ax.figure.canvas.blit(self.ax.figure.bbox)
            
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
        
        distance = np.square(self.ypix-y_point)+ np.square(self.xpix-x_point)
        index = np.argmin(distance)
        index = np.unravel_index(index, distance.shape)        
        return index
    
def nearly_equal(a,b,sig_fig=5):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )   
    
def main_stabil(modal_frequencies,modal_damping, mode_shapes, 
                 num_analised_channels=None, max_model_order=None, 
                 stab_frequency=1, stab_damping=5, stab_MAC=2,
                 measurement=None, sampling_rate=None,
                 result_folder=None, num_block_columns=0, num_block_rows=0, select_modes=[]):
    
    def handler(msg_type, msg_string):
        pass
    qInstallMsgHandler(handler)#suppress unimportant error msg

    if isinstance(modal_frequencies, str):
        modal_frequencies = np.load(modal_frequencies)
    if isinstance(modal_damping, str):
        modal_damping = np.load(modal_damping)
    if isinstance(mode_shapes, str):
        mode_shapes = np.load(mode_shapes)
    if isinstance(num_analised_channels, str):
        num_analised_channels = np.load(num_analised_channels).shape[0]

    if not 'app' in globals().keys():
        global app
        app=QApplication(sys.argv)
    if not isinstance(app, QApplication):
        app=QApplication(sys.argv)
    stabil_gui = StabilGUI(modal_frequencies,modal_damping, mode_shapes, 
                           num_analised_channels, max_model_order, 
                           stab_frequency, stab_damping, stab_MAC,
                           measurement, sampling_rate,
                           result_folder, num_block_columns, num_block_rows, select_modes)
    loop=QEventLoop()
    stabil_gui.destroyed.connect(loop.quit)
    loop.exec_()
    
    select_modes=stabil_gui.stabil_plot.select_modes

    return select_modes    
    
if __name__ =='__main__':
    main()             