'''
.. TODO::
 * Button for Axes3d.set_axis_off/on 
 * Use QTDesigner to design the GUI and rewrite the class
 * Use the logging module to replace print commands at an appropriate 
   logging level
   
'''

# system i/o
import sys
import os
import logging
logger = logging.getLogger('')

# GUI
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QPushButton,\
    QCheckBox, QButtonGroup, QLabel, QToolButton, QComboBox, QStyle,\
    QTextEdit, QGridLayout, QFrame, QVBoxLayout, QAction, \
    QFileDialog, QInputDialog, QMessageBox, QDoubleSpinBox, QTableWidget,\
    QSpinBox, QAbstractItemView, QTableWidgetItem, QApplication, QSizePolicy, QLineEdit, QTabWidget,\
    QSlider
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot, QTimer, qInstallMessageHandler, QEventLoop, QSize

# Matplotlib
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backend_bases import FigureCanvasBase

#tools
from GUI.Helpers import DelayedDoubleSpinBox, my_excepthook
sys.excepthook = my_excepthook
NoneType = type(None)

from classes.PlotMSH import ModeShapePlot

from copy import deepcopy


def nearly_equal(a,b,sig_fig=5):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )

old_resize_event = deepcopy(FigureCanvasQTAgg.resizeEvent)

def resizeEvent_(self, event):
    w = event.size().width()
    h = event.size().height()
    dpival = self.figure.dpi
    winch, hinch = self.figure.get_size_inches()
    aspect = winch/hinch
    if w/h <= aspect:               
        h = w/aspect
    else:               
        w = h *aspect
    winch = w/dpival
    hinch = h/dpival
    self.figure.set_size_inches(winch, hinch)
    FigureCanvasBase.resize_event(self)
    self.draw()
    self.update() 
    QWidget.resizeEvent(self, event)
       
class ModeShapeGUI(QMainWindow):
    '''
    A class for interacting with PlotMSH.ModeShapePlot
    '''
    
    # define this class's signals and the types of data they emit
    grid_requested = pyqtSignal(str, bool)
    beams_requested = pyqtSignal(str, bool)
    slaves_requested = pyqtSignal(str, bool)
    chan_dofs_requested = pyqtSignal(str, bool)

    def __init__(self,
                 mode_shape_plot,      
                 reduced_gui=False):

        QMainWindow.__init__(self)
        assert isinstance(mode_shape_plot, ModeShapePlot)
        self.mode_shape_plot = mode_shape_plot
        self.animated = False
        self.setWindowTitle('Plot Modeshapes')
        self.create_menu()
        self.create_main_frame(mode_shape_plot, reduced_gui)
        self.setGeometry(300, 300, 1000, 600)
        self.show()
            
    def create_main_frame(self, mode_shape_plot, reduced_gui=False):
        '''
        set up all the widgets and other elements to draw the GUI
        '''
        main_frame = QWidget()
        
        # Create the mpl Figure and FigCanvas objects.
        fig=mode_shape_plot.fig
        FigureCanvasQTAgg.resizeEvent = resizeEvent_
        self.canvas = fig.canvas.switch_backends(FigureCanvasQTAgg)
        #self.canvas.resize_event = resizeEvent_
        #self.canvas.resize_event  = funcType(resizeEvent_, self.canvas, FigureCanvasQTAgg)
        mode_shape_plot.canvas = self.canvas
        self.canvas.mpl_connect('button_release_event', self.update_lims)
        fig.get_axes()[0].mouse_init()

        #controls for changing what to draw
        view_layout = QHBoxLayout()
        
        view_layout.addStretch()
        self.axis_checkbox = QCheckBox('Show Axis Arrows')
        self.axis_checkbox.setTristate(False)
        self.axis_checkbox.setCheckState(Qt.Checked if mode_shape_plot.show_axis else Qt.Unchecked)
        self.axis_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_axis)
        
        self.nodes_checkbox = QCheckBox('Show Nodes')
        self.nodes_checkbox.setTristate(False)
        self.nodes_checkbox.setCheckState(Qt.Checked if mode_shape_plot.show_nodes else Qt.Unchecked)
        self.nodes_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_nodes)
        
        line_checkbox = QCheckBox('Show Lines')
        line_checkbox.setTristate(False)
        conn_lines_checkbox = QCheckBox('Show Connecting Lines')
        conn_lines_checkbox.setTristate(False)
        conn_lines_checkbox.setCheckState(Qt.Checked if mode_shape_plot.show_nd_lines else Qt.Unchecked)
        conn_lines_checkbox.stateChanged[int].connect(mode_shape_plot.refresh_nd_lines)
        
        ms_checkbox = QCheckBox('Show Master-Slaves Assignm.')
        ms_checkbox.setTristate(False)
        chandof_checkbox = QCheckBox('Show Channel-DOF Assignm.')
        chandof_checkbox.setTristate(False)
            
            
        self.draw_button_group = QButtonGroup()
        self.draw_button_group.setExclusive(False)
        self.draw_button_group.addButton(line_checkbox, 0)
        self.draw_button_group.addButton(ms_checkbox, 1)
        self.draw_button_group.addButton(chandof_checkbox, 2)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)
        
        if mode_shape_plot.show_lines:
            line_checkbox.setCheckState(Qt.Checked)
        elif mode_shape_plot.show_master_slaves:
            ms_checkbox.setCheckState(Qt.Checked)
        elif mode_shape_plot.show_chan_dofs:
            chandof_checkbox.setCheckState(Qt.Checked)
            
        view_layout.addWidget(self.axis_checkbox)
        view_layout.addWidget(self.nodes_checkbox)
        view_layout.addWidget(line_checkbox)
        view_layout.addWidget(ms_checkbox)
        view_layout.addWidget(chandof_checkbox)
        view_layout.addWidget(conn_lines_checkbox)

        # controls for changing the axis' limits and viewport i.e. zoom and shift
        axis_limits_layout = QGridLayout()
        
        

        
        
            
        # Buttons for creating/editing geometry and loading solutions
        #grid_button = QPushButton('Edit Grid')
        #grid_button.released.connect(self.stop_ani)
        #grid_button.released.connect(self.geometry_creator.load_nodes)

        #beam_button = QPushButton('Edit Beams')
        #beam_button.released.connect(self.stop_ani)
        #beam_button.released.connect(self.geometry_creator.load_lines)

        #ms_button = QPushButton('Edit Master Slaves')
        #ms_button.released.connect(self.stop_ani)
        #ms_button.released.connect(self.geometry_creator.load_master_slave)

        #cd_button = QPushButton('Edit Channel-DOFS-Assignment')
        #cd_button.released.connect(self.stop_ani)
        #cd_button.released.connect(self.geometry_creator.load_chan_dof)

        #ssi_button = QPushButton('Load Modal Data')
        #ssi_button.released.connect(self.stop_ani)
        #ssi_button.released.connect(self.reload_ssi_solutions)
        
        # GUI controls for selecting modes and changing various 
        # values for drawing the modeshapes
        #self.order_combo = QComboBox()
        #self.order_combo.currentIndexChanged[str].connect(self.change_order)
        #textbox for showing information about the currently displayed mode
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        
        
        self.mode_combo = QComboBox()
        frequencies = ['{}: {}'.format(i+1,f) for i,f in enumerate(self.mode_shape_plot.get_frequencies())]
        #print(frequencies)
        if frequencies:
            self.mode_combo.addItems(frequencies)
            self.mode_combo.currentIndexChanged[str].connect(self.change_mode)
        else:
            self.mode_combo.setEnabled(False) 
        
        
        

        self.amplitude_box = DelayedDoubleSpinBox()
        self.amplitude_box.setRange(0, 1000000000)
        self.amplitude_box.setValue(mode_shape_plot.amplitude)
        self.amplitude_box.valueChangedDelayed.connect(mode_shape_plot.change_amplitude)

        real_checkbox = QCheckBox('Magn.')
        real_checkbox.setTristate(False)
        

        imag_checkbox = QCheckBox('Magn.+Phase')
        imag_checkbox.setTristate(False)
        

        real_imag_group = QButtonGroup()
        real_imag_group.addButton(real_checkbox,0)
        real_imag_group.addButton(imag_checkbox,1)        
        real_imag_group.setExclusive(True)
        #print(real_imag_group.exclusive(), real_imag_group.checkedId())
        imag_checkbox.setCheckState(Qt.Unchecked if mode_shape_plot.real else Qt.Checked)
        
        #print(real_imag_group.exclusive(), real_imag_group.checkedId())
        real_checkbox.setCheckState(Qt.Checked if mode_shape_plot.real else Qt.Unchecked)
        
        #print(real_imag_group.exclusive(), real_imag_group.checkedId())
        
        self.test_=real_imag_group
        real_checkbox.stateChanged[int].connect(self.mode_shape_plot.change_part)
        #real_checkbox.stateChanged[int].connect(self.test)
        #plot_button = QPushButton('Draw')
        #plot_button.released.connect(self.draw_msh)

        self.ani_button = QToolButton()
        self.ani_button.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPlay))
        self.ani_button.setToolTip("Play")
        self.ani_button.released.connect(self.animate)
        
        if mode_shape_plot.prep_data is not None:
            self.ani_lowpass_box = DelayedDoubleSpinBox()
            self.ani_lowpass_box.setRange(0, 1000000000)
            self.ani_lowpass_box.valueChangedDelayed.connect(self.prepare_filter)
            
            self.ani_highpass_box = DelayedDoubleSpinBox()
            self.ani_highpass_box.setRange(0, 1000000000)
            self.ani_highpass_box.valueChangedDelayed.connect(self.prepare_filter)
            
            self.ani_speed_box = QDoubleSpinBox()        
            self.ani_speed_box.setRange(0, 1000000000)
            self.ani_speed_box.valueChanged[float].connect(self.change_animation_speed)
            
            self.ani_position_slider = QSlider(Qt.Horizontal)
            self.ani_position_slider.setRange(0,mode_shape_plot.prep_data.measurement.shape[0])
            self.ani_position_slider.valueChanged.connect(self.set_ani_time) 
            self.ani_position_data = QLineEdit()
        
        self.ani_data_button = QToolButton()
        self.ani_data_button.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPlay))
        self.ani_data_button.setToolTip("Play")
        self.ani_data_button.released.connect(self.filter_and_animate_data)
        

        
        #put everything in layouts
        controls_layout = QGridLayout()
#         controls_layout.addWidget(grid_button, 0, 0)
#         controls_layout.addWidget(beam_button, 1, 0)
#         controls_layout.addWidget(ms_button, 2, 0)
#         controls_layout.addWidget(cd_button, 3, 0)
#         controls_layout.addWidget(ssi_button, 4, 0)
        
        

#         sep = QFrame()
#         sep.setFrameShape(QFrame.VLine)
# 
#         controls_layout.addWidget(sep, 0, 1, 5, 1)
        
        controls_layout.addWidget(QLabel('Change Viewport:'),0,2,1,2)
        hbox=QHBoxLayout()
        for i,view in enumerate(['X', 'Y', 'Z', 'ISO']):
            button = QToolButton()
            button.setText(view)
            button.released.connect(self.change_viewport)
            hbox.addWidget(button)
        hbox.addStretch()
        controls_layout.addLayout(hbox, 0, 4,1,4)
        
        self.val_widgets={}
        lims = self.mode_shape_plot.subplot.get_w_lims()
        for row,dir in enumerate(['X', 'Y', 'Z']):
            label = QLabel(dir+' Limits:')
            r_but = QToolButton()
            r_but.setText('<-')
            r_but.released.connect(self.change_view)
            r_val = QLineEdit()
            r_val.setText(str(lims[row*2+0]))
            r_val.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            r_val.editingFinished.connect(self.change_view)
            l_val = QLineEdit()
            l_val.setText(str(lims[row*2+1]))
            l_val.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            l_val.editingFinished.connect(self.change_view)
            l_but = QToolButton()
            l_but.setText('->')
            l_but.released.connect(self.change_view)            
            self.val_widgets[dir]=[r_but,r_val,l_val,l_but]
            controls_layout.addWidget(label,row+1,0+2)
            controls_layout.addWidget(r_but,row+1,1+2)
            controls_layout.addWidget(r_val,row+1,2+2)
            controls_layout.addWidget(l_val, row+1,3+2)
            controls_layout.addWidget(l_but, row+1, 4+2)
        #controls_layout.setColumnStretch(5,10)
        
        label = QLabel('Zoom:')
        r_but = QToolButton()
        r_but.setText('+') 
        r_but.released.connect(self.change_view)
        l_but = QToolButton()
        l_but.setText('-')
        l_but.released.connect(self.change_view)  
        
        controls_layout.addWidget(label, row+2,0+2)
        controls_layout.addWidget(r_but, row+2,1+2)
        controls_layout.addWidget(l_but, row+2,2+2)
        
        reset_button = QPushButton('Reset View')
        reset_button.released.connect(self.reset_view)

        controls_layout.addWidget(reset_button, row+2,3+2)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)

        tab_widget = QTabWidget()

        tab_1 = QWidget()
        lay_1 = QGridLayout()
        tab_2 = QWidget()
        lay_2 = QGridLayout()
        #lay_1.setContentsMargins(0,0,0,0)
        tab_1.setContentsMargins(0,0,0,0)
        #lay_2.setContentsMargins(0,0,0,0)
        tab_2.setContentsMargins(0,0,0,0)
        tab_widget.setContentsMargins(0,0,0,0)
        #lay_1.setVerticalSpacing(0)
        #lay_2.setVerticalSpacing(0)
        
        controls_layout.addWidget(sep, 0, 7, 5, 1)

        lay_1.addWidget(QLabel('Mode'), 0,0)
        lay_1.addWidget(self.mode_combo, 0,1)

        lay_1.addWidget(QLabel('Amplitude'), 1,0)
        lay_1.addWidget(self.amplitude_box, 1,1)

        layout = QHBoxLayout()
        lay_1.addWidget(QLabel('Complex Modeshape:'), 2,0)
        layout.addWidget(real_checkbox)
        layout.addWidget(imag_checkbox)
        lay_1.addLayout(layout, 2,1)

        layout = QHBoxLayout()
        #layout.addWidget(QLabel('Show Modeshape:'))
        #layout.addWidget(self.ani_button)        
        #lay_1.addLayout(layout, 3,0,0,1)
        
        lay_1.addWidget(self.ani_button,3,0,)
        tab_1.setLayout(lay_1)
        if mode_shape_plot.prep_data is not None:
            lay_2.addWidget(QLabel('Lowpass [Hz]:'),0,0)
            lay_2.addWidget(self.ani_lowpass_box,0,1)
            lay_2.addWidget(QLabel('Highpass [Hz]:'),1,0)
            lay_2.addWidget(self.ani_highpass_box)
            lay_2.addWidget(QLabel('Animation Speed [ms]:'),2,0)
            lay_2.addWidget(self.ani_speed_box,2,1)
            #lay_2.addWidget(self.ani_data_button,3,0,)
            layout = QHBoxLayout()
            layout.addWidget(self.ani_data_button)
            layout.addWidget(self.ani_position_slider)
            layout.addWidget(self.ani_position_data)
            lay_2.addLayout(layout, 3,0,1,2)
        tab_2.setLayout(lay_2)
        
        policy = QSizePolicy.Minimum
        tab_1.setSizePolicy(policy,policy)
        tab_2.setSizePolicy(policy,policy)
        tab_widget.setSizePolicy(policy, policy)

        tab_widget.addTab(tab_1, 'Modeshape')
        tab_widget.addTab(tab_2, 'Time Histories')
        
        controls_layout.addWidget(tab_widget, 0, 8, 5, 1)
        
        if not reduced_gui:
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            controls_layout.addWidget(sep, 0, 9, 5, 1)        
            controls_layout.addWidget(self.info_box, 0, 10, 5, 2)

        vbox = QVBoxLayout()
        
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)

        vbox.addWidget(self.canvas,10,Qt.AlignCenter)

        

        vbox.addWidget(sep1)    
        vbox.addLayout(view_layout)
        vbox.addLayout(axis_limits_layout)   
        vbox.addWidget(sep2)     
        vbox.addLayout(controls_layout)

        main_frame.setLayout(vbox)
        self.setCentralWidget(main_frame)

        self.show()
        #self.reset_view()
        self.mode_combo.setCurrentIndex(1)
        imag_checkbox.setChecked(True)
        self.mode_combo.setCurrentIndex(0)
        
        
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
                                         slot=self.save_plot,
                                         tip="Save the plot")
        quit_action = create_action("&Quit", 
                                    slot=self.close,
                                    shortcut="Ctrl+Q", 
                                    tip="Close the application")

        add_actions(file_menu,
                    (load_file_action, None, quit_action))

        help_menu = self.menuBar().addMenu("&Help")
    
    def reset_view(self):
        self.stop_ani()
        self.mode_shape_plot.reset_view()
        self.axis_checkbox.setChecked(True)
        self.nodes_checkbox.setChecked(True)
        self.draw_button_group.button(0).setChecked(True)
        self.toggle_draw(0)
        lims=self.mode_shape_plot.subplot.get_w_lims()

        self.val_widgets['X'][1].setText(str(lims[0]))
        self.val_widgets['X'][2].setText(str(lims[1]))        
        self.val_widgets['Y'][1].setText(str(lims[2]))
        self.val_widgets['Y'][2].setText(str(lims[3]))        
        self.val_widgets['Z'][1].setText(str(lims[4]))
        self.val_widgets['Z'][2].setText(str(lims[5]))
            
    ##@pyqtSlot()
    def change_view(self):
        '''
        shift the view along specified axis by +-20 % (hardcoded)
        works in combination with the appropriate buttons as senders
        or by passing one of  ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        '''
        sender = self.sender()
        for dir, widgets in self.val_widgets.items():
            for i,widget in enumerate(widgets):
                if sender == widget:
                    break
            else:
                continue
            val1, val0 = float(widgets[2].text()), float(widgets[1].text())
            rang = val1-val0
            if i == 0:
                val0 -=rang*0.2
                val1 -=rang*0.2
            elif i == 3:
                val0 +=rang*0.2
                val1 +=rang*0.2
            widgets[2].setText(str(val1))
            widgets[1].setText(str(val0))
            if 'X' in dir:
                self.mode_shape_plot.subplot.set_xlim3d((val0,val1))
            elif 'Y' in dir:
                self.mode_shape_plot.subplot.set_ylim3d((val0,val1))
            elif 'Z' in dir:
                self.mode_shape_plot.subplot.set_zlim3d((val0,val1))
        else:
            val1, val0 = float(widgets[2].text()), float(widgets[1].text())
            rang = val1-val0
            if sender.text() == '+':
                rang *=0.8
            elif sender.text() == '-':
                rang*=1.25
                
        for dir, widgets in self.val_widgets.items():
            val1_, val0_ = float(widgets[2].text()), float(widgets[1].text())
            rang_ = val1_-val0_
            delta = (rang_ - rang)/2
            val1_ -= delta
            val0_ += delta
            widgets[2].setText(str(val1_))
            widgets[1].setText(str(val0_))    
            if 'X' in dir:
                self.mode_shape_plot.subplot.set_xlim3d((val0_,val1_))
            elif 'Y' in dir:
                self.mode_shape_plot.subplot.set_ylim3d((val0_,val1_))
            elif 'Z' in dir:
                self.mode_shape_plot.subplot.set_zlim3d((val0_,val1_))
        self.mode_shape_plot.canvas.draw_idle()
        
    def update_lims(self, event):
        if event.button == 3:
            lims = self.mode_shape_plot.subplot.get_w_lims()
            for row,dir in enumerate(['X', 'Y', 'Z']):
                [r_but,r_val,l_val,l_but]=self.val_widgets[dir]
                r_val.setText(str(lims[row*2+0]))
                l_val.setText(str(lims[row*2+1]))
        
                

            
    #@pyqtSlot()
    def change_viewport(self, viewport=None):
        '''
        change the viewport
        for non-ISO viewports the projection methods of matplotlib
        will be monkeypatched, because otherwise it would not be an 
        axonometric view (functions are defined at the top of document)
        works in combination with the appropriate buttons as senders or
        by passing one of ['X', 'Y', 'Z', 'ISO']
        
        '''
        if viewport is None:
            viewport = self.sender().text()
            
        self.mode_shape_plot.change_viewport(viewport)
                    
    #@pyqtSlot()
    def save_plot(self, path=None):
        '''
        save the curently displayed frame as a \*.png graphics file
        '''
        
        # copied and modified from matplotlib.backends.backend_qt4.NavigationToolbar2QT
        canvas=self.canvas
        
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
        
        
        fname,ext = QFileDialog.getSaveFileName(self, caption="Choose a filename to save to",
                                       directory=start, filter=filters)

        if fname:
            self.mode_shape_plot.save_plot(fname)
            self.statusBar().showMessage('Saved to %s' % fname, 2000)
            
    def plot_this(self, index):
        #self.mode_shape_plot.stop_ani()
        self.mode_shape_plot.change_mode(mode_index=index)
        #self.animate()
        
    #@pyqtSlot(str)
    def change_mode(self, mode):
        '''
        if user selects a new mode,
        extract the mode number from the passed string (contains frequency...)
        write modal values to the infobox
        and plot the mode shape
        '''
        
        #print('in change_mode: mode = ', mode)

        # mode numbering starts at 1 python lists start at 0
        mode_num = mode.split(':') # if mode is empty
        if not mode_num[0]: return
        
        mode_num = int(float(mode_num[0])) - 1 
        frequency = float(mode.split(':')[1])
        mode,order,frequency, damping, MPC, MP, MPD = self.mode_shape_plot.change_mode(frequency)
        
        text = 'Selected Mode\n'\
                      + '=======================\n'\
                      + 'Frequency [Hz]:\t'           + str(frequency)    + '\n'\
                      + 'Damping [%]:\t'            + str(damping)    + '\n'\
                      + 'Model order:\t'            + str(order)      + '\n'\
                      + 'Mode number: \t'           + str(mode)       + '\n'\
                      + 'MPC [-]:\t'                + str(MPC)        + '\n'\
                      + 'MP  [\u00b0]:\t'           + str(MP)         + '\n'\
                      + 'MPD [-]:\t'                + str(MPD)        + '\n\n'
        #print(text)             
        self.info_box.setText(text)


    #@pyqtSlot(int)
    def toggle_draw(self, i):
        '''
        helper function to receive the signal from the draw_button_group
        i is the number of the button that had it's state changed
        based on i and the checkstate the appropriate functions will be called
        '''
        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.mode_shape_plot.refresh_lines(False)
        self.mode_shape_plot.refresh_master_slaves(False)            
        self.mode_shape_plot.refresh_chan_dofs(False) 
        if self.draw_button_group.button(i).checkState():
            for j in range(3):
                if j==i:continue
                self.draw_button_group.button(j).setCheckState(Qt.Unchecked)
            if i == 0:
                self.mode_shape_plot.refresh_lines(True)
            elif i == 1:
                self.mode_shape_plot.refresh_master_slaves(True)
            elif i == 2:
                self.mode_shape_plot.refresh_chan_dofs(True)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)


    #@pyqtSlot()
    def stop_ani(self):
        '''
        convenience method to stop the animation and restore the still plot
        '''
        if self.animated:
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
            self.animated = False

    #@pyqtSlot()
    def animate(self):
        '''
        create necessary objects to animate the currently displayed
        deformed structure
        '''
        if self.mode_shape_plot.animated:
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
            self.mode_shape_plot.stop_ani()
        else:
            if self.mode_shape_plot.data_animated:
                self.ani_data_button.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
                self.mode_shape_plot.stop_ani()
            self.nodes_checkbox.setCheckState(False)
            #self.axis_checkbox.setCheckState(False)
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
            self.mode_shape_plot.animate()
    
    def prepare_filter(self):
        lowpass = self.ani_lowpass_box.value()
        highpass = self.ani_highpass_box.value()
        #print(lowpass, highpass)
        try:
            lowpass = float(lowpass)
        except ValueError:
            lowpass = None
        try:
            highpass = float(highpass)
        except ValueError:
            highpass = None
        
        if lowpass == 0.0:
            lowpass = None
        if highpass == 0.0:
            highpass = None
        if lowpass and highpass:
            assert lowpass > highpass
        #print(highpass, lowpass)
        self.mode_shape_plot.prep_data.filter_data(lowpass, highpass)
    
    def set_ani_time(self, pos):
        #print(pos)
        tot_len = self.mode_shape_plot.prep_data.measurement.shape[0]
        #pos = int(pos*tot_len)
        self.mode_shape_plot.line_ani.frame_seq = iter(range(pos,tot_len))
        
    def change_animation_speed(self, speed):
        try:
            speed = float(speed)
        except ValueError:
            return
        
        #print(speed)
        self.mode_shape_plot.line_ani.event_source.interval = int(speed)
        self.mode_shape_plot.line_ani.event_source._timer_set_interval()
         
    #@pyqtSlot()
    def filter_and_animate_data(self):
        '''
        create necessary objects to animate the currently displayed
        deformed structure
        '''
        if self.mode_shape_plot.data_animated:
            self.ani_data_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
            self.mode_shape_plot.stop_ani()
        else:
            if self.mode_shape_plot.animated:
                self.ani_button.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
                self.mode_shape_plot.stop_ani()
            self.nodes_checkbox.setCheckState(False)
            self.axis_checkbox.setCheckState(False)
            self.ani_data_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
            self.mode_shape_plot.filter_and_animate_data(callback=self.ani_position_data.setText)
            
    def closeEvent(self, *args, **kwargs):
        self.mode_shape_plot.stop_ani()
        FigureCanvasQTAgg.resizeEvent = old_resize_event
        self.deleteLater()
        return QMainWindow.closeEvent(self, *args, **kwargs)   



def start_msh_gui(mode_shape_plot):
    
    def handler(msg_type, msg_string):
        pass
    
    #qInstallMessageHandler(handler)#suppress unimportant error msg
    if not 'app' in globals().keys():
        global app
        app=QApplication(sys.argv)
    if not isinstance(app, QApplication):
        app=QApplication(sys.argv)

    form = ModeShapeGUI(mode_shape_plot)

    loop=QEventLoop()
    form.destroyed.connect(loop.quit)
    loop.exec_()
    #FigureCanvasQTAgg.resize_event=old_resize_event
    return

if __name__ == "__main__":
    pass