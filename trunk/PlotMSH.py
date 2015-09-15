# GUI
#from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QMainWindow, QWidget, QHBoxLayout, QPushButton,\
    QCheckBox, QButtonGroup, QLabel, QToolButton, QComboBox, QStyle,\
    QTextEdit, QGridLayout, QFrame, QVBoxLayout, QAction, QIcon,\
    QFileDialog, QInputDialog, QMessageBox, QDoubleSpinBox, QTableWidget,\
    QSpinBox, QAbstractItemView, QTableWidgetItem, QApplication
from PyQt4.QtCore import pyqtSignal, Qt, pyqtSlot, QTimer
# Matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import is_color_like
from matplotlib.animation import FuncAnimation
from matplotlib.markers import MarkerStyle
# system i/o
import sys
import csv
import os
import shelve
#math
import numpy as np
from math import cos, pi, fmod
#tools
import itertools
from numpy import disp



def nearly_equal(a,b,sig_fig=5):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )

# monkeypatch to allow orthogonal projection in mplot3d
# breaks automatic placement of axis
# credit: https://github.com/matplotlib/matplotlib/issues/537
def orthogonal_proj(zfront, zback):
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, a, b],
                        [0, 0, 0, zback]])

# copy of mpl_toolkits.mplot3d.proj3d.persp_transformation
# for restoring the projection in isonometriv view
def persp_transformation(zfront, zback):
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, a, b],
                        [0, 0, -1, 0]
                        ])

class PlotMSH(QMainWindow):
    '''
    This class is used for displaying modal values and modeshapes obtained
    by the SSICovRef class by Mihai-Andrei Udrea 2013 
    (Bauhaus-Universität Weimar, Institut für Strukturmechanik). An
    interactive GUI based on PyQt4 is used to create such plots.

    Abilities include:

    Drawing:
    - creation of 3d plots using matplotlib's mplot3 from the 
        matplotlib toolkit
    - adjusting axis limits for each of the three axis
        i.e. zoom view, shift view (along single and multiple axis')
    - change the viewport e.g. x, y, z and isometric view
    - rotating and zooming through mouse interaction is currently 
        supported by matplotlib, whereas panning is not
    - animate the currently displayed deformed structure
    - save the still frame


    Geometry definition:
    - draw single and multiple nodes (deformed and undeformed)
    - draw single and multiple lines (deformed and undeformed)
    - draw single and multiple master-slave assignments onto the nodes 
        (undeformed only)
    - draw single and multiple channel-degree of freedom assignments 
        onto the nodes (undeformed only
    - initiate creation/editing/loading/saving of such geometric information

    SSI Solutions:
    - load a SSI_solutions file
    - extract and display the following from the SSI_solutions file (*.slv):
        - available orders
        - available modes for selected order
        - modal values for selected mode and order
            frequency, damping, eigenvalue
        - mode shapes for selected mode and order
    - currently modeshapes are normalized to unit modal displacement by default

    currently __not__ supported:
    - 3D surface plots, as they are not properly supported by the 
        underlying matplotlib api
    - combination of several modeshapes or sensor setups 
        (this could be done easily in an external script)
    - saving of the animation as a movie file
    - drawing multiple modeshapes into one plot
    - plot modeshape in a single call from a script i.e. use static methods
    '''
    # define this class's signals and the types of data they emit
    grid_requested = pyqtSignal(str, bool)
    beams_requested = pyqtSignal(str, bool)
    slaves_requested = pyqtSignal(str, bool)
    chan_dofs_requested = pyqtSignal(str, bool)

    def __init__(self,
                 grid_path=None,
                 beam_path=None,
                 slave_path=None,
                 chan_dofs_path=None,
                 ssi_solutions_path=None,
                 mode_num=1,
                 order_num=1,
                 amplitude=1,
                 size=(5.0, 4.0),
                 dpi=100,
                 nodecolor='blue',
                 nodemarker='o',
                 nodesize=20,
                 beamcolor='blue',
                 beamstyle='-',
                 modecolor='blue',
                 modestyle='-'
                 ):
        '''
        conventions:

        - chan_dofs=[(chan, node, (x_amplif,y_amplif,z_amplif)),...]

        - channels = 0 ... #, starting at channel 0, should be a complete sequence

        - nodes = 1 ... #, starting at node 1, can be a sequence with missing entries

        - lines = [(node_start, node_end),...], unordered

        - master_slaves = [(node_master, x_master, y_master, z_master, 
                            node_slave, x_slave, y_slave, z_slave),...], unordered

        - order_nr = 0, 1, ..., 10, 11, ... 99, 100, 101, ... ,
             i.e. integers with variable width
            this is the convention taken from SSIcovRef as of 10/21/2014

        - mode_nr = 1 ... #
            actually in SSIcovRef modes are in a list which starts at 
            index 0 this will be compensated in the code


        - SSI_Solutions.slv     -    data structure
            |---Timestamp [int](usually only one)
            |   |---'Modal values'     [str]
            |   |    |---order_1         [str]
            |   |    |   |---frequencies      [list-of-floats]
            |   |    |   |---damping         [list-of-floats]
            |   |    |   |---eigenvalues      [list-of-complex]
            |   |    |---order_2         [str]
            |   |    | ...
            |   |---'Complex conjugate'[str]
            |   |    |----order_#         [list-of-int in a single list]
            |   |    | ...
            |   |---'Mode shapes'
            |   |    |----order_#         [list of lists-of-complex]
            |   |    | ...

        (Load it into VisualizeGui for browsing the dict's structure!)
        (If you get an error bad_magic_number or segmentation fault you 
        cannot open the shelves on the computer you are currently using
        It has been tested to work on arwen.bauing.uni-weimar.de, but this
        depends on were the shelve file was created and may not always work)
        '''
        QMainWindow.__init__(self)
        
        # linestyles available in matplotlib
        styles = ['-', '--', '-.', ':', 'None', ' ', '', None]
        
        #markerstylesavailable in matplotlib
        markers = list(MarkerStyle.markers.keys())

        if grid_path is not None:
            assert os.path.exists(grid_path)
        self.grid_path = grid_path

        if beam_path is not None:
            assert os.path.exists(beam_path)
        self.beam_path = beam_path

        if slave_path is not None:
            assert os.path.exists(slave_path)
        self.slave_path = slave_path

        if chan_dofs_path is not None:
            assert os.path.exists(chan_dofs_path)
        self.chan_dofs_path = chan_dofs_path

        #if ssi_solutions_path is not None:
        #    assert os.path.exists(ssi_solutions_path)
        self.ssi_solutions_path = ssi_solutions_path

        assert isinstance(mode_num, int)
        # mode numbering starts at 1, python lists start at 0
        self.mode_num = mode_num - 1

        assert isinstance(order_num, int)
        self.order_num = order_num

        assert isinstance(amplitude, (int, float))
        self.amplitude = amplitude

        assert is_color_like(beamcolor)
        self.beamcolor = beamcolor

        assert beamstyle in styles
        self.beamstyle = beamstyle

        assert is_color_like(nodecolor)
        self.nodecolor = nodecolor

        assert nodemarker in markers or \
            (isinstance(nodemarker, (tuple)) and len(nodemarker) == 3)
        self.nodemarker = nodemarker

        assert isinstance(nodesize, (float, int))
        self.nodesize = nodesize

        assert is_color_like(modecolor)
        self.modecolor = modecolor

        assert modestyle in styles
        self.modestyle = modestyle

        assert isinstance(dpi, int)
        self.dpi = dpi

        assert isinstance(size, (tuple, list))
        assert len(size) == 2
        self.size = size

        assert isinstance(amplitude, (int, float))
        self.amplitude = amplitude

        # objects
        #self.ssi_solutions_dict = {}
        self.complex_part = 'real'
        self.animated = False

        # geometry definitions
        self.xs, self.ys, self.zs = {}, {}, {}  # nodes
        self.xd, self.yd, self.zd = {}, {}, {}  # deflected nodes
        #self.xs, self.ys, self.zs = [], [], []  # nodes
        #self.xd, self.yd, self.zd = [], [], []  # deflected nodes
        self.lines = []  # beams
        self.master_slaves = []  # master-slave definitions
        self.chan_dofs = []  # channel degree of freedom assignments

        # plot objects
        self.patches_objects = {}
        self.lines_objects = []
        self.arrows_objects = []
        self.channels_objects = []
        self.axis_obj = {}

        # GUI for geometry creation
        self.geometry_creator = GeometryCreator()
        self.geometry_creator.node_added.connect(self.add_node)
        self.geometry_creator.node_taken.connect(self.take_node)
        self.geometry_creator.line_added.connect(self.add_line)
        self.geometry_creator.line_taken.connect(self.take_line)
        self.geometry_creator.slave_added.connect(self.add_master_slave)
        self.geometry_creator.slave_taken.connect(self.take_master_slave)
        self.geometry_creator.chan_dof_added.connect(self.add_chan_dof)
        self.geometry_creator.chan_dof_taken.connect(self.take_chan_dof)

        self.grid_requested.connect(self.geometry_creator.load_nodes)
        self.beams_requested.connect(self.geometry_creator.load_lines)
        self.slaves_requested.connect(self.geometry_creator.load_master_slave)
        self.chan_dofs_requested.connect(self.geometry_creator.load_chan_dof)

        self.setWindowTitle('Plot Modeshapes')
        self.create_menu()
        self.create_main_frame(size, dpi)
        self.setGeometry(300, 300, 1000, 600)
        self.show()

        if self.grid_path is not None:
            self.grid_requested.emit(self.grid_path, True)
        if self.beam_path is not None:
            self.beams_requested.emit(self.beam_path, True)
        if self.slave_path is not None:
            self.slaves_requested.emit(self.slave_path, True)
        if self.chan_dofs_path is not None:
            self.chan_dofs_requested.emit(self.chan_dofs_path, True)
        if self.ssi_solutions_path is not None:
            self.load_ssi_solutions()
            
        self.draw_lines(dict.fromkeys(self.xs.keys(), 0),dict.fromkeys(self.ys.keys(), 0),dict.fromkeys(self.zs.keys(), 0), beamstyle='dotted')
        self.draw_lines(dict.fromkeys(self.xs.keys(), 0),dict.fromkeys(self.ys.keys(), 0),dict.fromkeys(self.zs.keys(), 0), beamstyle='dotted')
            
    def create_main_frame(self, size, dpi):
        '''
        set up all the widgets and other elements to draw the GUI
        '''
        
        main_frame = QWidget()

        # Create the mpl Figure and FigCanvas objects.
        self.fig = Figure(size, dpi=dpi, facecolor='white')
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.canvas.setParent(main_frame)

        try:  # mpl 1.4
            self.subplot = self.fig.add_subplot(1, 1, 1, projection='3d')
        except ValueError:  # mpl 1.3
            self.subplot = Axes3D(self.fig)
        self.subplot.set_axis_off()
        #controls for changing what to draw
        view_layout = QHBoxLayout()

        reset_button = QPushButton('Reset View')
        reset_button.released.connect(self.reset_view)

        view_layout.addWidget(reset_button)
        
        view_layout.addStretch()
        
        self.axis_checkbox = QCheckBox('Show Axis Arrows')
        self.axis_checkbox.setCheckState(Qt.Checked)
        self.axis_checkbox.stateChanged[int].connect(self.draw_axis)
        
        self.grid_checkbox = QCheckBox('Show Grid')
        self.grid_checkbox.setCheckState(Qt.Checked)
        self.grid_checkbox.stateChanged[int].connect(self.draw_nodes)
        
        beam_checkbox = QCheckBox('Show Beams')
        ms_checkbox = QCheckBox('Show Master-Slaves Assignm.')
        chandof_checkbox = QCheckBox('Show Channel-DOF Assignm.')
        
        self.draw_button_group = QButtonGroup()
        self.draw_button_group.setExclusive(False)
        self.draw_button_group.addButton(beam_checkbox, 0)
        self.draw_button_group.addButton(ms_checkbox, 1)
        self.draw_button_group.addButton(chandof_checkbox, 2)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)
        
        view_layout.addWidget(self.axis_checkbox)
        view_layout.addWidget(self.grid_checkbox)
        view_layout.addWidget(beam_checkbox)
        view_layout.addWidget(ms_checkbox)
        view_layout.addWidget(chandof_checkbox)

        
        # controls for changing the axis' limits and viewport i.e. zoom and shift
        axis_limits_layout = QHBoxLayout()
        
        axis_limits_layout.addWidget(QLabel('Shift View:'))

        for shift in ['+X', '-X', '+Y', '-Y', '+Z', '-Z']:
            button = QToolButton()
            button.setText(shift)
            button.released.connect(self.shift_view)
            axis_limits_layout.addWidget(button)
        axis_limits_layout.addStretch()        
        
        axis_limits_layout.addWidget(QLabel('Viewport:'))

        for view in ['X', 'Y', 'Z', 'ISO']:
            button = QToolButton()
            button.setText(view)
            button.released.connect(self.change_viewport)
            axis_limits_layout.addWidget(button)
        axis_limits_layout.addStretch()      
            
        axis_limits_layout.addWidget(QLabel('Zoom:'))

        for zoom in ['+', '-','+X', '-X', '+Y', '-Y', '+Z', '-Z','eq.asp.']:
            button = QToolButton()
            button.setText(zoom)
            button.released.connect(self.zoom)
            axis_limits_layout.addWidget(button)
            
        # Buttons for creating/editing geometry and loading solutions
        grid_button = QPushButton('Edit Grid')
        grid_button.released.connect(self.stop_ani)
        grid_button.released.connect(self.geometry_creator.load_nodes)

        beam_button = QPushButton('Edit Beams')
        beam_button.released.connect(self.stop_ani)
        beam_button.released.connect(self.geometry_creator.load_lines)

        ms_button = QPushButton('Edit Master Slaves')
        ms_button.released.connect(self.stop_ani)
        ms_button.released.connect(self.geometry_creator.load_master_slave)

        cd_button = QPushButton('Edit Channel-DOFS-Assignment')
        cd_button.released.connect(self.stop_ani)
        cd_button.released.connect(self.geometry_creator.load_chan_dof)

        ssi_button = QPushButton('Load Modal Data')
        ssi_button.released.connect(self.stop_ani)
        ssi_button.released.connect(self.reload_ssi_solutions)
        
        # GUI controls for selecting modes and changing various 
        # values for drawing the modeshapes
        #self.order_combo = QComboBox()
        #self.order_combo.currentIndexChanged[str].connect(self.change_order)

        self.mode_combo = QComboBox()
        self.mode_combo.currentIndexChanged[str].connect(self.change_mode)

        self.amplitude_box = DelayedDoubleSpinBox()
        self.amplitude_box.setRange(0, 1000000000)
        self.amplitude_box.setValue(self.amplitude)
        self.amplitude_box.valueChangedDelayed.connect(self.change_amplitude)

        self.real_checkbox = QCheckBox('Magn.')
        self.real_checkbox.setCheckState(Qt.Checked)
        self.real_checkbox.stateChanged[int].connect(self.change_part)

        self.imag_checkbox = QCheckBox('Magn.+Phase')

        self.real_imag_group = QButtonGroup()
        self.real_imag_group.addButton(self.real_checkbox)
        self.real_imag_group.addButton(self.imag_checkbox)
        self.real_imag_group.setExclusive(True)

        plot_button = QPushButton('Draw')
        plot_button.released.connect(self.draw_msh)

        self.ani_button = QToolButton()
        self.ani_button.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPlay))
        self.ani_button.setToolTip("Play")
        self.ani_button.released.connect(self.animate)
        
        #textbox for showing information about the currently displayed mode
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        
        #put everything in layouts
        controls_layout = QGridLayout()
        controls_layout.addWidget(grid_button, 0, 0)
        controls_layout.addWidget(beam_button, 1, 0)
        controls_layout.addWidget(ms_button, 2, 0)
        controls_layout.addWidget(cd_button, 3, 0)
        controls_layout.addWidget(ssi_button, 4, 0)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)

        controls_layout.addWidget(sep, 0, 1, 5, 1)

        #controls_layout.addWidget(QLabel('Order'), 0, 2)
        #controls_layout.addWidget(self.order_combo, 0, 3)

        controls_layout.addWidget(QLabel('Mode'), 1, 2)
        controls_layout.addWidget(self.mode_combo, 1, 3)

        controls_layout.addWidget(QLabel('Amplitude'), 2, 2)
        controls_layout.addWidget(self.amplitude_box, 2, 3)

        layout = QHBoxLayout()
        controls_layout.addWidget(QLabel('Complex Part:'), 3, 2)
        layout.addWidget(self.real_checkbox)
        layout.addWidget(self.imag_checkbox)
        controls_layout.addLayout(layout, 3, 3)

        layout = QHBoxLayout()
        layout.addWidget(QLabel('Show Modeshape:'))
        layout.addWidget(plot_button)
        layout.addWidget(self.ani_button)
        controls_layout.addLayout(layout, 4, 2, 1, 2)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        controls_layout.addWidget(sep, 0, 4, 5, 1)

        controls_layout.addWidget(self.info_box, 0, 5, 5, 2)

        vbox = QVBoxLayout()
        
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        
        vbox.addWidget(self.canvas)
        vbox.setStretchFactor(self.canvas, 1)
        vbox.addWidget(sep1)    
        vbox.addLayout(view_layout)
        vbox.addLayout(axis_limits_layout)   
        vbox.addWidget(sep2)     
        vbox.addLayout(controls_layout)

        main_frame.setLayout(vbox)
        self.setCentralWidget(main_frame)
        
        self.show()
        
        #instantiate the x,y,z axis arrows
        self.draw_axis()

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

    @pyqtSlot()
    def reset_view(self):
        '''
        restore viewport, 
        restore axis' limits, 
        reset displacements values for all nodes, 
        clear the plot 
        and redraw the grid
        '''
        self.stop_ani()
        proj3d.persp_transformation = persp_transformation
        self.subplot.view_init(30, -60)
        self.subplot.autoscale_view()
        #self.xd = dict.fromkeys(self.xs.keys(), 0)
        #self.yd = dict.fromkeys(self.xs.keys(), 0)
        #self.zd = dict.fromkeys(self.xs.keys(), 0)
        self.xd = self.xs
        self.yd = self.xs
        self.zd = self.zs
        self.clear_plot()
        self.draw_nodes()

    @pyqtSlot()
    def shift_view(self, dir=None):
        '''
        shift the view along specified axis by +-20 % (hardcoded)
        works in combination with the appropriate buttons as senders
        or by passing one of  ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        '''
        if dir is None:
            dir = self.sender().text()
            
        if 'X' in dir:
            minx, maxx = self.subplot.get_xlim3d()
            deltax = (maxx - minx) / 5
            if '-' in dir:
                self.subplot.set_xlim3d(minx - deltax, maxx - deltax)
            elif '+' in dir:
                self.subplot.set_xlim3d(minx + deltax, maxx + deltax)
        elif 'Y' in dir:
            miny, maxy = self.subplot.get_ylim3d()
            deltay = (maxy - miny) / 5
            if '-' in dir:
                self.subplot.set_ylim3d(miny - deltay, maxy - deltay)
            elif '+' in dir:
                self.subplot.set_ylim3d(miny + deltay, maxy + deltay)
        elif 'Z' in dir:
            minz, maxz = self.subplot.get_zlim3d()
            deltaz = (maxz - minz) / 5
            if '-' in dir:
                self.subplot.set_zlim3d(minz - deltaz, maxz - deltaz)
            elif '+' in dir:
                self.subplot.set_zlim3d(minz + deltaz, maxz + deltaz)
        self.canvas.draw()

    @pyqtSlot()
    def zoom(self, dir=None):
        '''
        zoom the view by +-20 % (hardcoded)
        can zoom a single axis or all axis'
        works in combination with the appropriate buttons as senders or by 
        passing one of ['+', '-','+X', '-X', '+Y', '-Y', '+Z', '-Z','eq.asp.']
        '''
        if dir is None:
            dir = self.sender().text()

        minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()
        factor=1
        
        if '+' in dir:
            factor = 0.8
        elif '-' in dir:
            factor = 1.25

        if 'X' in dir:
            self.subplot.set_xlim3d(minx * factor, maxx * factor)
        elif 'Y' in dir:
            self.subplot.set_ylim3d(miny * factor, maxy * factor)
        elif 'Z' in dir:
            self.subplot.set_zlim3d(minz * factor, maxz * factor)
        elif dir == 'eq.asp.':
            midx=0.5*(minx+maxx)
            midy=0.5*(miny+maxy)
            midz=0.5*(minz+maxz)
            range=max((maxx-minx),(maxy-miny),(maxz-minz))
            minx,maxx=(midx-0.5*range,midx+0.5*range)
            miny,mayy=(midy-0.5*range,midy+0.5*range)
            minz,mazz=(midz-0.5*range,midz+0.5*range)
            self.subplot.set_xlim3d(minx * factor, maxx * factor)
            self.subplot.set_ylim3d(miny * factor, maxy * factor)
            self.subplot.set_zlim3d(minz * factor, maxz * factor)
        else:
            self.subplot.set_xlim3d(minx * factor, maxx * factor)
            self.subplot.set_ylim3d(miny * factor, maxy * factor)
            self.subplot.set_zlim3d(minz * factor, maxz * factor)

        self.subplot.get_proj()
        self.canvas.draw_idle()

    @pyqtSlot()
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
            
        if viewport == 'X':
            azim, elev = 0, 0
            proj3d.persp_transformation = orthogonal_proj
        elif viewport == 'Y':
            azim, elev = 270, 0
            proj3d.persp_transformation = orthogonal_proj
        elif viewport == 'Z':
            azim, elev = 0, 90
            proj3d.persp_transformation = orthogonal_proj
        elif viewport == 'ISO':
            azim, elev = -60, 30
            proj3d.persp_transformation = persp_transformation
        else:
            print('viewport not recognized: ', viewport)
            azim, elev = -60, 30
            proj3d.persp_transformation = persp_transformation
        self.subplot.view_init(elev, azim)
        self.canvas.draw()

    @pyqtSlot()
    def save_plot(self, path=None):
        '''
        save the curently displayed frame as a *.png graphics file
        '''
        
        file_choices = "PNG (*.png)|*.png"
        
        if path is None:
            path = str(QFileDialog.getSaveFileName(self,
                                                   'Save file', '',
                                                    file_choices))
        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)

###############################################################################
    '''
    @pyqtSlot(str)
    def change_order(self, order):
        ''' '''
        if the user selects a new order, update the mode selector 
        with the available modes and modal values for this order
        ''' '''
        self.order_num = int(float(order))
        modes = self.get_modal_properties(order)
        modes.sort(key=float)
        items = []
        current_index = 0
        for mode in modes:
            modal_values = self.get_modal_properties(
                self.order_num, mode + 1, 'Modal values')
            this_str = '{}: {} Hz'.format(
                mode + 1, modal_values['frequencies'])
            items.append(this_str)
            if mode + 1 == self.mode_num:
                current_index = mode + 1
        self.mode_combo.currentIndexChanged[str].disconnect(self.change_mode)
        self.mode_combo.clear()
        self.mode_combo.addItems(items)
        self.mode_combo.currentIndexChanged[str].connect(self.change_mode)
        for index in range(self.mode_combo.count()):
            if int(float(self.mode_combo.itemText(index).split(':')[0])) == self.mode_num:
                break
        else:
            index=0
        self.mode_combo.setCurrentIndex(index)
        '''

###############################################################################

    @pyqtSlot(str)
    def change_mode(self, mode):
        '''
        if user selects a new mode,
        extract the mode number from the passed string (contains frequency...)
        write modal values to the infobox
        and plot the mode shape
        '''
        
        #print('in change_mode: mode = ', mode)

        # mode numbering starts at 1 python lists start at 0
        self.mode_num = int(float(mode.split(':')[0])) - 1 
        
        self.frequencies, self.damping, self.mode_shape = \
            self.get_modal_properties(self.mode_num)
        #    self.order_num, self.mode_num + 1, 'Modal values')
        
        self.info_box.clear()
        self.info_box.append('Mode:\t{}'.format(self.mode_num + 1))
        #for key, unit in [('frequencies', 'Hz'), ('damping', '%'), ('eigenvalues', '')]:
        #    value = modal_values[key]
        #    self.info_box.append('{}:\t{} {}'.format(key, value, unit))
            
        self.draw_msh()

    @pyqtSlot()
    @pyqtSlot(float)
    def change_amplitude(self, amplitude=None):
        '''
        changes the amplitude
        amplitude either gets passed or will be read from the widget
        redraw the modeshapes based on this amplitude
        '''
        if amplitude is None:
            amplitude = self.amplitude_box.value()

        if amplitude == self.amplitude:
            return

        self.amplitude = amplitude
        self.draw_msh()

###############################################################################
#################### Das ist noch zu überarbeiten!!!  #########################
###############################################################################


    @pyqtSlot(bool)
    def change_part(self, b):
        '''
        change, which part of the complex number modeshapes should be 
        drawn, set the pointer variable based on which widget sent the signal
        redraw the modeshapes 
        '''
        part = self.sender().text()
        if part == 'Magn.' and b:
            self.complex_part = 'real'
        elif part == 'Magn.' and not b:
            self.complex_part = 'complex'
        elif part == 'Magn.+Phase' and b:
            self.complex_part = 'complex'
        elif part == 'Magn.+Phase' and not b:
            self.complex_part = 'real'
        self.draw_msh()

###############################################################################

    @pyqtSlot(int)
    def toggle_draw(self, i):
        '''
        helper function to receive the signal from the draw_button_group
        i is the number of the button that had it's state changed
        based on i and the checkstate the appropriate functions will be called
        '''
        checkstate = self.draw_button_group.button(i).checkState()

        if i == 0:  # show/unshow beams
            if checkstate == Qt.Checked:
                self.draw_lines()
            else:
                self.undraw_lines()
        elif i == 1:  # show/unshow master slave
            if checkstate == Qt.Checked:
                self.draw_master_slaves()
            else:
                self.undraw_master_slaves()
        elif i == 2:  # show/unshow chan dofs
            if checkstate == Qt.Checked:
                self.draw_chan_dofs()
            else:
                self.undraw_chan_dofs()
        else:
            print(i)

    @pyqtSlot(float, float, float, int)
    def add_node(self, x, y, z, i):
        '''
        receive a node from a signal
        add the coordinates to the internal node table
        add zero-value displacements for this node to the internal displacements table
        draw a single point at coordinates
        draw the number of the node
        store the two plot objects in a table
        remove any objects that might be in the table at the desired place
        i.e. avoid duplicate nodes
        '''
        self.grid_checkbox.stateChanged[int].disconnect(self.draw_nodes)
        self.grid_checkbox.setCheckState(Qt.Checked)
        self.grid_checkbox.stateChanged[int].connect(self.draw_nodes)

        self.xs[i], self.ys[i], self.zs[i] = x, y, z

        # leave present value if there is any else put 0
        self.xd[i] = self.xd.get(i, 0)
        self.yd[i] = self.yd.get(i, 0)
        self.zd[i] = self.zd.get(i, 0)

        x, y, z = x + self.xd[i], y + self.yd[i], z + \
            self.zd[i]  # draw displaced nodes

        patch = self.subplot.scatter(
            x, y, z, color=self.nodecolor, marker=self.nodemarker, s=self.nodesize)
        text = self.subplot.text(x, y, z, i)

        if self.patches_objects.get(i) is not None:
            if isinstance(self.patches_objects[i], (tuple, list)):
                for object in self.patches_objects[i]:
                    object.remove()
        self.patches_objects[i] = (patch, text)
        self.draw_axis()
        self.canvas.draw_idle()

    @pyqtSlot(tuple, int)
    def add_line(self, line, i):
        '''
        receive a line coordinates from a signal
        add the start node and end node to the internal line table
        draw a line between the tow nodes
        store the line object in a table
        remove any objects that might be in the table at the desired place
        i.e. avoid duplicate lines
        '''
        while len(self.lines) <= i:
            self.lines.append((0, 0))
        node_start, node_end = line
        self.lines[i] = line

        self.undraw_master_slaves()
        self.undraw_chan_dofs()

        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(0).setCheckState(Qt.Checked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        if self.complex_part == 'real':
            beamcolor = self.beamcolor
            beamstyle = self.beamstyle
        elif self.complex_part == 'complex':
            beamcolor = self.modecolor
            beamstyle = self.modestyle

        line_object = self.subplot.plot(
                        [self.xs[node] + self.xd[node] for node in line],
                        [self.ys[node] + self.yd[node] for node in line],
                        [self.zs[node] + self.zd[node] for node in line],
                        color=beamcolor, linestyle=beamstyle)[0]

        while len(self.lines_objects) < i + 1:
            self.lines_objects.append(None)
        if self.lines_objects[i] is not None:
            self.lines_objects[i].remove()
        self.lines_objects[i] = line_object

        self.canvas.draw_idle()

    @pyqtSlot(int, float, float, float, int, float, float, float, int)
    def add_master_slave(self, i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl, i):
        '''
        receive master-slave definitions from a signal
        add these definitions to the internal master-slave table
        draw an arrow indicating the DOF at each node of master and slave
            as a specialty arrows at equal positions and direction will 
            be offset to avoid overlapping
        arrow length's do not scale with the total dimensions of the plot
        store the two arrow objects in a table
        remove any objects that might be in the table at the desired place
        i.e. avoid duplicate arrows
        '''
        def offset_arrows(verts3d_new, all_arrows_list):
            '''
            avoid overlapping arrows as they are hard to distinguish
            therefore loop through all arrow object and compare their
            coordinates and directions (but ignore length) with the 
            arrow to be newly created if there is an overlapping then
            offset the coordinates of the new arrow by 5 % of the 
            length (hardcoded) in each direction (which should actually
            only be in the perpendicular plane)
            '''
            ((x_s, x_e), (y_s, y_e), (z_s, z_e)) = verts3d_new
            start_point = (x_s, y_s, z_s)
            length = x_e ** 2 + y_e ** 2 + z_e ** 2
            dir_norm = (x_e / length, y_e / length, z_e / length)
            while True:
                for arrow in itertools.chain.from_iterable(all_arrows_list):
                    (x_a, x_b), (y_a, y_b), (z_a, z_b) = arrow._verts3d
                    # transform from position vector to direction vector
                    x_c, y_c, z_c = (x_b - x_a), (y_b - y_a), (z_b - z_a)
                    this_start_point = (x_a, y_a, z_b)
                    this_length = x_c ** 2 + y_c ** 2 + z_c ** 2
                    if this_length == 0:
                        continue
                    this_dir_norm = (
                        x_c / this_length, y_c / this_length, z_c / this_length)
                    if start_point != this_start_point:  # starting point equal
                        continue
                    if this_dir_norm != dir_norm:  # direction equal
                        continue
                    # offset hardcoded
                    x_s, y_s, z_s = [
                        coord + 0.05 * this_length for coord in start_point]
                    # lazy offset, it should actually be in the plane
                    # perpendicular to the vector
                    start_point = (x_s, y_s, z_s)
                    length = x_e ** 2 + y_e ** 2 + z_e ** 2
                    dir_norm = (x_e / length, y_e / length, z_e / length)
                    break
                else:
                    break
            return ((x_s, x_e), (y_s, y_e), (z_s, z_e))

        while len(self.master_slaves) <= i:
            self.master_slaves.append((0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0))

        master_slave = (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl)
        self.master_slaves[i] = master_slave

        self.undraw_lines()  # do not show beams, as they are distracting
        self.undraw_chan_dofs()
        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(1).setCheckState(Qt.Checked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        color = "bgrcmyk"[int(fmod(i, 7))]  # equal colors for both arrows

        x_s, y_s, z_s = self.xs[i_m], self.ys[i_m], self.zs[i_m]
        ((x_s, x_m), (y_s, y_m), (z_s, z_m)) = offset_arrows(
            ((x_s, x_m), (y_s, y_m), (z_s, z_m)), self.arrows_objects)

        # point the arrow towards the resulting direction
        arrow_m = Arrow3D([x_s, x_s + x_m], [y_s, y_s + y_m], [z_s, z_s + z_m],
                          mutation_scale=5, lw=1, arrowstyle="-|>", color=color)
        arrow_m = self.subplot.add_artist(arrow_m)

        x_s, y_s, z_s = self.xs[i_sl], self.ys[i_sl], self.zs[i_sl]
        ((x_s, x_sl), (y_s, y_sl), (z_s, z_sl)) = offset_arrows(
            ((x_s, x_sl), (y_s, y_sl), (z_s, z_sl)), self.arrows_objects)

        # point the arrow towards the resulting direction
        arrow_sl = Arrow3D([x_s, x_s + x_sl], [y_s, y_s + y_sl], [z_s, z_s + z_sl],
                           mutation_scale=5, lw=1, arrowstyle="-|>", color=color)
        
        arrow_sl = self.subplot.add_artist(arrow_sl)

        while len(self.arrows_objects) < i + 1:
            self.arrows_objects.append(None)
        if self.arrows_objects[i] is not None:
            for obj in self.arrows_objects[i]:
                obj.remove()
        self.arrows_objects[i] = (arrow_m, arrow_sl)

        self.canvas.draw_idle()

    @pyqtSlot(int, int, tuple, int)
    def add_chan_dof(self, chan, node, dof, i):
        '''
        receive a channel - degree of freedom assignment from a signal
        add the values to the internal channel-dof table
        draw an arrow indicating the DOF 
        arrow lengths do not scale with the total dimension of the plot
        add the channel number to the arrow
        store the two objects in a table
        remove any objects that might be in the table at the desired place
        i.e. avoid duplicate arrows/texts
        '''
        while len(self.chan_dofs) <= i:
            self.chan_dofs.append((0, 0, (0.0, 0.0, 0.0)))
        self.chan_dofs[i] = (chan, node, dof)

        x_s, y_s, z_s = self.xs[node], self.ys[node], self.zs[node]
        x_m, y_m, z_m = dof[0], dof[1], dof[2]

        self.undraw_master_slaves()
        self.undraw_lines()
        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(2).setCheckState(Qt.Checked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        # point the arrow towards the resulting direction
        arrow = Arrow3D([x_s, x_s + x_m], [y_s, y_s + y_m],
                        [z_s, z_s + z_m], mutation_scale=5, lw=1, arrowstyle="-|>")
        arrow = self.subplot.add_artist(arrow)

        text = self.subplot.text(x_s + x_m, y_s + y_m, z_s + z_m, chan)

        while len(self.channels_objects) < i + 1:
            self.channels_objects.append(None)
        if self.channels_objects[i] is not None:
            for obj in self.channels_objects[i]:
                obj.remove()
        self.channels_objects[i] = (arrow, text)

        self.canvas.draw_idle()

    @pyqtSlot(float, float, float, int)
    def take_node(self, x, y, z, node):
        '''
        take a node at coordinates received by a signal
        take any objects connected to this node first (there should not be any)
        remove the patch objects from the plot
        remove the coordinates from the node and displacement tables
        '''
        # remove channel dof assignments for this node (should be covered by
        # GeometryCreator)
        while True:
            for chan_dof in self.chan_dofs:
                if chan_dof[1] == node:
                    index = self.chan_dofs.index(chan_dof)
                    self.take_chan_dof(
                        chan_dof[0], chan_dof[1], chan_dof[2], index)
                    break
            else:
                break  # executed when for loop runs through

        # remove connected lines (should be covered by GeometryCreator)
        while True:
            for line in self.lines:
                if line[0] == node or line[1] == node:
                    self.take_line(line, self.lines.index(line))
                    break
            else:
                break  # executed when for loop runs through

        # remove connected master-slaves (should be covered by GeometryCreator)
        while True:
            for master_slave in self.master_slaves:
                if master_slave[0] == node or master_slave[4] == node:
                    self.take_master_slave(*master_slave, 
                                    i=self.master_slaves.index(master_slave))
                    break
            else:
                break  # executed when for loop runs through

        d_x, d_y, d_z = abs(self.xd.get(node, 0)), \
                        abs(self.yd.get(node, 0)), \
                        abs(self.zd.get(node, 0))

        for j in [node] + list(range(max(len(self.patches_objects), node))):
            if self.patches_objects.get(j) is None:
                continue
            # ._offsets3d = ([x],[y],np.ndarray([z]))
            x_, y_, z_ = [float(val[0])
                          for val in self.patches_objects[j][0]._offsets3d]
            if   x - d_x <= x_ <= x + d_x and \
                    y - d_y <= y_ <= y + d_y and \
                    z - d_z <= z_ <= z + d_z:
                for object in self.patches_objects[j]:
                    object.remove()
                del self.patches_objects[j]
                break
        else:  # executed when for loop runs through
            if self.patches_objects:
                print('patches_object not found')

        for j in [node] + list(range(max(len(self.xs), node))):

            if  self.xs.get(j) == x and \
                    self.ys.get(j) == y and \
                    self.zs.get(j) == z:
                del self.xs[j]
                del self.ys[j]
                del self.zs[j]
                del self.xd[j]
                del self.yd[j]
                del self.zd[j]
                break
        else:  # executed when for loop runs through
            if self.patches_objects:
                print('node not found')

        self.canvas.draw_idle()

    @pyqtSlot(tuple)
    def take_line(self, line):
        '''
        remove a line between to node received by a signal
        if the plot objects are already in there displaced state
        the comparison between the actual coordinates and these
        objects have to account for this by comparing to an interval
        of coordinates
        remove the line nodes from the internal table, too
        '''
        assert isinstance(line, (tuple, list))
        assert len(line) == 2

        (x_s, x_e) = [self.xs[node] for node in line]
        (y_s, y_e) = [self.ys[node] for node in line]
        (z_s, z_e) = [self.zs[node] for node in line]

        (d_x_s, d_x_e) = [abs(self.xd.get(node, 0)) for node in line]
        (d_y_s, d_y_e) = [abs(self.yd.get(node, 0)) for node in line]
        (d_z_s, d_z_e) = [abs(self.zd.get(node, 0)) for node in line]

        for j in range(len(self.lines_objects)):
            (x_s_, x_e_), (y_s_, y_e_), (z_s_, z_e_) = self.lines_objects[
                j]._verts3d
            if  x_s - d_x_s <= x_s_ <= x_s + d_x_s and \
                    x_e - d_x_e <= x_e_ <= x_e + d_x_e and \
                    y_s - d_y_s <= y_s_ <= y_s + d_y_s and \
                    y_e - d_y_e <= y_e_ <= y_e + d_y_e and \
                    z_s - d_z_s <= z_s_ <= z_s + d_z_s and \
                    z_e - d_z_e <= z_e_ <= z_e + d_z_e:  # account for displaced lines

                self.lines_objects[j].remove()
                del self.lines_objects[j]
                break
            elif x_s - d_x_s <= x_e_ <= x_s + d_x_s and \
                    x_e - d_x_e <= x_s_ <= x_e + d_x_e and \
                    y_s - d_y_s <= y_e_ <= y_s + d_y_s and \
                    y_e - d_y_e <= y_s_ <= y_e + d_y_e and \
                    z_s - d_z_s <= z_e_ <= z_s + d_z_s and \
                    z_e - d_z_e <= z_s_ <= z_e + d_z_e:  # account for inverted lines

                self.lines_objects[j].remove()
                del self.lines_objects[j]
                break
        else:
            if self.lines_objects:
                print('line_object not found')

        for j in range(len(self.lines)):
            if self.lines[j] == line:
                del self.lines[j]
                break
        else:
            if self.lines:
                print('line not found')
        self.canvas.draw_idle()

    @pyqtSlot(int, float, float, float, int, float, float, float)
    def take_master_slave(self, i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl):
        '''
        remove the two arrows associated with the master-slave definition
        received by a signal
        remove the master-slave definition from the internal table, too
        '''
        arrow_m = (i_m, x_m, y_m, z_m)
        arrow_sl = (i_sl, x_sl, y_sl, z_sl)

        node_m = arrow_m[0]
        x_s_m, y_s_m, z_s_m = self.xs[node_m], self.ys[node_m], self.zs[node_m]
        x_e_m, y_e_m, z_e_m = arrow_m[1:4]
        length_m = x_e_m ** 2 + y_e_m ** 2 + z_e_m ** 2

        node_sl = arrow_sl[0]
        x_s_sl, y_s_sl, z_s_sl = self.xs[
            node_sl], self.ys[node_sl], self.zs[node_sl]
        x_e_sl, y_e_sl, z_e_sl = arrow_sl[1:4]
        length_sl = x_e_sl ** 2 + y_e_sl ** 2 + z_e_sl ** 2

        for j in range(len(self.arrows_objects)):
            arrow_found = [False, False]
            for arrow in self.arrows_objects[j]:
                (x_s, x_e), (y_s, y_e), (z_s, z_e) = arrow._verts3d
                # transform from position vector to direction vector
                x_e, y_e, z_e = (x_e - x_s), (y_e - y_s), (z_e - z_s)

                # check positions with offsets and directions
                if (x_s - 0.05 * length_m <= x_s_m <= x_s + 0.05 * length_m and
                    y_s - 0.05 * length_m <= y_s_m <= y_s + 0.05 * length_m and
                    z_s - 0.05 * length_m <= z_s_m <= z_s + 0.05 * length_m and
                    x_e == x_e_m and
                    y_e == y_e_m and
                    z_e == z_e_m):
                    arrow_found[0] = True

                if (x_s - 0.05 * length_sl <= x_s_sl <= x_s + 0.05 * length_sl and
                    y_s - 0.05 * length_sl <= y_s_sl <= y_s + 0.05 * length_sl and
                    z_s - 0.05 * length_sl <= z_s_sl <= z_s + 0.05 * length_sl and
                    x_e == x_e_sl and
                    y_e == y_e_sl and
                    z_e == z_e_sl):
                    arrow_found[1] = True

            # ie found the right master slave pair
            if arrow_found[0] and arrow_found[1]:
                # remove both master slave arrows
                for arrow in self.arrows_objects[j]:
                    arrow.remove()
                del self.arrows_objects[j]
                # restart the first for loop i.e. start j at 0 again
                break
            else:
                continue
        else:
            if self.arrows_objects:
                print('arrows_object not found')

        for j in range(len(self.master_slaves)):
            if self.master_slaves[j] == (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl):
                # could be stored in a tuple
                del self.master_slaves[j]
                break
            if self.master_slaves[j] == [i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl]:
                # or could be stored in a list
                del self.master_slaves[j]
                break
        else:
            if self.master_slaves:
                print('master slave not found')

        self.canvas.draw_idle()

    @pyqtSlot(int, int, tuple, int)
    def take_chan_dof(self, chan, node, dof):
        '''
        remove the arrow and text objects associated with the channel -
        DOF assignement received by a signal
        remove the channel - DOF assignement from the internal table, too
        '''
        assert isinstance(node, int)
        assert isinstance(chan, int)
        assert isinstance(dof, (tuple, list))
        assert len(dof) == 3

        x_s, y_s, z_s = self.xs[node], self.ys[node], self.zs[node]
        x_e, y_e, z_e = dof[0] + x_s, dof[1] + y_s, dof[2] + z_s

        for j in range(len(self.channels_objects)):
            (x_s_, x_e_), (y_s_, y_e_), (z_s_, z_e_) = \
                    self.channels_objects[j][0]._verts3d
            if nearly_equal(x_s_, x_s, 2) and nearly_equal(x_e_, x_e, 2) and \
               nearly_equal(y_s_, y_s, 2) and nearly_equal(y_e_, y_e, 2) and \
               nearly_equal(z_s_, z_s, 2) and nearly_equal(z_e_, z_e, 2):
                for obj in self.channels_objects[j]:
                    obj.remove()
                del self.channels_objects[j]
                break
        else:
            if self.channels_objects:
                print('chandof_object not found')

        for j in range(len(self.chan_dofs)):
            if self.chan_dofs[j][0] == chan and \
               self.chan_dofs[j][1] == node  and \
               nearly_equal(self.chan_dofs[j][2][0], dof[0], 3) and \
               nearly_equal(self.chan_dofs[j][2][1], dof[1], 3) and \
               nearly_equal(self.chan_dofs[j][2][2], dof[2], 3):
                del self.chan_dofs[j]
                break
        else:
            if self.chan_dofs:
                print('chandof not found')

        self.canvas.draw_idle()

    def undraw_nodes(self):
        '''
        remove all point and text objects belonging to the grid from the plot
        '''

        self.grid_checkbox.stateChanged[int].disconnect(self.draw_nodes)
        self.grid_checkbox.setCheckState(Qt.Unchecked)
        self.grid_checkbox.stateChanged[int].connect(self.draw_nodes)

        if self.patches_objects:
            for patch, text in self.patches_objects.values():
                patch.remove()
                text.remove()
            self.patches_objects.clear()
            self.canvas.draw_idle()

    def undraw_lines(self):
        '''
        remove all line objects from the plot
        '''

        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(0).setCheckState(Qt.Unchecked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        if self.lines_objects:
            for line in self.lines_objects:
                line.remove()
            self.lines_objects.clear()
            self.canvas.draw_idle()

    def undraw_master_slaves(self):
        '''
        remove all arrows belonging to the master-slave 
        definitions from the plot
        '''

        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(1).setCheckState(Qt.Unchecked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        if self.arrows_objects:
            for arrow_m, arrow_sl in self.arrows_objects:
                arrow_m.remove()
                arrow_sl.remove()
            self.arrows_objects.clear()
            self.canvas.draw_idle()

    def undraw_chan_dofs(self):
        '''
        remove all arrows and text objects belonging to the channel - 
        DOF assignments from the plot
        '''

        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(2).setCheckState(Qt.Unchecked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        if self.channels_objects:
            for arrow, text in self.channels_objects:
                arrow.remove()
                text.remove()
            self.channels_objects.clear()
            self.canvas.draw_idle()

    def clear_plot(self):
        '''
        remove everything from the plot, redraw the axis arrows
        '''
        self.undraw_nodes()
        self.undraw_lines()
        self.undraw_master_slaves()
        self.undraw_chan_dofs()
        self.axis_obj.clear()
        self.subplot.cla()
        #self.draw_axis()
        self.subplot.set_axis_off()

    def draw_axis(self):
        '''
        draw the axis arrows
        length is based on the current datalimits
        removes the current arrows if the exist
        '''
        
        for axis in ['X', 'Y', 'Z']:
            if axis in self.axis_obj:
                self.axis_obj[axis][0].remove()
                self.axis_obj[axis][1].remove()
                del self.axis_obj[axis]
                
        if self.axis_checkbox.checkState() == Qt.Unchecked: 
            self.canvas.draw_idle()
            return  
                
        if self.xs:
            axis = self.subplot.add_artist(
                    Arrow3D([0, max(self.xs.values()) / 2], [0, 0], [0, 0], 
                            mutation_scale=20, lw=3, arrowstyle="-|>", color="r"))
            text = self.subplot.text(
                     max(self.xs.values()) / 2, 0, 0, 'X', zdir='x', color='r')
            self.axis_obj['X'] = (axis, text)

        if self.ys:
            axis = self.subplot.add_artist(
                    Arrow3D([0, 0], [0, max(self.ys.values()) / 2], [0, 0], 
                            mutation_scale=20, lw=3, arrowstyle="-|>", color="g"))
            text = self.subplot.text(
                    0, max(self.ys.values()) / 2, 0, 'Y', zdir='y', color='g')
            self.axis_obj['Y'] = (axis, text)

        if self.zs:
            axis = self.subplot.add_artist(
                    Arrow3D([0, 0], [0, 0], [0, max(self.zs.values()) / 2], 
                            mutation_scale=20, lw=3, arrowstyle="-|>", color="b"))
            text = self.subplot.text(
                    0, 0, max(self.zs.values()) / 2, 'Z', zdir='z', color='b')
            self.axis_obj['Z'] = (axis, text)
        
        self.canvas.draw_idle()
        
    @pyqtSlot()
    @pyqtSlot(int)
    @pyqtSlot(dict, dict, dict)
    def draw_nodes(self, xd=None, yd=None, zd=None):
        ''''
        draw gridpoints from self.xs, self.ys, self.zs
        if displacement values are not passed the currently stored 
        displacement values are used for moving the nodes
        '''
        sender = self.sender()
        if isinstance(sender, QCheckBox):
            if self.grid_checkbox.checkState() == Qt.Unchecked:
                return self.undraw_nodes()
            elif isinstance(xd, (int, bool)):
                xd = None

        self.grid_checkbox.stateChanged[int].disconnect(self.draw_nodes)
        self.grid_checkbox.setCheckState(Qt.Checked)
        self.grid_checkbox.stateChanged[int].connect(self.draw_nodes)
        
        if xd is None:
            xd = self.xd
        else:
            assert isinstance(xd, dict)
        if yd is None:
            yd = self.yd
        else:
            assert isinstance(yd, dict)
        if zd is None:
            zd = self.zd
        else:
            assert isinstance(zd, dict)

        for key in self.xs.keys():
            x, y, z, i = self.xs[key] + xd[key], \
                         self.ys[key] + yd[key], \
                         self.zs[key] + zd[key], \
                         key
            patch = self.subplot.scatter(
                     x, y, z, color=self.nodecolor, 
                     marker=self.nodemarker, s=self.nodesize)
            text = self.subplot.text(x, y, z, i)

            if self.patches_objects.get(i) is not None:
                if isinstance(self.patches_objects[i], (tuple, list)):
                    for object in self.patches_objects[i]:
                        object.remove()
            self.patches_objects[i] = (patch, text)

        self.draw_axis()

        self.canvas.draw_idle()

    def draw_lines(self, xd=None, yd=None, zd=None, beamcolor=None, beamstyle=None):
        '''
        draw all the beams in self.lines
        self.lines=[line1, line2,....]
            line = [node_start, node_end]
            node numbering refers to elements in self.xs, self.ys, self.zs
        xd, yd, zd may be passed to draw custom deflections, else
        the currently stored displacement values are taken
        '''
        #print(xd,yd,zd)
        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(0).setCheckState(Qt.Checked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        self.undraw_master_slaves()
        self.undraw_chan_dofs()

        if xd is None:
            xd = self.xd
        else:
            assert isinstance(xd, dict)
        if yd is None:
            yd = self.yd
        else:
            assert isinstance(yd, dict)
        if zd is None:
            zd = self.zd
        else:
            assert isinstance(zd, dict)
        if beamcolor is None:
            if self.complex_part == 'real':
                beamcolor = self.beamcolor
            elif self.complex_part == 'complex':
                beamcolor = self.modecolor
        if beamstyle is None:
            if self.complex_part == 'real':
                beamstyle = self.beamstyle
            elif self.complex_part == 'complex':
                beamstyle = self.modestyle
        
        
        for i, line in enumerate(self.lines):
            line_object = self.subplot.plot([self.xs[node] + xd[node] for node in line],
                                            [self.ys[node] + yd[node] for node in line],
                                            [self.zs[node] + zd[node] for node in line], 
                                            color=beamcolor, linestyle=beamstyle)[0]

            while len(self.lines_objects) < i + 1:
                self.lines_objects.append(None)
            self.lines_objects[i] = line_object
        
        self.canvas.draw_idle()

    def draw_master_slaves(self):
        '''
        draw arrows for all master-slave definitions stored in the
        internal master-slave definition table
        '''
        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(1).setCheckState(Qt.Checked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        self.undraw_lines()
        self.undraw_chan_dofs()
        self.draw_nodes()

        for i, (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl) in enumerate(self.master_slaves):
            self.add_master_slave(i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl, i)
        self.canvas.draw_idle()

    def draw_chan_dofs(self):
        '''
        draw arrows and numbers for all channel-DOF assignemnts stored 
        in the internal channel - DOF assignment table
        '''
        self.draw_button_group.buttonClicked[int].disconnect(self.toggle_draw)
        self.draw_button_group.button(2).setCheckState(Qt.Checked)
        self.draw_button_group.buttonClicked[int].connect(self.toggle_draw)

        self.undraw_lines()
        self.undraw_master_slaves()
        self.draw_nodes()
        for i, (chan, node, dof) in enumerate(self.chan_dofs):
            self.add_chan_dof(chan, node, dof, i)
        self.canvas.draw_idle()




###############################################################################
########## Brauche ich die nächsten zwei Methoden überhaupt? ##################
###############################################################################

    @pyqtSlot()
    def reload_ssi_solutions(self):
        '''
        convenience method for changing the ssi_solutions_dict
        '''
        #self.ssi_solutions_path = QFileDialog.getOpenFileName(
        #    caption='Open SSI Solutions File', filter="*.slv")
        self.ssi_solutions_path = \
           QFileDialog.getExistingDirectory(caption='Open Directory with modal results', \
           directory=self.ssi_solutions_path, \
           options=QFileDialog.ShowDirsOnly)
   
        self.load_ssi_solutions()

    @pyqtSlot()
    def load_ssi_solutions(self, path=''):
        '''
        check prerequisits for drawing a modeshape
        i.e. grid, beam, master-slave definitions, channel-DOF assignments
        load the solutions file 
        read all the available order numbers and add them to the combo box
        restore the order number in the combo box, which will lead to a
        signal being emitted that will cause the mode selection box to be 
        updated as well
        '''

        if not self.xs or not self.ys or not self.zs:
            self.grid_requested.emit(self.grid_path, True)
        if not self.chan_dofs:
            self.chan_dofs_requested.emit(self.chan_dofs_path, True)
        if os.path.exists(path):
            self.ssi_solutions_path=path
        if self.ssi_solutions_path is None:
            #self.ssi_solutions_path = QFileDialog.getOpenFileName(
            #    caption='Open SSI Solutions File', filter="*.slv")
            self.ssi_solutions_path = \
               QFileDialog.getExistingDirectory(caption='Open Directory with modal results', \
               directory=self.ssi_solutions_path, \
               options=QFileDialog.ShowDirsOnly)

        self.frequency_file = self.ssi_solutions_path + '/frequencies.npy'
        self.damping_file = self.ssi_solutions_path + '/damping.npy'
        self.mode_shape_file = self.ssi_solutions_path + '/mode_shapes.npy'
        
        self.all_frequencies = np.load(self.frequency_file)
        self.all_damping = np.load(self.damping_file)
        self.all_mode_shapes = np.load(self.mode_shape_file)

        
        #print('mode_num_in get_modal_properties_1 = ', mode_num)
        items = []
        #current_index = 0
        for mode in range(len(self.all_frequencies)) :
            this_str = str(mode+1) + ': ' + str(self.all_frequencies[mode]) + ' Hz'
            items.append(this_str)

        self.mode_combo.currentIndexChanged[str].disconnect(self.change_mode)
        self.mode_combo.clear()
        self.mode_combo.addItems(items)
        self.mode_combo.currentIndexChanged[str].connect(self.change_mode)
        for index in range(self.mode_combo.count()):
            if int(float(self.mode_combo.itemText(index).split(':')[0])) == self.mode_num:
                break
            
        else:
            index=0
        
        self.mode_combo.setCurrentIndex(index)

###############################################################################
###################### Hier muss ich etwas verändern!!! #######################
###############################################################################

############### ORIGINALVERSION!!! ###########################

    '''
    def get_modal_properties(self, order_num=None, 
                             mode_num=None, property='Mode shapes'):
        ''' '''
        call with mode numbering starting at 1
        if order_num is None: returns all orders in the dict
        if mode_num is None: returns all mode nums for a given order
        property=['Mode shapes','Modal values', 'Complex conjugate']
        return the property:
        Mode shapes -> list of displacement values for each channel
        Modal values -> dict (frequencies: float, damping: float, eigenvalues: complex
        complex conjugate -> float/int
        ''' '''

        assert property in ['Mode shapes', 'Modal values', 'Complex conjugate']
        if mode_num is not None:
            mode_num -= 1  # mode given starts at 1 python listst start at 0

        if property in self.ssi_solutions_dict[self.ts]:
            property_dict = self.ssi_solutions_dict[self.ts][property]
            if order_num is None:
                # return the list of available order nums (integers)
                return [int(float(order.lstrip('order_'))) for order in property_dict]
            elif 'order_{}'.format(order_num) in property_dict:
                order_dict = property_dict['order_{}'.format(order_num)]
                if mode_num is None:
                    # list of all modes (integers) starts at 0 but modes start
                    # at 1 which should be noted always
                    return list(range(len(order_dict)))

                if property == 'Complex conjugate':
                    if len(order_dict[0]) >= mode_num:
                        return order_dict[0][mode_num]  # single float/int
                    else:
                        pprint.pprint(order_dict[0])
                        raise RuntimeError(
                            'Mode #{} is not in Complex conjugate of order {}.'
                            'List has been printed above.'.format(mode_num, order_num))
                elif property == 'Modal values':
                    modal_values = {}
                    for key, value in order_dict.items():
                        if len(value) >= mode_num:
                            modal_values[key] = value[mode_num]
                        else:
                            pprint.pprint(order_dict[key])
                            raise RuntimeError(
                                'Mode #{} is not in Modal values {} of order {}.'
                                'List has been printed above.'.format(
                                                    mode_num, key, order_num))
                    # dict {frequencies: float, damping: float, eigenvalues:
                    # complex}
                    return modal_values
                else:
                    if len(order_dict) >= mode_num:
                        return order_dict[mode_num]  # list of node deflections
                    else:
                        pprint.pprint(order_dict)
                        raise RuntimeError(
                            'Mode #{} is not in Mode shapes of order {}. '
                            'List has been printed above.'.format(
                                                    mode_num, order_num))
            else:
                print(list(property_dict.keys()))
                raise RuntimeError(
                    'Modal value {} of order {} is not present in the '
                    'ssi_solutions_dict'.format(property, order_num))
        else:
            print(self.ssi_solutions_dict[self.ts].keys())
            raise RuntimeError(
                'There are no {} in the solutions dictionary.'.format(property))
        '''
###############################################################################

############### VERÄNDERTE VERSION!!! ###########################


    def get_modal_properties(self, mode_num=None):
        '''
        call with mode numbering starting at 1
        if mode_num is None: returns all mode nums
        Mode shapes -> 2D numpy array from mode_shape_file -> columns are components
        of one mode shape, values are complex, node assignment needs to be done
        Natural frequencies [Hz] -> 1D numpy array from frequency_file -> each entry 
        corresponds to one column in mode shapes
        Damping coefficients [%] -> 1D numpy array from damping_file -> each entry 
        corresponds to one column in mode shapes
        '''


        #print('mode_num_in get_modal_properties_2 = ', mode_num)
        
        if mode_num is not None:
            
            if mode_num < 0:
                raise RuntimeError(
                    'The mode number must be greater than 0.')                
            
            if mode_num > len(self.all_frequencies):
                raise RuntimeError(
                    'The mode number must be greater than max. number of identified modes')                
            
            else:
                #mode_num -= 1  # mode given starts at 1 python listst start at 0
                
                self.frequencies = self.all_frequencies[mode_num]
                self.damping = self.all_damping[mode_num]
                self.mode_shapes = self.all_mode_shapes[:,mode_num]

        if mode_num is None:
            # list of all modes (integers) starts at 0 but modes start
            # at 1 which should be noted always
        
            self.frequencies = self.all_frequencies
            self.damping = self.all_damping
            self.mode_shapes = self.all_mode_shapes
        
        #print('mode_num_in get_modal_properties_3 = ', mode_num)
        
        #print('frequencies: ', self.frequencies)

        return (self.frequencies, self.damping, self.mode_shapes)

###############################################################################

############### ORIGINALVERSION!!! ###########################
    '''
    def draw_msh(self, order_num=None, mode_num=None, ampli=None, complex_part=None):
        ''' '''
        call with mode numbering starting at 1
        extracts the modeshapes for given order and mode from the
        ssi_solutions_dict and assigns displacement values to the
        nodes based on the channel - DOF assignments and the master - 
        slave definitions
        finally draws the displaced nodes and beams
        if any of the arguments are omited the default value as it 
        is internally stored will be taken (which will be mostly the case)
        ''' '''
        if order_num is None:
            order_num = self.order_num
        if mode_num is None:
            mode_num = self.mode_num
        else:
            mode_num -= 1  # mode given starts at 1 python lists start at 0
        if ampli is None:
            ampli = self.amplitude
        if complex_part is None:
            complex_part = self.complex_part
        else:
            assert complex_part in ('real', 'imaginary')

        mode_shape = self.get_modal_properties(
            order_num, mode_num, 'Mode shapes')

        if len(mode_shape) < len(self.chan_dofs):
            QMessageBox.information(
                self, 'Information', 'Not all channels ({})are assigned'
                'to a node and dof!'.format(len(mode_shape)))

        self.xd = dict.fromkeys(self.xs.keys(), 0)
        self.yd = dict.fromkeys(self.ys.keys(), 0)
        self.zd = dict.fromkeys(self.zs.keys(), 0)

        if isinstance(mode_shape[0], numpy.complex):
            if complex_part == 'real':
                norm = max([abs(disp.real) for disp in mode_shape])
            elif complex_part == 'imaginary':
                norm = max([abs(disp.imag) for disp in mode_shape])
        else:
            norm = max([abs(disp) for disp in mode_shape])

        for chan, disp in enumerate(mode_shape):
            if isinstance(disp, numpy.complex):
                if complex_part == 'real':
                    disp = disp.real
                elif complex_part == 'imaginary':
                    disp = disp.imag

            for chan_, node, dof in self.chan_dofs:
                if chan_ == chan:
                    break
            else:
                print('Could not find channel - DOF assignment for '
                      'channel {}!'.format(chan))
                continue
            self.xd[node] = dof[0] * disp * ampli / norm
            self.yd[node] = dof[1] * disp * ampli / norm
            self.zd[node] = dof[2] * disp * ampli / norm

        for i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl in self.master_slaves:
            master_disp =   self.xd[i_m] * x_m + \
                            self.yd[i_m] * y_m + \
                            self.zd[i_m] * z_m

            self.xd[i_sl] += master_disp * x_sl
            self.yd[i_sl] += master_disp * y_sl
            self.zd[i_sl] += master_disp * z_sl

        if not self.animated:
            self.undraw_lines()
            if self.grid_checkbox.checkState() == Qt.Checked:
                self.draw_nodes(self.xd, self.yd, self.zd)
            self.draw_lines(self.xd, self.yd, self.zd)
    '''

############### VERÄNDERTE VERSION!!! ###########################

    def draw_msh(self, mode_num=None, ampli=None, complex_part=None):
        '''
        call the modeshape with mode numbering starting at 1
        and assigns displacement values to the
        nodes based on the channel - DOF assignments and the master - 
        slave definitions
        finally draws the displaced nodes and beams
        if any of the arguments are omited the default value as it 
        is internally stored will be taken (which will be mostly the case)
        '''
        #print('mode_num_1 = ', mode_num)
        #print('self.mode_num = ', self.mode_num)
        if mode_num is None:
            mode_num = self.mode_num
            #print('mode_num_2 = ', mode_num)

        else:
            mode_num -= 1  # mode given starts at 1 python lists start at 0
        #print('mode_num_3 = ', mode_num)

        if ampli is None:
            ampli = self.amplitude

        if complex_part is None:
            complex_part = self.complex_part
        else:
            assert complex_part in ('real', 'complex')

        nat_freq, zeta, mode_shape = self.get_modal_properties(mode_num)
        
        if len(mode_shape) < len(self.chan_dofs):
            QMessageBox.information(
                self, 'Information', 'Not all channels ({}) are assigned'
                'to a node and dof!'.format(len(mode_shape)))

        self.xd = dict.fromkeys(self.xs.keys(), 0)
        self.yd = dict.fromkeys(self.ys.keys(), 0)
        self.zd = dict.fromkeys(self.zs.keys(), 0)
        
        self.xphi = dict.fromkeys(self.xs.keys(), 0)
        self.yphi = dict.fromkeys(self.ys.keys(), 0)
        self.zphi = dict.fromkeys(self.zs.keys(), 0)
        #### Normierung für reelle / komplexe Eigenformen sind noch zu machen!!!        
        '''
        if isinstance(mode_shape[0], np.complex):
            if complex_part == 'real':
                norm = max(abs(mode_shape)
            elif complex_part == 'complex':
                norm = max([abs(disp.imag) for disp in mode_shape])
        else:
            norm = max([abs(disp) for disp in mode_shape])
        '''

        for chan, disp in enumerate(mode_shape):
            if isinstance(disp, np.complex):
                if complex_part == 'real':
                    phase = np.angle(disp, True) 
                    disp = np.abs(disp)
                    if phase < 0 : 
                        phase += 180
                        disp = -disp                    
                    if phase > 90 and phase < 270:
                        disp = - disp
                    phase = 0
                else:
                    phase = np.angle(disp)
                    disp = np.abs(disp)#*np.cos(phase)
            else:
                phase = 0    
            for chan_, node, dof in self.chan_dofs:
                if chan_ == chan:
                    break
            else:
                print('Could not find channel - DOF assignment for '
                      'channel {}!'.format(chan))
                continue
                
            # assumes vectors have components in one direction (x,y,z) only
            # to convert, run: SSI_cov_ref_.compute_common_components
            self.xd[node] += dof[0] * disp * ampli #/ norm
            self.yd[node] += dof[1] * disp * ampli #/ norm
            self.zd[node] += dof[2] * disp * ampli #/ norm
            
            self.xphi[node] += dof[0]*phase
            self.yphi[node] += dof[1]*phase
            self.zphi[node] += dof[2]*phase
            
        for i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl in self.master_slaves:
            master_disp =   self.xd[i_m] * x_m + \
                            self.yd[i_m] * y_m + \
                            self.zd[i_m] * z_m

            self.xd[i_sl] += master_disp * x_sl
            self.yd[i_sl] += master_disp * y_sl
            self.zd[i_sl] += master_disp * z_sl

        if not self.animated:
            #self.clear_plot()
            self.undraw_lines()
            
            if self.grid_checkbox.checkState() == Qt.Checked:
                self.draw_nodes(self.xd, self.yd, self.zd)
            #self.subplot.cla()
            self.undraw_lines()
            self.draw_lines(self.xd, self.yd, self.zd)
        self.subplot.set_axis_off()
        self.canvas.draw()
###############################################################################


    @pyqtSlot()
    def stop_ani(self):
        '''
        convenience method to stop the animation and restore the still plot
        '''
        if self.animated:
            self.line_ani._stop()
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
            self.animated = False

    @pyqtSlot()
    def animate(self):
        '''
        create necessary objects to animate the currently displayed
        deformed structure
        '''
        def init_lines():
            
            #self.clear_plot()
            
            beamcolor = 'w'#self.beamcolor
            beamstyle = self.beamstyle
            
            self.lines_objects = [self.subplot.plot(
                                [self.xs[node] for node in line],
                                [self.ys[node] for node in line],
                                [self.zs[node] for node in line], 
                                color=beamcolor, linestyle=beamstyle)[0] \
                              for line in self.lines]
            return self.lines_objects
        
        def update_lines(num):
            '''
            subfunction to calculate displacements based 
            '''

            for line, line_node in zip(self.lines_objects, self.lines):
                x = [self.xs[node] + self.xd[node]
                     * np.cos(num / 25 * 2 * np.pi  + self.xphi[node]) 
                     for node in line_node]
                y = [self.ys[node] + self.yd[node]
                     * np.cos(num / 25 * 2 * np.pi  + self.yphi[node]) 
                     for node in line_node]
                z = [self.zs[node] + self.zd[node]
                     * np.cos(num / 25 * 2 * np.pi  + self.zphi[node]) 
                     for node in line_node]
                # NOTE: there is no .set_data() for 3 dim data...
                line.set_data([x, y])
                line.set_color('b')
                line.set_3d_properties(z)
            return self.lines_objects

        self.clear_plot()
        if self.animated:
            self.line_ani._stop()
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
            self.animated = False
            return self.draw_msh()
        else:
            self.ani_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
            self.animated = True

        if self.complex_part == 'real':
            beamcolor = self.beamcolor
            beamstyle = self.beamstyle
        elif self.complex_part == 'complex':
            beamcolor = self.modecolor
            beamstyle = self.modestyle

#         self.lines_objects = [self.subplot.plot(
#                                 [self.xs[node] for node in line],
#                                 [self.ys[node] for node in line],
#                                 [self.zs[node] for node in line], 
#                                 color=beamcolor, linestyle=beamstyle)[0] \
#                               for line in self.lines]
        
        self.line_ani = FuncAnimation(fig=self.fig,
                                      func=update_lines,
                                      init_func=init_lines,
                                      interval=50,
                                      save_count=0,
                                      blit=True)

        self.canvas.draw()


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


    @pyqtSlot()
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


class Arrow3D(FancyArrowPatch):
    '''
    credit goes to (don't know the original author):
    http://pastebin.com/dWvFxb1Q
    draw an arrow in 3D space
    '''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        '''
        inherit from FancyArrowPatch
        and set self._verts3d class variable
        '''
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        '''
        get the projection from the 3D point to 2D point to draw the arrow
        '''
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class GeometryCreator(QMainWindow):
    '''
    GUI for creating/editing/loading/saving node, lines, 
    master-slave definitions and channel-DOF assignments.
    several signals automatically emit any changes made with this GUI,
    such that they can be processed in a class that plots these things.
    
    Files are assumed to be in a matlab compatible, csv readable ascii 
    format, with whitespace delimiters:
    possible node definitions (scientific notation):
      1.0000000e+000 -2.4000000e+000 -2.4000000e+000  0.0000000e+000
      2.0000000e+000 -2.4000000e+000  2.4000000e+000  0.0000000e+000
      ....
    possible channel-DOF assignments (float/int notation):
     1  0  9  0.000  0.000  1.000
     2  1  1  0.000  0.000  1.000
     ...
    others are similar.
    '''
    # define pyqtSignals that can be connected to
    node_added = pyqtSignal(float, float, float, int)
    node_taken = pyqtSignal(float, float, float, int)

    line_added = pyqtSignal(tuple, int)
    line_taken = pyqtSignal(tuple)

    slave_added = pyqtSignal(
        int, float, float, float, int, float, float, float, int)
    slave_taken = pyqtSignal(
        int, float, float, float, int, float, float, float)

    chan_dof_added = pyqtSignal(int, int, tuple, int)
    chan_dof_taken = pyqtSignal(int, int, tuple)

    return_to_func = pyqtSignal()

    def __init__(self):
        '''
        initiate the class,
        inherit from QMainWindow
        create class variables
        '''
        super(GeometryCreator, self).__init__()

        # not sure if it is necessary to hold a copy of these variables in this class's memory
        # these variables should always be equal to the ones in PlotMSH object
        
        # node tables are dicts to make node numbering with gaps possible
        # else we would have to store node numbers in a separate list which
        # has to be kept in sync with the coordinates tables
        self.xs, self.ys, self.zs = {}, {}, {}
        self.lines = []
        self.master_slaves = []
        self.chan_dofs = []

        self.last_path = os.getcwd()

    def show_creation_widget(self, type_='nodes'):
        '''
        draw a widget containing:
        several inputboxes (QSpinBox, QDoubleSpinBox) for defining 
        the values for nodes, lines, master slaves and channel dof assignments
        '''
        'types =  [nodes, lines master_slaves, chan_dofs]'

        main_frame = QWidget()

        inp_box = QGridLayout()
        self.add_button = QPushButton('Add')

        if type_ == 'nodes':
            self.setWindowTitle('Edit Nodes')
            
            self.x_input = QDoubleSpinBox()
            self.x_input.setRange(-1000, 1000)
            self.y_input = QDoubleSpinBox()
            self.y_input.setRange(-1000, 1000)
            self.z_input = QDoubleSpinBox()
            self.z_input.setRange(-1000, 1000)

            inp_box.addWidget(QLabel('x'), 0, 0)
            inp_box.addWidget(self.x_input, 1, 0)
            inp_box.addWidget(QLabel('y'), 0, 1)
            inp_box.addWidget(self.y_input, 1, 1)
            inp_box.addWidget(QLabel('z'), 0, 2)
            inp_box.addWidget(self.z_input, 1, 2)

            self.add_button.released.connect(self.add_node)

            self.coordinates_list = QTableWidget(0, 4)

            load_button = QPushButton('Load Gridfile')
            load_button.released.connect(self.load_new_nodes)
            
        elif type_ == 'lines':
            self.setWindowTitle('Edit Lines')

            max_n = max(self.xs.keys()) # do not select nonexisting nodes
            self.start_input = QSpinBox()
            self.start_input.setRange(0, max_n)
            self.end_input = QSpinBox()
            self.end_input.setRange(0, max_n)

            inp_box.addWidget(QLabel('start node'), 0, 0)
            inp_box.addWidget(self.start_input, 1, 0)
            inp_box.addWidget(QLabel('end node'), 0, 1)
            inp_box.addWidget(self.end_input, 1, 1)

            self.add_button.released.connect(self.add_line)

            self.coordinates_list = QTableWidget(0, 2)

            load_button = QPushButton('Load Beamfile')
            load_button.released.connect(self.load_new_lines)
            
        elif type_ == 'master_slaves':
            self.setWindowTitle('Edit Master-Slave Definitions')

            max_n = max(self.xs.keys()) # do not select nonexisting nodes
            self.master_node = QSpinBox()
            self.master_node.setRange(0, max_n)
            self.master_x = QDoubleSpinBox()
            self.master_x.setRange(-100, 100)
            self.master_y = QDoubleSpinBox()
            self.master_y.setRange(-100, 100)
            self.master_z = QDoubleSpinBox()
            self.master_z.setRange(-100, 100)
            self.slave_node = QSpinBox()
            self.slave_node.setRange(0, max_n)
            self.slave_x = QDoubleSpinBox()
            self.slave_x.setRange(-100, 100)
            self.slave_y = QDoubleSpinBox()
            self.slave_y.setRange(-100, 100)
            self.slave_z = QDoubleSpinBox()
            self.slave_z.setRange(-100, 100)

            inp_box.addWidget(QLabel('master node'), 0, 0)
            inp_box.addWidget(self.master_node, 1, 0)
            inp_box.addWidget(QLabel('master x'), 0, 1)
            inp_box.addWidget(self.master_x, 1, 1)
            inp_box.addWidget(QLabel('master y'), 0, 2)
            inp_box.addWidget(self.master_y, 1, 2)
            inp_box.addWidget(QLabel('master z'), 0, 3)
            inp_box.addWidget(self.master_z, 1, 3)
            inp_box.addWidget(QLabel('slave node'), 0, 4)
            inp_box.addWidget(self.slave_node, 1, 4)
            inp_box.addWidget(QLabel('slave x'), 0, 5)
            inp_box.addWidget(self.slave_x, 1, 5)
            inp_box.addWidget(QLabel('slave y'), 0, 6)
            inp_box.addWidget(self.slave_y, 1, 6)
            inp_box.addWidget(QLabel('slave z'), 0, 7)
            inp_box.addWidget(self.slave_z, 1, 7)

            self.add_button.released.connect(self.add_master_slave)

            self.coordinates_list = QTableWidget(0, 8)

            load_button = QPushButton('Load Master-Slave-File')
            load_button.released.connect(self.load_new_master_slave)

        elif type_ == 'chan_dofs':
            self.setWindowTitle('Edit Channel-DOF Assignmnents')

            max_n = max(self.xs.keys())
            self.chan_input = QSpinBox()
            self.chan_input.setRange(0, 1000)
            self.node_input = QSpinBox()
            self.node_input.setRange(0, max_n)
            self.x_input = QDoubleSpinBox()
            self.x_input.setRange(-1000, 1000)
            self.y_input = QDoubleSpinBox()
            self.y_input.setRange(-1000, 1000)
            self.z_input = QDoubleSpinBox()
            self.z_input.setRange(-1000, 1000)

            inp_box.addWidget(QLabel('channel'), 0, 0)
            inp_box.addWidget(self.chan_input, 1, 0)
            inp_box.addWidget(QLabel('node'), 0, 1)
            inp_box.addWidget(self.node_input, 1, 1)
            inp_box.addWidget(QLabel('x ampl.'), 0, 2)
            inp_box.addWidget(self.x_input, 1, 2)
            inp_box.addWidget(QLabel('y ampl.'), 0, 3)
            inp_box.addWidget(self.y_input, 1, 3)
            inp_box.addWidget(QLabel('z ampl.'), 0, 4)
            inp_box.addWidget(self.z_input, 1, 4)

            self.add_button.released.connect(self.add_chan_dof)

            self.coordinates_list = QTableWidget(0, 5)

            load_button = QPushButton('Load Channel DOF File')
            load_button.released.connect(self.load_new_chan_dof)

        remove_button = QPushButton('remove')
        remove_button.released.connect(self.remove_rows)

        button_box_top = QHBoxLayout()
        button_box_top.addWidget(remove_button)
        button_box_top.addStretch()
        button_box_top.addWidget(self.add_button)

        self.coordinates_list.horizontalHeader().hide()
        self.coordinates_list.verticalHeader().hide()
        self.coordinates_list.setShowGrid(False)
        for column in range(self.coordinates_list.columnCount()):
            self.coordinates_list.setColumnWidth(column, 50)
        self.coordinates_list.setSelectionBehavior(
            QAbstractItemView.SelectRows)

        clear_button = QPushButton('Clear')
        clear_button.released.connect(self.clear_list)

        save_button = QPushButton('Save')
        save_button.released.connect(self.save_file)

        close_button = QPushButton('Close')
        close_button.released.connect(self.close_return)

        button_box_bottom = QHBoxLayout()
        button_box_bottom.addWidget(clear_button)
        button_box_bottom.addWidget(load_button)
        button_box_bottom.addWidget(save_button)
        button_box_bottom.addWidget(close_button)

        vbox = QVBoxLayout()
        vbox.addLayout(inp_box)
        vbox.addLayout(button_box_top)
        vbox.addWidget(self.coordinates_list, stretch=1)
        vbox.addLayout(button_box_bottom)

        main_frame.setLayout(vbox)

        self.setCentralWidget(main_frame)
        self.show()
        
    def keyPressEvent(self, e):        
        "define which signals will be emitted if a specified key is pressed"
        
        #inherit the original method from QMainWindows to not break keyboard navigation
        super(GeometryCreator, self).keyPressEvent(e)
        if e.key() == Qt.Key_Enter: 
            self.add_button.released.emit()
               
    @pyqtSlot()
    @pyqtSlot(int, int, int)
    def add_node(self, x=None, y=None, z=None, node=None):
        '''
        add a node either programatically by passing the corrdinates
        or via a signal from a button, such that the coordinates will be
        read from the input widgets
        coordinates will be added to the list widget, to the internal
        node table and sent via a signal
        x, y, z = float
        node = int
        '''
        if x is None:
            x = self.x_input.value()
        else:
            x = float(x)
        if y is None:
            y = self.y_input.value()
        else:
            y = float(y)
        if z is None:
            z = self.z_input.value()
        else:
            z = float(z)

        rows = self.coordinates_list.rowCount()
        self.coordinates_list.insertRow(rows)

        if node is None:
            node = int(rows)
        else:
            node = int(float(node))

        for col, val in enumerate([node, x, y, z]):
            if isinstance(val, int):
                item = QTableWidgetItem('{:d}'.format(val))
            else:
                item = QTableWidgetItem('{:2.3f}'.format(val))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.coordinates_list.setItem(rows, col, item)
        self.coordinates_list.resizeRowsToContents()

        self.xs[node], self.ys[node], self.zs[node] = x, y, z

        self.node_added.emit(x, y, z, node)
        
    @pyqtSlot()
    @pyqtSlot(tuple)
    def add_line(self, line=None):
        '''
        add a line either programatically by passing the line nodes
        or via a signal from a button, such that the coordinates will be
        read from the input widgets
        line nodes (start node and end node; must be existing)
        will be added to the list widget, to the internal lines table
        and sent via a signal
        line = (start_node [int], end_node [int])
        '''
        if line is None:
            n_start = self.start_input.value()
            n_end = self.end_input.value()
            line = (n_start, n_end)

        assert isinstance(line, (tuple, list))
        assert len(line) == 2
        for value in line:
            assert isinstance(value, int)

        n_start, n_end = line

        assert len(self.xs) == len(self.ys) == len(self.zs)
        assert max(list(self.xs.keys())) >= n_start
        assert max(list(self.xs.keys())) >= n_end

        rows = self.coordinates_list.rowCount()
        self.coordinates_list.insertRow(rows)

        for col, val in enumerate(line):
            item = QTableWidgetItem('{:d}'.format(val))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.coordinates_list.setItem(rows, col, item)
        self.coordinates_list.resizeRowsToContents()

        while len(self.lines) <= rows:
            self.lines.append((0, 0))
        self.lines[rows] = line
        self.line_added.emit(line, rows)

    @pyqtSlot()
    @pyqtSlot(int, float, float, float, int, float, float, float)
    def add_master_slave(self, i_m=None,  x_m=None,  y_m=None,  z_m=None, 
                               i_sl=None, x_sl=None, y_sl=None, z_sl=None):
        '''
        add a master slave definition either programatically by passing
        the nodes and directional factors or via a signal from a button,
        such that these values will be read from the input widgets
        master-slave definitions will be added to the list widget, to 
        the internal master-slave table and sent via a signal
        i_m, i_sl = node_num [int]
        x_m, y_m, z_m, x_sl, y_sl, z_sl = float
        '''
        if i_m is None:
            i_m = self.master_node.value()
        i_m = int(float(i_m))
        if x_m is None:
            x_m = self.master_x.value()
        x_m = float(x_m)
        if y_m is None:
            y_m = self.master_y.value()
        y_m = float(y_m)
        if z_m is None:
            z_m = self.master_z.value()
        z_m = float(z_m)
        if i_sl is None:
            i_sl = self.slave_node.value()
        i_sl = int(float(i_sl))
        if x_sl is None:
            x_sl = self.slave_x.value()
        x_sl = float(x_sl)
        if y_sl is None:
            y_sl = self.slave_y.value()
        y_sl = float(y_sl)
        if z_sl is None:
            z_sl = self.slave_z.value()
        z_sl = float(z_sl)

        if x_m == y_m == z_m == 0 or x_sl == y_sl == z_sl == 0:
            QMessageBox.warning(
                self, 'Warning', 'You have to select at least one direction' 
                'for each of master and slave to be non-zero! Will omit now!')
            return  # arrows of zero length cause error messages

        rows = self.coordinates_list.rowCount()
        self.coordinates_list.insertRow(rows)

        for col, val in enumerate([i_m,  x_m,  y_m,  z_m, 
                                   i_sl, x_sl, y_sl, z_sl]):
            if isinstance(val, int): # e.g. node
                item = QTableWidgetItem('{:d}'.format(val))
            else: # e.g. DOF
                item = QTableWidgetItem('{:2.3f}'.format(val))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.coordinates_list.setItem(rows, col, item)
        self.coordinates_list.resizeRowsToContents()

        while len(self.master_slaves) <= rows:
            self.master_slaves.append((0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0))
        self.master_slaves[rows] = [i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl]

        self.slave_added.emit(i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl, rows)
 
    @pyqtSlot()
    @pyqtSlot(int, tuple)
    def add_chan_dof(self, chan=None, node=None, dof=None):
        '''
        add a channel - DOF assignment either programatically by passing
        the channel, node and direction information or via a signal from
        a button, such that these values will be read from the input widgets
        values will be added to the list widget, to the internal 
        channel-dof table and sent via a signal
        chan = int
        node = int
        dof = (x_ampli [float], y_ampli [float], z_ampli [float])
        '''
        if chan is None:
            chan = self.chan_input.value()
            self.chan_input.setValue(chan + 1)
        else:
            chan = int(float(chan))
        if node is None:
            node = self.node_input.value()
        else:
            node = int(float(node))
        if dof is None:
            x_amp = self.x_input.value()
            y_amp = self.y_input.value()
            z_amp = self.z_input.value()
            dof = (x_amp, y_amp, z_amp)
        else:
            assert isinstance(dof, (tuple, list))
            assert len(dof) == 3
            for amp in dof:
                assert isinstance(amp, (int, float))

        if dof[0] == dof[1] == dof[2] == 0:
            QMessageBox.warning(
                self, 'Warning', 'You have to select at least one direction'
                'for amplification to be non-zero! Will omit now!')
            return  # arrows of zero length cause error messages

        rows = self.coordinates_list.rowCount()
        self.coordinates_list.insertRow(rows)

        for col, val in enumerate([chan, node] + list(dof)):
            if isinstance(val, int):
                item = QTableWidgetItem('{:d}'.format(val))
            else:
                item = QTableWidgetItem('{:2.3f}'.format(val))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.coordinates_list.setItem(rows, col, item)
        self.coordinates_list.resizeRowsToContents()

        while len(self.chan_dofs) <= rows:
            self.chan_dofs.append((0, 0, (0.0, 0.0, 0.0)))
        self.chan_dofs[rows] = (chan, node, dof)
        
        self.chan_dof_added.emit(chan, node, dof, rows)

    @pyqtSlot()
    def load_new_nodes(self):
        self.clear_list()
        self.load_nodes()
        
    @pyqtSlot()
    def load_new_lines(self):
        self.clear_list()
        self.load_lines()
        
    @pyqtSlot()    
    def load_new_master_slave(self):
        self.clear_list()
        self.load_master_slave()
 
    @pyqtSlot()
    def load_new_chan_dof(self):
        self.clear_list()
        self.load_chan_dof()

    @pyqtSlot()
    @pyqtSlot(str)
    @pyqtSlot(str, bool)
    def load_nodes(self, filename='', silent=False):
        '''
        this is an overloaded function with the following functionalities:
        - create a new set of nodes via a GUI
        - edit an already loaded/created set of nodes via a GUI
        - load an existing set of nodes
            - and show/edit it 
            - load it silently
        filename = str (should exist and point to a whitespace 
            delimited ascii file (see class description)
        silent = bool (if False show the GUI)
        '''
        def csv_loader(filename):
            xs, ys, zs = {}, {}, {} # dictionary (why? -> see __init__)
            with open(filename, 'r') as f:
                for line in csv.reader(f, delimiter=' ', skipinitialspace=True):
                    if not line:
                        continue
                    node, x, y, z =  \
                        [float(line[i]) for i in range(4)]  # cut trailing empty columns
                    node = int(node)
                    xs[node], ys[node], zs[node] = x, y, z
            return xs, ys, zs

        self.show_creation_widget(type_='nodes')
        xs, ys, zs = {}, {}, {} # dictionary (why? -> see __init__)
        if self.xs and self.ys and self.zs:  # edit an existing grid
            xs, ys, zs = self.xs, self.ys, self.zs
        elif not os.path.exists(filename):  # load grid by button request
            filename = QFileDialog.getOpenFileName(
                caption='Open Gridfile', filter="*.asc", directory=self.last_path)
            self.last_path = os.path.dirname(filename)
        # load grid by filename; continue button loading
        if os.path.exists(filename):
            xs, ys, zs = csv_loader(filename)
        # add nodes to the table, to the plot and to the coordinates lists
        for key in xs.keys():
            self.add_node(xs[key], ys[key], zs[key], key)
        if silent and xs and ys and zs:  # don't show the GUI
            self.hide()

    @pyqtSlot()
    @pyqtSlot(str)
    @pyqtSlot(str, bool)
    def load_lines(self, filename='', silent=False):
        '''
        this is an overloaded function with the following functionalities:
        - create a new set of lines via a GUI
        - edit an already loaded/created set of lines via a GUI
        - load an existing set of lines
            - and show/edit it 
            - load it silently
        filename = str (should exist and point to a whitespace 
            delimited ascii file (see class description)
        silent = bool (if False show the GUI)
        
        nodes have to exis before loading/creating lines
        '''
        def csv_loader(filename):
            lines = []
            with open(filename, 'r') as f:
                for line in csv.reader(f, delimiter=' ', skipinitialspace=True):
                    if not line:
                        continue
                    node_start, node_end = \
                        [int(float(line[i])) for i in range(2)]
                    lines.append((node_start, node_end))
            return lines

        try:
            self.return_to_func.disconnect(self.load_lines)
        except TypeError: # signal is not connected to slot
            pass
        
        # check if nodes are present
        if not self.xs or not self.ys or not self.zs:
            self.return_to_func.connect(self.load_lines)
            QMessageBox.information(
                self, 'Information', 
                'No Nodes found! Create or load nodes first!')
            self.load_nodes()
            return

        self.show_creation_widget(type_='lines')
        lines = []
        if self.lines:  # edit  existing lines
            lines = self.lines
        elif not os.path.exists(filename):  # load lines by button request
            filename = QFileDialog.getOpenFileName(
                caption='Open Beam', filter="*.asc", directory=self.last_path)
            self.last_path = os.path.dirname(filename)

        # load lines by filename, continue button loading
        if os.path.exists(filename):
            lines = csv_loader(filename)
        # add lines to the table, to the plot and to the coordinates lists
        for line in lines:
            self.add_line(line)
        if silent and lines:  # don't show the GUI
            self.hide()

    @pyqtSlot()
    @pyqtSlot(str)
    @pyqtSlot(str, bool)
    def load_master_slave(self, filename='', silent=False):
        '''
        this is an overloaded function with the following functionalities:
        - create a new set of master slave definitions via a GUI
        - edit an already loaded/created set of master slave definitions via a GUI
        - load an existing set of master slave definitions
            - and show/edit it 
            - load it silently
        filename = str (should exist and point to a whitespace 
            delimited ascii file (see class description)
        silent = bool (if False show the GUI)
        
        nodes have to exist before loading/creating master slave definitions
        '''
        def csv_loader(filename):
            master_slaves = []
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
                for line in reader:
                    if not line:
                        continue
                    i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl = \
                            [float(line[i]) for i in range(8)]
                    i_m, i_sl = [int(node) for node in [i_m, i_sl]]
                    master_slaves.append(
                        (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl))
            return master_slaves

        try:
            self.return_to_func.disconnect(self.load_master_slave)
        except TypeError: # signal is not connected to slot
            pass

        # check if nodes are present
        if not self.xs or not self.ys or not self.zs:
            self.return_to_func.connect(self.load_master_slave)
            QMessageBox.information(
                self, 'Information', 
                'No Nodes found! Create or load nodes first!')
            self.load_nodes()
            return

        self.show_creation_widget(type_='master_slaves')
        master_slaves = []
        if self.master_slaves:  # edit  existing master_slaves
            master_slaves = self.master_slaves
        elif not os.path.exists(filename):  # load master_slaves by button request
            filename = QFileDialog.getOpenFileName(
                caption='Open Master-Slave-File', 
                filter="*.asc", directory=self.last_path)
            self.last_path = os.path.dirname(filename)

        # load master_slaves by filename; continue button loading
        if os.path.exists(filename):
            master_slaves = csv_loader(filename)
        # add master_slaves to the table, to the plot and to the list
        for master_slave in master_slaves:
            self.add_master_slave(*master_slave)
        if silent and master_slaves:  # don't show the GUI
            self.hide()

    @pyqtSlot()
    @pyqtSlot(str)
    @pyqtSlot(str, bool)
    def load_chan_dof(self, filename='', silent=False):
        '''
        this is an overloaded function with the following functionalities:
        - create a new set of channel-DOF-assignments via a GUI
        - edit an already loaded/created set of channel-DOF-assignments via a GUI
        - load an existing set of master slave definitions
            - and show/edit it 
            - load it silently
        filename = str (should exist and point to a whitespace 
            delimited ascii file (see class description)
        silent = bool (if False show the GUI)
        
        nodes have to exist before loading/creating channel-DOF-assignments
        '''
        def csv_loader(filename):
            chan_dofs = []
            with open(filename, 'r') as f:
                for line in csv.reader(f, delimiter=' ', skipinitialspace=True):
                    if not line:
                        continue  # skip empty lines
                    chan, node, x_amp, y_amp, z_amp = \
                        [float(line[i]) for i in range(5)]  # cut trailing empties
                    x_amp = int(x_amp * 10**3)/10**3
                    y_amp = int(y_amp * 10**3)/10**3
                    z_amp = int(z_amp * 10**3)/10**3
                    
                    chan_dofs.append(
                        (int(chan), int(node), (x_amp, y_amp, z_amp)))
            return chan_dofs
        
        def json_loader(filename):
            import json
            chan_dofs_file=json.load(open(filename))
            
            chan_dofs=[]
            if isinstance(chan_dofs_file, dict):
                ok=False
                if len(chan_dofs_file.keys())>1:
                    while not ok:
                        measurement_name, ok = QInputDialog.getText(self, 'Input Dialog', 
                        'Choose measurement_name: {}'.format(str(list(chan_dofs_file.keys()))))
                else: measurement_name = list(chan_dofs_file.keys())[0]
                chan_dofs_file = chan_dofs_file[measurement_name]
                
            for chan, node, az, elev in chan_dofs_file:
                az = az/180*np.pi
                elev=elev/180*np.pi
                x_amp=1*np.cos(elev)*np.cos(az) # for elevation angle defined from XY-plane up
                #x=r*numpy.sin(elev)*numpy.cos(az) # for elevation angle defined from Z-axis down
                y_amp=1*np.cos(elev)*np.sin(az) # for elevation angle defined from XY-plane up
                #y=r*numpy.sin(elev)*numpy.sin(az)# for elevation angle defined from Z-axis down
                z_amp=1*np.sin(elev)
                chan_dofs.append(
                        (int(chan), int(node), (x_amp, y_amp, z_amp)))
            return chan_dofs
        
        def numpy_loader(filename):
            chan_dofs_file = np.load(filename)
            chan_dofs=[]
            for chan, node, az, elev in chan_dofs_file:
                az = az/180*np.pi
                elev=elev/180*np.pi
                x_amp=1*np.cos(elev)*np.cos(az) # for elevation angle defined from XY-plane up
                #x=r*numpy.sin(elev)*numpy.cos(az) # for elevation angle defined from Z-axis down
                y_amp=1*np.cos(elev)*np.sin(az) # for elevation angle defined from XY-plane up
                #y=r*numpy.sin(elev)*numpy.sin(az)# for elevation angle defined from Z-axis down
                z_amp=1*np.sin(elev)
                chan_dofs.append(
                        (int(chan), int(node), (x_amp, y_amp, z_amp)))
            return chan_dofs
                        
        try:
            self.return_to_func.disconnect(self.load_chan_dof)
        except TypeError: # signal is not connected to slot
            pass

        # check if nodes are present
        if not self.xs or not self.ys or not self.zs:
            self.return_to_func.connect(self.load_chan_dof)
            QMessageBox.information(
                self, 'Information', 
                'No nodes found! Create or load nodes first!')
            self.load_nodes()
            return

        self.show_creation_widget(type_='chan_dofs')
        chan_dofs = []

        if self.chan_dofs:  # edit existing channel-DOF-assignments
            chan_dofs = self.chan_dofs
        elif not os.path.exists(filename):  # load channel-DOF-assignments by button request
            filename = QFileDialog.getOpenFileName(
                caption='Open Channel DOF Assignment File', 
                #filter="*.asc",
                 directory=self.last_path)
            self.last_path = os.path.dirname(filename)

        # load channel-DOF-assignments by filename, continue button loading
        if os.path.exists(filename):
            try:
                chan_dofs = csv_loader(filename)
            except:
                try:
                    chan_dofs = json_loader(filename)
                except:
                    chan_dofs = numpy_loader(filename)
        # add channel-DOF-assignments to the table, to the plot and to the lists
        for chan, node, dof in chan_dofs:
            self.add_chan_dof(chan, node, dof)
        if silent and chan_dofs:  # don't show the GUI
            self.hide()
            
    @pyqtSlot()
    def remove_rows(self, rows=None):
        '''
        remove rows from the listwidget either pass a list-of-int or call 
        via a signal from a button and get selected rows from the list widget
        based on the number of columns in the list widget it is identified
        what type of geometry is currently edited 
        - columns = 4 -> nodes
        - columns = 2 -> lines
        - columns = 8 -> master slave definitions
        - columns = 5 -> channel-DOF assignments
        for each removed row a signal is emited containg the removed values
        so that they can be removed in any connected class, too
        when removing a node, a check is performed on the other objects
        if they are connected to this node and if yes they will be removed
        as well
        
        '''
        cols = self.coordinates_list.columnCount()
        if rows is None:
            items = self.coordinates_list.selectedItems()

            rows = set() # avoid duplicate rows if more than one item in a row is selected
            for item in items:
                rows.add(item.row())
            rows = list(rows)
            
        rows.sort()
        # removing from start would change numbering each time an item 
        # is removed, therefore reverse the list
        rows.reverse() 
        for row in rows:
            if row == -1: # no row selected
                continue

            if cols == 4: # nodes
                node = int(self.coordinates_list.item(row, 0).text())
                x = float(self.coordinates_list.item(row, 1).text())
                y = float(self.coordinates_list.item(row, 2).text())
                z = float(self.coordinates_list.item(row, 3).text())
                self.coordinates_list.removeRow(row)

                while True: # check if any channel is assigned to this node
                    for j, (chan, node_,  dof) in enumerate(self.chan_dofs):
                        if node == node_:
                            self.chan_dof_taken.emit(chan, node_, dof)
                            del self.chan_dofs[j]
                            break
                    else:
                        break

                while True: # check if any line is connected to this node
                    for j in range(len(self.lines)):
                        line = self.lines[j]
                        if node in line:
                            self.line_taken.emit(line)
                            del self.lines[j]
                            break
                    else:
                        break

                while True: # check if this node is a master or slave for another node
                    for j, master_slave in enumerate(self.master_slaves):
                        if node == master_slave[0] or node == master_slave[4]:
                            m = master_slave
                            self.slave_taken.emit(
                                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7])
                            del self.master_slaves[j]
                            break
                    else:
                        break
                    
                self.node_taken.emit(x, y, z, node)

                if self.xs[node] == x:
                    del self.xs[node]
                else:
                    print(
                        "self.xs[i] does not correspond to i'th "
                        "element in coordinates list")
                if self.ys[node] == y:
                    del self.ys[node]
                else:
                    print(
                        "self.ys[i] does not correspond to i'th "
                        "element in coordinates list")
                if self.zs[node] == z:
                    del self.zs[node]
                else:
                    print(
                        "self.zs[i] does not correspond to i'th "
                        "element in coordinates list")

            elif cols == 2:
                node_start = int(self.coordinates_list.item(row, 0).text())
                node_end = int(self.coordinates_list.item(row, 1).text())
                self.coordinates_list.removeRow(row)
                
                self.line_taken.emit((node_start, node_end))

                if self.lines[row] == (node_start, node_end):
                    del self.lines[row]
                else:
                    print(
                        "self.lines[i] does not correspond to i'th element in coordinates list")

            elif cols == 8:
                i_m = int(self.coordinates_list.item(row, 0).text())
                x_m = float(self.coordinates_list.item(row, 1).text())
                y_m = float(self.coordinates_list.item(row, 2).text())
                z_m = float(self.coordinates_list.item(row, 3).text())
                i_sl = int(self.coordinates_list.item(row, 4).text())
                x_sl = float(self.coordinates_list.item(row, 5).text())
                y_sl = float(self.coordinates_list.item(row, 6).text())
                z_sl = float(self.coordinates_list.item(row, 7).text())
                self.coordinates_list.removeRow(row)
                
                self.slave_taken.emit(
                    i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl)

                if self.master_slaves[row] == [i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl]:
                    del self.master_slaves[row]
                else:
                    print(
                        "self.master_slaves[i] does not correspond to i'th element in coordinates list")

            elif cols == 5:
                chan = int(self.coordinates_list.item(row, 0).text())
                node = int(self.coordinates_list.item(row, 1).text())
                x_amp = float(self.coordinates_list.item(row, 2).text())
                y_amp = float(self.coordinates_list.item(row, 3).text())
                z_amp = float(self.coordinates_list.item(row, 4).text())
                self.coordinates_list.removeRow(row)
                
                self.chan_dof_taken.emit(
                    chan, node, (x_amp, y_amp, z_amp))

                if self.chan_dofs[row][0] == chan and \
                    self.chan_dofs[row][1] == node and \
                    nearly_equal(self.chan_dofs[row][2][0],x_amp,3) and \
                    nearly_equal(self.chan_dofs[row][2][1],y_amp,3) and \
                    nearly_equal(self.chan_dofs[row][2][2],z_amp,3):
                    
                    del self.chan_dofs[row]
                else:
                    print(
                        "self.chan_dofs[i] does not correspond to i'th element in coordinates list")

    @pyqtSlot()
    def clear_list(self):
        '''
        convenience function to remove all rows in a list
        '''
        rows = self.coordinates_list.rowCount()
        self.remove_rows(list(range(rows)))   
                        
    @pyqtSlot()
    def close_return(self):
        '''
        function is used in load_lines, load_master_slave and load_chan_dof
        if there are no nodes present load_* function will connect itself 
        to the return_to_func signal such that when the user is done
        loading/creating the grid this close_return function is called,
        which will hide the creation_widget for nodes and emit the 
        return_to_func signal, which will call the load_* function again
        '''
        self.hide()
        self.return_to_func.emit()


    @pyqtSlot()
    def save_file(self):
        '''
        save the contents of the listwidget to a whitespace delimited
        ascii file
        '''
        filename = QFileDialog.getSaveFileName(
            caption='Save File', filter="*.asc", directory=self.last_path)
        self.last_path = os.path.dirname(filename)
        if filename == "":
            return
        if not filename.endswith('.asc'):
            filename += '.asc'
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for row in range(self.coordinates_list.rowCount()):
                row_list = []
                for column in range(self.coordinates_list.columnCount()):
                    row_list.append(
                        self.coordinates_list.item(row, column).text())
                writer.writerow(row_list)


def main():
    app = QApplication(sys.argv)
    
    
    series_num=5
    
    project_path = os.path.expanduser('~/Documents/Uni/masterarbeit/') 
    
    measurement_series=['sim_i_full/', # 0
                        'sim_ii_pre-conv-(singl)-refacc/', # 1
                        'sim_iii_post-conv-singl-refacc/', # 2
                        'sim_iv_post-conv-distr-refacc/', # 3
                        'sim_v_post-conv-distr-refvib/', # 4
                        'sim_vi_post-conv-distr-refself/',# 5
                        'slang_base_results'] #6
    measurement_series=['meas_i_full/', # 0 no exist
                        'meas_ii_pre-conv-(singl)-refacc/', # 1 good
                        'meas_iii_post-conv-singl-refacc/', # 2 good
                        'meas_iv_post-conv-distr-refacc/', # 3 good
                        'meas_v_post-conv-distr-refvib/', # 4
                        'meas_vi_post-conv-distr-refself/'
                        ]
    
    measurement_path = project_path + 'messungen_simulationen/' + measurement_series[series_num]
    
    merge = 0
    ortho = True
    if merge == 0: # PoSER
        if ortho:
            
            result_folder = measurement_path + 'merged_poser_ortho'  + '/'  
        else:
            result_folder = measurement_path + 'merged_poser'  + '/' 
    elif merge == 1: #PoGER
        if ortho:
            if series_num != 5:
                result_folder = measurement_path + 'merged_poger_ortho'  + '/'   
            else:
                result_folder = measurement_path + 'merged_poger_poser_ortho'  + '/'  
        elif series_num != 5:
            result_folder = measurement_path + 'merged_poger'  + '/'    
        else:
            return
    elif merge == 2: #PreGER
        if ortho:
            if series_num != 5:
                result_folder = measurement_path + 'merged_preger_ortho'  + '/'   
            else:
                result_folder = measurement_path + 'merged_preger_poser_ortho'  + '/'  
        elif series_num != 5:
            result_folder = measurement_path + 'merged_preger'  + '/'    
        else:
            return
    else:
        result_folder = measurement_path + '/'
    
    print(result_folder)
    
    chan_dofs_path = result_folder + 'chan_dofs.txt'
    
    grid_path = project_path+ 'messungen_simulationen/macec/grid_full.asc'
    #grid_path = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/macec/grid_2_4_ref5.asc'
    beam_path = project_path+ 'messungen_simulationen/macec/beam_full.asc'
    #beam_path = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/macec/beam_2_4_ref5.asc'
    #chan_dofs_path = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/sim_ii_pre-conv-(singl)-refacc/PreGER/merged/chan_dofs.txt'
    #chan_dofs_path = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/incomplete/sim_ii/merged/chan_dofs.txt'
    #chan_dofs_path = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/sim_ii_pre-conv-(singl)-refacc/merged/chan_dofs.txt'
    #chan_dofs_path = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/sim_i_full/channel_dof_assignment.txt'
    #chan_dofs_path = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/sim_iii_post-conv-singl-refacc/merged/chan_dofs.txt'
    #result_folder = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/sim_i_full/measurement_1/'
    #result_folder = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/sim_ii_pre-conv-(singl)-refacc/PreGER/merged/'
    #result_folder = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/incomplete/sim_ii/merged/'
    #result_folder = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/sim_iii_post-conv-singl-refacc/merged/'
    #result_folder = '/home/simon/Dropbox/simon_master/masterarbeit/messungen_simulationen/sim_ii_pre-conv-(singl)-refacc/PoSER/merged/'
    
    form = PlotMSH(#size=(5, 4), dpi=100,
                   grid_path = grid_path,
                   beam_path = beam_path,
                   #slave_path='/home/simon/Dropbox/hiwi/PlotMSH/slaves.asc',
                   chan_dofs_path = chan_dofs_path,
                   ssi_solutions_path = result_folder,)
    app.exec_()
    

if __name__ == "__main__":
    main()
