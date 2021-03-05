'''
Module PlotMSH contains classes and functions for plotting mode shapes
obtained from any of the classes derived from ModalBase of the pyOMA project

.. TODO::
 * Implement scale (for correct drawing of axis arrows)
 * Use current axes settings when starting the animation
 * Remove PyQT dependency -> move the signal definitions somewhere else
 * Restore functionality needed to create the geometry in another GUI
 * Use the logging module to replace print commands at an appropriate 
   logging level
 * Implement the plotting in  pyvista for better and faster 3D graphics 
   `https://docs.pyvista.org/examples/99-advanced/warp-by-vector-eigenmodes.html`
 
'''

# system i/o
import os
import logging
logger = logging.getLogger('')

from PyQt5.QtCore import pyqtSignal

# Matplotlib
import matplotlib
# check if python is running in headless mode i.e. as a server script
if 'DISPLAY' in os.environ:
    matplotlib.use("Qt5Agg",force=True) 
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D, proj3d  # @UnresolvedImport
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import is_color_like
import matplotlib.animation 
from matplotlib.markers import MarkerStyle



import numpy as np
#tools
import itertools
from classes.StabilDiagram import StabilCalc
from classes.PreprocessingTools import PreprocessData, GeometryProcessor

from classes.ModalBase import ModalBase
from classes.SSICovRef import PogerSSICovRef
from classes.VarSSIRef import VarSSIRef
from classes.PostProcessingTools import MergePoSER

NoneType = type(None)



def nearly_equal(a,b,sig_fig=5):
    return ( a==b or 
             int(a*10**sig_fig) == int(b*10**sig_fig)
           )


def orthogonal_proj(zfront, zback):
    '''
    monkeypatch to allow orthogonal projection in mplot3d
    breaks automatic placement of axis
    credit: https://github.com/matplotlib/matplotlib/issues/537
    '''
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, a, b],
                        [0, 0, 0, zback]])


def persp_transformation(zfront, zback):
    ''' 
    copy of mpl_toolkits.mplot3d.proj3d.persp_transformation
    for restoring the projection in isonometriv view
    '''
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, a, b],
                        [0, 0, -1, 0]
                        ])

old_draw = Axes3D.draw
def draw_axes(self, renderer=None):
    ''' 
    monkeypatch draw method to always enforce an aspect ratio of 1 on all axis'
    '''
    
    minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
    dx,dy,dz=(maxx-minx),(maxy-miny),(maxz-minz)
    
    if dx != dy or dx != dz:            
        midx=0.5*(minx+maxx)
        midy=0.5*(miny+maxy)
        midz=0.5*(minz+maxz)
    
        hrange=max(dy,dy,dz)*0.5
        self.set_xlim3d(midx-hrange , midx+hrange)
        self.set_ylim3d(midy-hrange , midy+hrange)
        self.set_zlim3d(midz-hrange , midz+hrange)
    old_draw(self, renderer)

class ModeShapePlot(object):
    '''
    This class is used for displaying modal values and modeshapes obtained
    by one of the classes derived from ModalBase as part the of the pyOMA project  
    (Bauhaus-Universität Weimar, Institut für Strukturmechanik).
    
    Test
    Test
    Test
    
    
    Drawing abilities:
        * creation of 3d plots using matplotlib's mplot3 from the 
          matplotlib toolkit
        * adjusting axis limits for each of the three axis
          i.e. zoom view, shift view (along single and multiple axis')
        * change the viewport e.g. x, y, z and isometric view
        * rotating and zooming through mouse interaction is currently 
          supported by matplotlib, whereas panning is not
        * animate the currently displayed deformed structure
        * save the still frame

    currently **not** supported:
        * 3D surface plots, as they are not properly supported by the 
          underlying matplotlib api
        * combination of several modeshapes or sensor setups 
          (this could be done easily in an external script)
        * saving of the animation as a movie file
        * drawing multiple modeshapes into one plot
        * plot modeshape in a single call from a script i.e. use static methods
    
    
    +----------------+--------------+-------------+--------------+
    |Variable in     |  Merging Routine                          |
    |PlotMSH         +--------------+-------------+--------------+
    |                | single-setup |poger/preger |poser merging |
    +----------------+--------------+-------------+--------------+
    |modal_freq.     | modal_data   |modal_data   |merged_data   |
    +----------------+--------------+-------------+--------------+
    |modal_damping   | modal_data   |modal_data   |merged_data   |
    +----------------+--------------+-------------+--------------+
    |modeshapes      | modal_data   |modal_data   |merged_data   |
    +----------------+--------------+-------------+--------------+
    |num_channels    | prep_data    |modal_data   |merged_data   |
    +----------------+--------------+-------------+--------------+
    |chan_dofs       | prep_data    |modal_data   |merged_data   |
    +----------------+--------------+-------------+--------------+
    |select_modes    | stabil_data  |stabil_data  |merged_data   |
    +----------------+--------------+-------------+--------------+
    |nodes           | geometry_data|geometry_data|geometry_data |
    +----------------+--------------+-------------+--------------+
    |lines           | geometry_data|geometry_data|geometry_data |
    +----------------+--------------+-------------+--------------+
    |master-slaves   | geometry_data|geometry_data|geometry_data |
    +----------------+--------------+-------------+--------------+
                     
    '''
    # define this class's signals and the types of data they emit
    grid_requested = pyqtSignal(str, bool)
    beams_requested = pyqtSignal(str, bool)
    slaves_requested = pyqtSignal(str, bool)
    chan_dofs_requested = pyqtSignal(str, bool)

    def __init__(self,
                 geometry_data,
                 stabil_calc=None,
                 modal_data=None,
                 prep_data=None,
                 merged_data=None,
                 selected_mode=[0,0],
                 amplitude=1,
                 real=False,
                 scale=1, #0.1*10^x [m] where x=scale
                 dpi=100,
                 nodecolor='blue',
                 nodemarker='o',
                 nodesize=20,
                 beamcolor='blue',
                 beamstyle='-',
                 linewidth = 1,
                 callback_fun=None
                 ):
        '''
        Parameters 
        ------------
            geometry_data : PreprocessingTools.GeometryProcessor
                    Object containing all the necessary geometry information.
            
            stabil_calc : StabilDiagram.StabilCalc, optional
                    Object containing the information, which modes were 
                    selected from modal_data.
                    
            modal_data : ModalBase.ModalBase, optional
                    Object of one the classes derived from ModalBase.ModalBase,
                    containing the estimated modal parameters at multiple
                    model orders.
                    
            prep_data : PreprocessingTools.PreprocessData, optional
                    Object containing the measurement data and information 
                    about it.
                    
            merged_data : PostProcessingTools.MergePoSER, SSICovRef.PogerSSICovRef, optional
                    Object containing the merged data
                    
            selected_mode : list, optional
                    List of [model_order, mode_index] to define the mode
                    that is displayed upon startup 
                    
            amplitude : float, optional
                    Scaling factor to scale the magnitude of mode shape displacements
                    
            real : bool, optional
                    Whether to plot only the real part or the magnitude 
                    of the complex modal coordinates
                    
            scale : float, optional
                    Scaling factor for other elements such as arrows, etc.
                    
            dpi : float, optional
                    Resolution of the drawing canvas
                    
            nodecolor : matplotlib color, optional
                    Color which is used to draw the nodes
                    
            nodemarker : matplotlib marker, optional
                    Marker which is used to draw the nodes
                    
            nodesize : float, optional
                    Marker size for the nodes
                    
            beamcolor : matplotlib color, optional
                    Color which is used to draw the lines
                    
            beamstyle : matplotlib linestyle, optional
                    Linestyle which is used to draw the lines
                    
            linewidth : float, optional
                    Line width which is used to draw the lines
                    
            callback_fun : function, optional
                    A function that is executed upon changing to a new 
                    mode, allows to print mode information or change some 
                    other behaviour of the class. It takes the class itself
                    and the mode index as its parameters.
        '''
        if stabil_calc is not None:
            assert isinstance(stabil_calc, StabilCalc)
        self.stabil_calc = stabil_calc
        

        
        #modal_data = modal_data
        if modal_data is not None:
            assert isinstance(modal_data, ModalBase)
        self.modal_data = modal_data
        
        assert isinstance(geometry_data, GeometryProcessor)
        self.geometry_data = geometry_data
        
        #prep_data = prep_data
        if prep_data is not None:
            assert isinstance(prep_data, PreprocessData)
        self.prep_data = prep_data
        
        if merged_data is not None:
            assert isinstance(merged_data, MergePoSER)
        

        self.merged_data = merged_data
        
        if merged_data is not None:
            self.chan_dofs = merged_data.merged_chan_dofs
            self.num_channels = merged_data.merged_num_channels
        
            self.modal_frequencies = merged_data.mean_frequencies
            self.modal_damping = merged_data.mean_damping
            self.mode_shapes = merged_data.merged_mode_shapes
        
            self.std_frequencies = merged_data.std_frequencies
            self.std_damping = merged_data.std_damping
            
            self.select_modes = list(zip(range(len(self.modal_frequencies)),[0]*len(self.modal_frequencies)))
            
            self.setup_name = merged_data.setup_name
            self.start_time = merged_data.start_time
        elif isinstance(modal_data, PogerSSICovRef):
            self.chan_dofs = modal_data.merged_chan_dofs
            self.num_channels = modal_data.merged_num_channels
        
            self.modal_frequencies = modal_data.modal_frequencies
            self.modal_damping = modal_data.modal_damping
            self.mode_shapes = modal_data.mode_shapes
                
            self.select_modes = stabil_calc.select_modes
            
            self.setup_name = modal_data.setup_name
            self.start_time = modal_data.start_time
            
        elif prep_data is not None:
            self.chan_dofs = prep_data.chan_dofs
            self.num_channels = prep_data.num_analised_channels
            
            if modal_data is not None:
                
                self.modal_frequencies = modal_data.modal_frequencies
                self.modal_damping = modal_data.modal_damping
                self.mode_shapes = modal_data.mode_shapes
                
                if isinstance(modal_data, VarSSIRef):
                    self.std_frequencies = modal_data.std_frequencies
                    self.std_damping = modal_data.std_damping
                else:
                    self.std_frequencies = None
                    self.std_damping = None
            else:
                self.modal_frequencies = np.array([[]])
                self.modal_damping = np.array([[]])
                self.mode_shapes = np.array([[[]]])
                self.std_frequencies = None
                self.std_damping = None
                
            if stabil_calc is not None:
                self.select_modes = stabil_calc.select_modes
                
                self.setup_name = modal_data.setup_name
                self.start_time = modal_data.start_time
            else:
                self.select_modes = []
                self.setup_name = ''
                self.start_time = None
                
        else:
            self.chan_dofs = []
            self.num_channels = 0
            self.modal_frequencies = np.array([[]])
            self.modal_damping = np.array([[]])
            self.mode_shapes = np.array([[[]]])
            self.select_modes = []
            self.setup_name = ''
            self.start_time = None
        #if not geometry_data:
        #geometry_data = prep_data.geometry_data          
        
        self.disp_nodes = { i : [0,0,0] for i in self.geometry_data.nodes.keys() }
        self.phi_nodes = { i : [0,0,0] for i in self.geometry_data.nodes.keys() }
        
        # linestyles available in matplotlib
        styles = ['-', '--', '-.', ':', 'None', ' ', '', None]
        
        #markerstylesavailable in matplotlib
        markers = list(MarkerStyle.markers.keys())
        
        assert isinstance(real, bool)
        self.real = real  
             
        assert isinstance(scale, (int,float))
        self.scale = scale

        assert is_color_like(beamcolor) or isinstance(beamcolor, (list, tuple, np.ndarray))
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

        assert isinstance(dpi, int)
        self.dpi = dpi

        assert isinstance(amplitude, (int, float))
        self.amplitude = amplitude
        
        assert isinstance(linewidth, (int,float))
        self.linewidth = linewidth
        
        if callback_fun is not None:
            assert callable(callback_fun)
        self.callback_fun = callback_fun

        #bool objects
        self.show_nodes = True
        self.show_lines = True
        self.show_nd_lines = True
        self.show_master_slaves = True
        self.show_chan_dofs = True
        self.show_axis = True
        self.animated = False
        self.data_animated = False
        self.draw_trace = True
        self.save_ani = False

        # plot objects
        self.patches_objects = {}
        self.lines_objects = []
        self.nd_lines_objects = []
        self.cn_lines_objects = {}
        self.arrows_objects = []
        self.channels_objects = []
        self.trace_objects = []
        self.axis_obj = {}
        self.seq_num = 0
        
        self.fig = Figure((10,10), dpi=dpi, facecolor='white')
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvasBase(self.fig)
        
        try:  # mpl 1.4
            self.subplot = self.fig.add_subplot(1, 1, 1, projection='3d')
        except ValueError:  # mpl 1.3
            self.subplot = Axes3D(self.fig)
        Axes3D.draw = draw_axes
        
        self.subplot.grid(False)
        self.subplot.set_axis_off()
        
        self.mode_index = [0, 0]
        
        #instantiate the x,y,z axis arrows
        self.draw_axis()
    
    #@pyqtSlot()
    def reset_view(self):
        '''
        restore viewport, 
        restore axis' limits, 
        reset displacements values for all nodes, 
        '''
        self.stop_ani()
        proj3d.persp_transformation = persp_transformation
        self.subplot.view_init(30, -60)
        self.subplot.autoscale_view()
        xmin,xmax,ymin,ymax,zmin,zmax=None,None,None,None,None,None
        for node in self.geometry_data.nodes.values():
            if xmin is None: xmin=node[0]
            if xmax is None: xmax=node[0]
            if ymin is None: ymin=node[1]
            if ymax is None: ymax=node[1]
            if zmin is None: zmin=node[2]
            if zmax is None: zmax = node[2]
            xmin=min(node[0],xmin)
            xmax=max(node[0],xmax)
            ymin=min(node[1],ymin)
            ymax=max(node[1],ymax)
            zmin=min(node[2],zmin)
            zmax=max(node[2],zmax)
        self.subplot.set_xlim3d(xmin,xmax)
        self.subplot.set_ylim3d(ymin,ymax)
        self.subplot.set_zlim3d(zmin,zmax)    
        
        self.draw_nodes()
        self.draw_lines()
        self.draw_chan_dofs()
        self.draw_master_slaves()
        if self.mode_index[1]:
            self.draw_msh()
        
        #self.disp_nodes = { i : [0,0,0] for i in self.geometry_data.nodes.keys() }
        
        self.canvas.draw()
        

    #@pyqtSlot()
    def change_viewport(self, viewport=None):
        '''
        change the viewport
        works in combination with the appropriate buttons as senders or
        by passing one of ['X', 'Y', 'Z', 'ISO']
        
        '''
            
        if viewport == 'X':
            azim, elev = 0, 0
            #proj3d.persp_transformation = orthogonal_proj
            self.subplot.set_proj_type('ortho')
        elif viewport == 'Y':
            azim, elev = 270, 0
            #proj3d.persp_transformation = orthogonal_proj
            self.subplot.set_proj_type('ortho')
        elif viewport == 'Z':
            azim, elev = 0, 90
            #proj3d.persp_transformation = orthogonal_proj
            self.subplot.set_proj_type('ortho')
        elif viewport == 'ISO':
            azim, elev = -60, 30
            #proj3d.persp_transformation = persp_transformation
            self.subplot.set_proj_type('persp')
        else:
            print('viewport not recognized: ', viewport)
            azim, elev = -60, 30
            #proj3d.persp_transformation = persp_transformation
            self.subplot.set_proj_type('persp')
        self.subplot.view_init(elev, azim)
        self.canvas.draw()

        if self.animated or self.data_animated:
            for line in self.lines_objects:
                line.set_visible(False)
            for line in self.nd_lines_objects:
                line.set_visible(False)
            for line in self.cn_lines_objects.values():
                line.set_visible(False)
            self.line_ani._setup_blit()

    #@pyqtSlot(str)
    def change_mode(self, frequency=None, index=None, mode_index=None,):
        '''
        if user selects a new mode,
        extract the mode number from the passed string (contains frequency...)
        write modal values to the infobox
        and plot the mode shape
        '''
        # mode numbering starts at 1 python lists start at 0
        selected_indices = self.select_modes
        if frequency is not None:       
            frequencies =np.array([self.modal_frequencies[index[0],index[1]] for index in selected_indices])     
            f_delta= abs(frequencies-frequency)
            index = np.argmin(f_delta)
        #print(selected_indices)
        if index is not None:
            mode_index = selected_indices[index]
        if mode_index is None:
            raise RuntimeError('No arguments provided!')
        #print(mode_index)
        frequency = self.modal_frequencies[mode_index[0],mode_index[1]]
        damping = self.modal_damping[mode_index[0],mode_index[1]]
        if self.stabil_calc:
            MPC = self.stabil_calc.MPC_matrix[mode_index[0],mode_index[1]]
            MP = self.stabil_calc.MP_matrix[mode_index[0],mode_index[1]]
            MPD = self.stabil_calc.MPD_matrix[mode_index[0],mode_index[1]]
        else:
            MPC=None
            MP=None
            MPD=None
        self.mode_index = mode_index
        
        if self.save_ani:
            cwd=os.getcwd()
            cwd += '/{}/'.format(self.select_modes.index(self.mode_index))
            if not os.path.exists(cwd):
                os.makedirs(cwd)
        
        self.draw_msh()

        #print('self.callback_fun', self.callback_fun)
        if self.callback_fun is not None:
            #print('call')
            try:
                self.callback_fun(self,mode_index)
            except Exception as e:
                print(e)
                pass
        
        return mode_index[1], mode_index[0], frequency, damping, MPC, MP, MPD #order, mode_num,....

    def get_frequencies(self):
        selected_indices = self.select_modes
        
        frequencies =[self.modal_frequencies[index[0],index[1]] for index in selected_indices]
        frequencies.sort()
        return frequencies
    
    #@pyqtSlot()
    #@pyqtSlot(float)
    def change_amplitude(self, amplitude=None):
        '''
        changes the amplitude
        amplitude either gets passed or will be read from the widget
        redraw the modeshapes based on this amplitude
        '''
        if amplitude is None: return
        amplitude = float(amplitude)
        if amplitude == self.amplitude:return

        self.amplitude = amplitude
        
        if self.mode_shapes.shape[2]:
            self.draw_msh()

    #@pyqtSlot(bool)
    def change_part(self, b):
        '''
        change, which part of the complex number modeshapes should be 
        drawn, set the pointer variable based on which widget sent the signal
        redraw the modeshapes 
        '''
        if b == self.real: return
        
        self.real = b
        self.draw_msh()
        

    def save_plot(self, path=None):
        '''
        save the curently displayed frame as a \*.png graphics file
        '''

        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            
            
    #@pyqtSlot(float, float, float, int)
    def add_node(self, x, y, z, i):
        '''
        receive a node from a signal
        add zero-value displacements for this node to the internal displacements table
        draw a single point at coordinates
        draw the number of the node
        store the two plot objects in a table
        remove any objects that might be in the table at the desired place
        to avoid duplicate nodes
        '''
        # leave present value if there is any else put 0
        self.disp_nodes[i] = self.disp_nodes.get(i,[0,0,0])        

        x, y, z = x + self.disp_nodes[i][0], y + self.disp_nodes[i][1], z + \
            self.disp_nodes[i][2]  # draw displaced nodes
        
        patch = self.subplot.scatter(
            x, y, z, color=self.nodecolor, marker=self.nodemarker, s=self.nodesize, visible = self.show_nodes)
        text = self.subplot.text(x, y, z, i, visible = self.show_nodes)

        if self.patches_objects.get(i) is not None:
            if isinstance(self.patches_objects[i], (tuple, list)):
                for obj in self.patches_objects[i]:
                    try: obj.remove()
                    except: pass
                    
        self.patches_objects[i] = (patch, text)
        
        self.canvas.draw_idle()

    #@pyqtSlot(tuple, int)
    def add_line(self, line, i):
        '''
        receive a line coordinates from a signal
        add the start node and end node to the internal line table
        draw a line between the tow nodes
        store the line object in a table
        remove any objects that might be in the table at the desired place
        i.e. avoid duplicate lines
        '''
        if isinstance(self.beamcolor, (list, tuple, np.ndarray)):
            beamcolor = self.beamcolor[i]
        else:
            beamcolor = self.beamcolor
        if isinstance(self.beamstyle, (list, tuple, np.ndarray)):
            beamstyle = self.beamstyle[i]
        else:
            beamstyle = self.beamstyle

        line_object = self.subplot.plot(
                        [self.geometry_data.nodes[node][0] + self.disp_nodes[node][0] for node in line],
                        [self.geometry_data.nodes[node][1] + self.disp_nodes[node][1] for node in line],
                        [self.geometry_data.nodes[node][2] + self.disp_nodes[node][2] for node in line],
                        color=beamcolor, linestyle=beamstyle,  visible = self.show_lines, linewidth = self.linewidth)[0]

        while len(self.lines_objects) < i + 1:
            self.lines_objects.append(None)
        if self.lines_objects[i] is not None:
            try:
                self.lines_objects[i].remove()
            except ValueError:
                pass
                #del self.lines_objects[i]
        self.lines_objects[i] = line_object

        self.canvas.draw_idle()
        
    #@pyqtSlot(tuple, int)
    def add_nd_line(self, line, i):
        '''

        '''
        if isinstance(self.beamcolor, (list, tuple, np.ndarray)):
            beamcolor = self.beamcolor[i]
        else:
            beamcolor = self.beamcolor

        beamstyle = 'dotted'
        
        line_object = self.subplot.plot(
                        [self.geometry_data.nodes[node][0]  for node in line],
                        [self.geometry_data.nodes[node][1]  for node in line],
                        [self.geometry_data.nodes[node][2]  for node in line],
                        color=beamcolor, linestyle=beamstyle,  linewidth = self.linewidth, visible = self.show_lines)[0]

        while len(self.nd_lines_objects) < i + 1:
            self.nd_lines_objects.append(None)
        if self.nd_lines_objects[i] is not None:
            try:
                self.nd_lines_objects[i].remove()
            except ValueError:
                pass
                #del self.nd_lines_objects[i]
        self.nd_lines_objects[i] = line_object

        self.canvas.draw_idle() 
           
    #@pyqtSlot(tuple, int)
    def add_cn_line(self, i):
        '''

        '''

        beamcolor = 'lightgray'

        beamstyle = 'dotted'
        node = self.geometry_data.nodes[i]
        disp_node = self.disp_nodes.get(node, [0,0,0])
        
        line_object = self.subplot.plot(
                        [node[0],node[0]+disp_node[0]],
                        [node[1],node[1]+disp_node[1]],
                        [node[2],node[2]+disp_node[2]],
                        color=beamcolor, linestyle=beamstyle,  linewidth = self.linewidth, visible = self.show_nd_lines)[0]

        if self.cn_lines_objects.get(i,None) is not None:
            try:
                self.cn_lines_objects[i].remove()
            except ValueError:
                pass
        self.cn_lines_objects[i] = line_object

        self.canvas.draw_idle()   
     
        
    #@pyqtSlot(int, float, float, float, int, float, float, float, int)
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


        #self.undraw_lines()  # do not show beams, as they are distracting
        #self.undraw_chan_dofs()

        color = "bgrcmyk"[int(np.fmod(i, 7))]  # equal colors for both arrows

        x_s, y_s, z_s = self.geometry_data.nodes[i_m]
        ((x_s, x_m), (y_s, y_m), (z_s, z_m)) = offset_arrows(
            ((x_s, x_m), (y_s, y_m), (z_s, z_m)), self.arrows_objects)

        # point the arrow towards the resulting direction
        arrow_m = Arrow3D([x_s, x_s + x_m], [y_s, y_s + y_m], [z_s, z_s + z_m],
                          mutation_scale=5, lw=1, arrowstyle="-|>", color=color, visible = self.show_master_slaves)
        arrow_m = self.subplot.add_artist(arrow_m)

        x_s, y_s, z_s = self.geometry_data.nodes[i_sl]
        ((x_s, x_sl), (y_s, y_sl), (z_s, z_sl)) = offset_arrows(
            ((x_s, x_sl), (y_s, y_sl), (z_s, z_sl)), self.arrows_objects)

        # point the arrow towards the resulting direction
        arrow_sl = Arrow3D([x_s, x_s + x_sl], [y_s, y_s + y_sl], [z_s, z_s + z_sl],
                           mutation_scale=5, lw=1, arrowstyle="-|>", color=color, visible = self.show_master_slaves)
        
        arrow_sl = self.subplot.add_artist(arrow_sl)

        while len(self.arrows_objects) < i + 1:
            self.arrows_objects.append(None)
        if self.arrows_objects[i] is not None:
            for obj in self.arrows_objects[i]:
                obj.remove()
        self.arrows_objects[i] = (arrow_m, arrow_sl)

        self.canvas.draw_idle()
        
    def calc_xyz(self,az, elev, r=1):
        if np.abs(az)>2*np.pi:
            print('You forgot to convert to radians ',az)
        if np.abs(elev)>2*np.pi:
            print('You forgot to convert to radians ', elev)
        x=r*np.cos(elev)*np.cos(az) # for elevation angle defined from XY-plane up
        #x=r*np.sin(elev)*np.cos(az) # for elevation angle defined from Z-axis down
        y=r*np.cos(elev)*np.sin(az) # for elevation angle defined from XY-plane up
        #y=r*np.sin(elev)*np.sin(az)# for elevation angle defined from Z-axis down
        z=r*np.sin(elev)# for elevation angle defined from XY-plane up
        #z=r*np.cos(elev)# for elevation angle defined from Z-axis down
        
        #correct numerical noise
        for a in (x,y,z):
            if np.allclose(a,0): a*=0
        
        return x,y,z
    
    #@pyqtSlot(int, int, tuple, int)
    def add_chan_dof(self, chan, node, az, elev, chan_name, i):
        '''
        draw an arrow indicating the DOF 
        arrow lengths do not scale with the total dimension of the plot
        add the channel number to the arrow
        store the two objects in a table
        remove any objects that might be in the table at the desired place
        i.e. avoid duplicate arrows/texts
        '''

        x_s, y_s, z_s = self.geometry_data.nodes[node]
        
        x_m, y_m, z_m = self.calc_xyz(az/180*np.pi, elev/180*np.pi,r=self.scale)
        
        #print(chan_name, node, x_m,y_m,z_m)
        

        # point the arrow towards the resulting direction
        arrow = Arrow3D([x_s, x_s + x_m], [y_s, y_s + y_m],
                        [z_s, z_s + z_m], mutation_scale=5, lw=1, arrowstyle="-|>", visible = self.show_chan_dofs)
        arrow = self.subplot.add_artist(arrow)

        text = self.subplot.text(x_s + x_m, y_s + y_m, z_s + z_m, chan_name, visible = self.show_chan_dofs)

        while len(self.channels_objects) < i + 1:
            self.channels_objects.append(None)
        if self.channels_objects[i] is not None:
            for obj in self.channels_objects[i]:
                obj.remove()
        self.channels_objects[i] = (arrow, text)

        self.canvas.draw_idle()

    #@pyqtSlot(float, float, float, int)
    def take_node(self, x, y, z, node):
        '''
        take a node at coordinates received by a signal
        take any objects connected to this node first (there should not be any)
        remove the patch objects from the plot
        remove the coordinates from the node and displacement tables
        '''
    
        d_x, d_y, d_z = self.disp_nodes.get(node, [0,0,0])
        d_x, d_y, d_z = abs(d_x), abs(d_y), abs(d_z)

        for j in [node] + list(range(max(len(self.patches_objects), node))):
            if self.patches_objects.get(j) is None:
                continue
            # ._offsets3d = ([x],[y],np.ndarray([z]))
            x_, y_, z_ = [float(val[0])
                          for val in self.patches_objects[j][0]._offsets3d]
            if   x - d_x <= x_ <= x + d_x and \
                    y - d_y <= y_ <= y + d_y and \
                    z - d_z <= z_ <= z + d_z:
                for obj in self.patches_objects[j]:
                    obj.remove()
                del self.patches_objects[j]
                break
        else:  # executed when for loop runs through
            if self.patches_objects:
                print('patches_object not found')

        for j in [node] + list(range(max(len(self.geometry_data.nodes), node))):

            if  self.geometry_data.nodes.get(j) == [x, y, z]:           
                del self.disp_nodes[j]
                break
        else:  # executed when for loop runs through
            if self.patches_objects:
                print('node not found')

        self.canvas.draw_idle()

    #@pyqtSlot(tuple)
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
        node_s, node_e = self.geometry_data.nodes[line[0]], self.geometry_data.nodes[line[1]]
        x_s, y_s, z_s = node_s
        x_e, y_e, z_e = node_e
        
        d_node_s = self.disp_nodes.get(line[0], [0,0,0])
        d_node_e = self.disp_nodes.get(line[1], [0,0,0])
        
        d_x_s, d_y_s, d_z_s = abs(d_node_s[0]),abs(d_node_s[1]),abs(d_node_s[2])
        d_x_e, d_y_e, d_z_e = abs(d_node_e[0]),abs(d_node_e[1]),abs(d_node_e[2])
        

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
                
        for j in range(len(self.nd_lines_objects)):
            (x_s_, x_e_), (y_s_, y_e_), (z_s_, z_e_) = self.nd_lines_objects[
                j]._verts3d
            if  x_s - d_x_s <= x_s_ <= x_s + d_x_s and \
                    x_e - d_x_e <= x_e_ <= x_e + d_x_e and \
                    y_s - d_y_s <= y_s_ <= y_s + d_y_s and \
                    y_e - d_y_e <= y_e_ <= y_e + d_y_e and \
                    z_s - d_z_s <= z_s_ <= z_s + d_z_s and \
                    z_e - d_z_e <= z_e_ <= z_e + d_z_e:  # account for displaced lines

                self.nd_lines_objects[j].remove()
                del self.nd_lines_objects[j]
                break
            elif x_s - d_x_s <= x_e_ <= x_s + d_x_s and \
                    x_e - d_x_e <= x_s_ <= x_e + d_x_e and \
                    y_s - d_y_s <= y_e_ <= y_s + d_y_s and \
                    y_e - d_y_e <= y_s_ <= y_e + d_y_e and \
                    z_s - d_z_s <= z_e_ <= z_s + d_z_s and \
                    z_e - d_z_e <= z_s_ <= z_e + d_z_e:  # account for inverted lines

                self.nd_lines_objects[j].remove()
                del self.nd_lines_objects[j]
                break
        else:
            if self.nd_lines_objects:
                print('line_object not found')
        self.canvas.draw_idle()

    #@pyqtSlot(int, float, float, float, int, float, float, float)
    def take_master_slave(self, i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl):
        '''
        remove the two arrows associated with the master-slave definition
        received by a signal
        remove the master-slave definition from the internal table, too
        '''
        arrow_m = (i_m, x_m, y_m, z_m)
        arrow_sl = (i_sl, x_sl, y_sl, z_sl)

        node_m = arrow_m[0]
        x_s_m, y_s_m, z_s_m = self.geometry_data.nodes[node_m]
        x_e_m, y_e_m, z_e_m = arrow_m[1:4]
        length_m = x_e_m ** 2 + y_e_m ** 2 + z_e_m ** 2

        node_sl = arrow_sl[0]
        x_s_sl, y_s_sl, z_s_sl = self.geometry_data.nodes[node_sl]
        
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

        self.canvas.draw_idle()

    #@pyqtSlot(int, int, tuple, int)
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

        x_s, y_s, z_s = self.geometry_data.nodes[node]
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


        self.canvas.draw_idle()

    def draw_axis(self):
        '''
        draw the axis arrows
        length is based on the current datalimits
        removes the current arrows if the exist
        '''
        
        for axis in ['X', 'Y', 'Z']:
            if axis in self.axis_obj:
                try:
                    self.axis_obj[axis][0].remove()
                    self.axis_obj[axis][1].remove()
                    del self.axis_obj[axis]
                except ValueError:
                    continue
        
        self.scale
        
        axis = self.subplot.add_artist(
                Arrow3D([0, self.scale], [0, 0], [0, 0], 
                        mutation_scale=20, lw=1, arrowstyle="-|>", color="r", visible= self.show_axis))
        text = self.subplot.text(
                 self.scale*1.1, 0, 0, 'X', zdir=None, color='r', visible= self.show_axis)
        self.axis_obj['X'] = (axis, text)
        

        axis = self.subplot.add_artist(
                Arrow3D([0, 0], [0, self.scale], [0, 0], 
                        mutation_scale=20, lw=1, arrowstyle="-|>", color="g", visible= self.show_axis))
        text = self.subplot.text(
                0, self.scale*1.1, 0, 'Y', zdir=None, color='g', visible= self.show_axis)
        self.axis_obj['Y'] = (axis, text)


        axis = self.subplot.add_artist(
                Arrow3D([0, 0], [0, 0], [0, self.scale], 
                        mutation_scale=20, lw=1, arrowstyle="-|>", color="b", visible= self.show_axis))
        text = self.subplot.text(
                0, 0, self.scale*1.1, 'Z', zdir=None, color='b', visible= self.show_axis)
        self.axis_obj['Z'] = (axis, text)
        
        self.canvas.draw_idle()
        
    def refresh_axis(self, visible=None):
        
        visible = bool(visible)

        if visible is not None:
            self.show_axis = visible
        
        for objs in self.axis_obj.values():
            for obj in objs:
                obj.set_visible(self.show_axis)
        self.canvas.draw_idle()
        
    #@pyqtSlot()
    def draw_nodes(self):
        ''''
        draw gridpoints from self.geometry_data.nodes
        the currently stored displacement values are used for moving the nodes
        '''
        for key, node  in self.geometry_data.nodes.items():
            self.add_node(*node, i=key)

    def refresh_nodes(self, visible=None):
        
        if visible is not None:
            visible=bool(visible)
            self.show_nodes = visible
            
        for key in self.geometry_data.nodes.keys():
            node = self.geometry_data.nodes[key]
            disp_node = self.disp_nodes.get(key, [0,0,0])
            phase_node = self.phi_nodes.get(key,[0,0,0])
            patch = self.patches_objects.get(key,None)
            if isinstance(patch, (tuple, list)):
                for obj in patch:
                    obj.set_visible(self.show_nodes)
                x = node[0]+disp_node[0]* np.cos(self.seq_num / 25 * 2 * np.pi  + phase_node[0])
                y = node[1]+disp_node[1]* np.cos(self.seq_num / 25 * 2 * np.pi  + phase_node[1])
                z = node[2]+disp_node[2]* np.cos(self.seq_num / 25 * 2 * np.pi  + phase_node[2])
                #print('in refresh nodes', x,y,z)
                #if 'PIV' in key:
                #    print(key, disp_node, phase_node)
                    
                patch[0].set_offsets([x,y])
                patch[0].set_3d_properties(z, 'z')
                
                patch[1].set_position([x,y])
                patch[1].set_3d_properties(z, None)
        
        self.canvas.draw_idle()
        
    def draw_lines(self):
        '''
        draw all the beams in self.geometry_data.lines
        self.geometry_data.lines=[line1, line2,....]
        line = [node_start, node_end]
        node numbering refers to elements in self.nodes
        '''
        for i, line in enumerate(self.geometry_data.lines):            
            self.add_line(line, i)
            self.add_nd_line(line, i)
            self.refresh_lines()
            self.refresh_nd_lines()
            
        for i in self.geometry_data.nodes.keys():
            self.add_cn_line(i)
            
        
    def refresh_lines(self, visible=None):
        
        if visible is not None:
            visible=bool(visible)
            self.show_lines = visible
            
        for line, line_node in zip(self.lines_objects, self.geometry_data.lines):
            x = [self.geometry_data.nodes[node][0] + self.disp_nodes[node][0] 
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + self.phi_nodes[node][0])
                 for node in line_node]
            y = [self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + self.phi_nodes[node][1])
                 for node in line_node]
            z = [self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + self.phi_nodes[node][2])
                 for node in line_node]
            line.set_visible(self.show_lines)     
            line.set_data([x, y])
            line.set_3d_properties(z)
            
        for key in self.geometry_data.nodes.keys():
            node = self.geometry_data.nodes[key]
            disp_node = self.disp_nodes.get(key, [0,0,0])
            phi_node = self.phi_nodes.get(key,[0,0,0])
            line = self.cn_lines_objects.get(key,None)
            if line is None: continue
            
            x = [node[0],node[0]+disp_node[0]
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + phi_node[0])]
            y = [node[1],node[1]+disp_node[1]
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + phi_node[1])]
            z = [node[2],node[2]+disp_node[2]
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + phi_node[2])]
            line.set_visible(self.show_nd_lines)     
            line.set_data([x, y])
            line.set_3d_properties(z)      
            
        self.canvas.draw_idle()    
        
    def refresh_nd_lines(self, visible=None):
        
        if visible is not None:
            visible=bool(visible)
            self.show_nd_lines = visible
            
        for line, line_node in zip(self.nd_lines_objects, self.geometry_data.lines):
            x = [self.geometry_data.nodes[node][0] 
                 for node in line_node]
            y = [self.geometry_data.nodes[node][1] 
                 for node in line_node]
            z = [self.geometry_data.nodes[node][2]
                 for node in line_node]
            line.set_visible(self.show_nd_lines)     
            line.set_data([x, y])
            line.set_3d_properties(z)
               
        for key, node in self.geometry_data.nodes.items():
            disp_node = self.disp_nodes.get(key, [0,0,0])
            phi_node = self.phi_nodes.get(key, [0,0,0])
            line = self.cn_lines_objects.get(key, None)
            if line is not None:
                x = [node[0], node[0]+disp_node[0]
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + phi_node[0])]
                y = [node[1], node[1]+disp_node[1]
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + phi_node[1])]
                z = [node[2], node[2]+disp_node[2]
                 * np.cos(self.seq_num / 25 * 2 * np.pi  + phi_node[2])]
                line.set_visible(self.show_nd_lines)     
                line.set_data([x, y])
                line.set_3d_properties(z)    
            
        self.canvas.draw_idle()
        
    def draw_master_slaves(self):
        '''
        draw arrows for all master-slave definitions stored in the
        internal master-slave definition table
        '''
        for i, (i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl) in enumerate(self.geometry_data.master_slaves):
            self.add_master_slave(i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl, i)
            
    def refresh_master_slaves(self, visible=None):
        '''
        will not be shown in displaced mode (modeshape)
        '''
        
        if visible is not None:
            visible=bool(visible)
            self.show_master_slaves = visible
            
        for patch in self.arrows_objects:
            for obj in patch:
                obj.set_visible(self.show_master_slaves)  
        self.canvas.draw_idle() 
            
    def draw_chan_dofs(self):
        '''
        draw arrows and numbers for all channel-DOF assignments stored 
        in the internal channel - DOF assignment table
        '''
        for i, chan_dof in enumerate(self.chan_dofs):
            
            chan, node, az, elev, chan_name = chan_dof[0:4]+chan_dof[-1:]
            if node is None:
                continue
            if not node in self.geometry_data.nodes.keys():
                continue
            self.add_chan_dof(chan, node, az, elev, chan_name, i)
            
    def refresh_chan_dofs(self, visible = None):
        '''
        will not be shown in displaced mode (modeshape)
        '''
        if visible is not None:
            visible=bool(visible)
            self.show_chan_dofs = visible
            
        for patch in self.channels_objects:
            for obj in patch:
                obj.set_visible(self.show_chan_dofs)
        self.canvas.draw_idle()
    
    def draw_msh(self):
        '''
        assigns displacement values to the
        nodes based on the channel - DOF assignments and the master - 
        slave definitions
        draws the displaced nodes and beams
        '''
        
        def to_phase_mag(disp):
            if self.real:
                phase = np.angle(disp, True) 
                mag = np.abs(disp)
                if phase < 0 : 
                    phase += 180
                    mag = -mag
                if phase > 90 and phase < 270:
                    mag = - mag
                phase = 0
            else:
                phase = np.angle(disp)
                mag = np.abs(disp)
            return phase, mag
        
        mode_shape = self.mode_shapes[:,self.mode_index[1], self.mode_index[0]]
        #print(mode_shape)
        mode_shape = ModalBase.rescale_mode_shape(mode_shape)
        ampli = self.amplitude

        self.disp_nodes = { i : [0,0,0] for i in self.geometry_data.nodes.keys() } 
        
        self.phi_nodes = { i : [0,0,0] for i in self.geometry_data.nodes.keys() }
        
        chan_found = [False for chan in range(len(mode_shape))]
        
        for node in self.geometry_data.nodes.keys():
            this_chan_dofs = []
            for chan_dof in self.chan_dofs:
                chan, node_, az, elev, chan_name = chan_dof[0:4]+chan_dof[-1:]
                if node_ == node:
                    disp = mode_shape[chan]
                        
                    x,y,z = self.calc_xyz(az*np.pi/180, elev*np.pi/180, r=1)# radius 1 is needed for the coordinate transformation to work
                    
                    this_chan_dofs.append([chan,x,y,z,disp])
                    
                    chan_found[chan] = True
                    
            if len(this_chan_dofs)==0: continue # no sensors in this node
            
            elif len(this_chan_dofs)==1: # only one sensor in this node
                chan,x,y,z,disp = this_chan_dofs[0]
                
                phase, mag = to_phase_mag(disp)
                        
                self.phi_nodes[node][0] = phase
                self.disp_nodes[node][0] = x * mag * ampli
                
                self.phi_nodes[node][1] = phase
                self.disp_nodes[node][1] = y * mag * ampli
                
                self.phi_nodes[node][2] = phase
                self.disp_nodes[node][2] = z * mag * ampli
                
            else: # two or more sensors in this node
                
                # check if sensors are in direction of the coordinate 
                # system or if they need to be transformed
                sum_x= 0
                sum_y= 0
                sum_z= 0
                for chan,x,y,z,disp in this_chan_dofs:
                    #print(chan,x,y,z)
                    if x!=0: sum_x += 1
                    if y!=0: sum_y += 1
                    if z!=0: sum_z += 1
                #print(sum_x, sum_y, sum_z)
                if sum_x<=1 and sum_y<=1 and sum_z<=1: # sensors are in coordinate direction
                    
                    for chan,x,y,z,disp in this_chan_dofs:
                        
                        phase, mag = to_phase_mag(disp)
                        
                        if x!=0:
                            self.phi_nodes[node][0] = phase
                            self.disp_nodes[node][0] = x * mag * ampli
                        elif y!=0:
                            self.phi_nodes[node][1] = phase
                            self.disp_nodes[node][1] = y * mag * ampli
                        elif z!=0:
                            self.phi_nodes[node][2] = phase
                            self.disp_nodes[node][2] = z * mag * ampli
                else:
                    num_sensors = max(len(this_chan_dofs),3)
                    # at least three sensors are needed for the coordinate transformation
                    # if only two sensors are present, they will be complemented by 
                    # a zero displacement assumption in perpendicular direction 
                    normal_matrix = np.zeros((num_sensors,3))
                    disp_vec = np.zeros(num_sensors, dtype=complex)
                    for i,(chan,x,y,z,disp) in enumerate(this_chan_dofs):
                        normal_matrix[i,:]=[x,y,z]
                        disp_vec[i] =disp

                    if i ==1: # only two sensors were present
                        #print('Not enough sensors for a full 3D transformation at node {}, '
                        #      'will complement vectors with a zero displacement assumption in orthogonal direction.'.format(node))
                        c=np.cross(normal_matrix[0,:],normal_matrix[1,:]) # vector c is perpendicular to the first two vectors
                        c/=np.linalg.norm(c) # if angle between first two vectors is different from 90° vector c has to be normalized
                        #print(node, c)
                        normal_matrix[2,:]=c
                    
                    
                    '''
                    ⎡ n_1,x  n_1,y  n_1,z ⎤ ⎡ q_res_x ⎤   ⎡ d_1 ⎤
                    ⎢ n_2,x  n_2,y  n_2,z ⎥ ⎢ q_res_y ⎥ = ⎢ d_2 ⎥
                    ⎣ n_3,x  n_3,y  n_3,z ⎦ ⎣ q_res_z ⎦   ⎣ d_3 ⎦
                    '''
                    # solve the well- or over-determined system of equations
                    q_res = np.linalg.lstsq(normal_matrix, disp_vec)[0]
                    
                    for i in range(3):
                        disp = q_res[i]
                        #print(disp)
                        phase, mag = to_phase_mag(disp)
                            
                        self.phi_nodes[node][i] = phase
                        self.disp_nodes[node][i] =  mag * ampli

#         for chan, disp in enumerate(mode_shape):
#             if isinstance(disp, np.complex):
#                 if self.real:
#                     phase = np.angle(disp, True) 
#                     disp = np.abs(disp)
#                     if phase < 0 : 
#                         phase += 180
#                         disp = -disp                    
#                     if phase > 90 and phase < 270:
#                         disp = - disp
#                     phase = 0
#                 else:
#                     phase = np.angle(disp)
#                     disp = np.abs(disp)
#             else:
#                 phase = 0   
#             found = False 
#             for chan_dof in self.chan_dofs:
#                 chan_, node, az, elev, chan_name = chan_dof[0:4]+chan_dof[-1:]
#                 if node is None:
#                     continue
#                 if not node in self.geometry_data.nodes.keys():
#                     continue
#                 if chan_ == chan:
#                     x,y,z = self.calc_xyz(az*np.pi/180, elev*np.pi/180)
#                     if 'PIV' in chan_name:
#                         print(chan_name,x,y,z)
#                         print(node, self.disp_nodes[node],self.phi_nodes[node])
#                     #print(chan_, chan, node, x,y,disp,np.degrees(phase))
#                     #print(x,y,z)
#                     
#                     # TODO: Account for skewed measurement angles
#                     # assumes vectors have components in one direction (x,y,z) only
#                     
#                     
#                     
#                     
#                     
#                     if not np.allclose(x,0):
#                         #print(x, phase)
#                         if self.disp_nodes[node][0] != 0:
#                             print('A modal coordinate has already been assigned to this DOF x of node {}. Overwriting!'.format(node))
#                             
#                         self.phi_nodes[node][0] = phase
#                         self.disp_nodes[node][0] = x * disp * ampli #/ norm
#                     if not np.allclose(y,0):
#                         #print(y,phase)
#                         if self.disp_nodes[node][1] != 0:
#                             print('A modal coordinate has already been assigned to this DOF y of node {}. Overwriting!'.format(node))
#                         self.phi_nodes[node][1] = phase
#                         self.disp_nodes[node][1] = y * disp * ampli #/ norm
#                     if not np.allclose(z,0):
#                         #print(z,phase)
#                         if self.disp_nodes[node][2] != 0:
#                             print('A modal coordinate has already been assigned to this DOF z of node {}. Overwriting!'.format(node))
#                         self.phi_nodes[node][2] = phase
#                         self.disp_nodes[node][2] = z * disp * ampli #/ norm
#                     
#                     found=True
            
        for chan, found in enumerate(chan_found):
            if not found:
                print('Could not find channel - DOF assignment for '
                      'channel {}!'.format(chan))
        '''
        .. TODO::
             * change master_slave assignment to skewed coordinate 
             * change master_slaves to az, elev
        '''
        
        for i_m, x_m, y_m, z_m, i_sl, x_sl, y_sl, z_sl in self.geometry_data.master_slaves:
                        
            if (x_m>0+y_m>0+z_m>0)>1:
                print('Master DOF includes more than one cartesian direction. Phase angles will be distorted.')
            
            master_disp =   self.disp_nodes[i_m][0] * x_m + \
                            self.disp_nodes[i_m][1] * y_m + \
                            self.disp_nodes[i_m][2] * z_m
                            
            master_phase =   self.phi_nodes[i_m][0] * x_m + \
                            self.phi_nodes[i_m][1] * y_m + \
                            self.phi_nodes[i_m][2] * z_m
                            
            if not np.allclose(x,0):
                #print(x, phase)
                if self.disp_nodes[i_sl][0] > 0:
                    print('A modal coordinate has already been assigned to this DOF x of node {}. Overwriting!'.format(node))
                self.phi_nodes[i_sl][0] = phase
                self.disp_nodes[i_sl][0] += master_disp * x_sl
            if not np.allclose(y,0):
                #print(y,phase)
                if self.disp_nodes[i_sl][1] > 0:
                    print('A modal coordinate has already been assigned to this DOF y of node {}. Overwriting!'.format(node))
                self.phi_nodes[i_sl][1] = phase
                self.disp_nodes[i_sl][1] += master_disp * y_sl
            if not np.allclose(z,0):
                #print(z,phase)
                if self.disp_nodes[i_sl][2] > 0:
                    print('A modal coordinate has already been assigned to this DOF z of node {}. Overwriting!'.format(node))
                self.phi_nodes[i_sl][2] = phase
                self.disp_nodes[i_sl][2] += master_disp * z_sl

#             if self.draw_trace:
#                 if self.trace_objects:
#                     for i in range(len(self.trace_objects)-1,-1,-1):
#                         try:
#                             self.trace_objects[i].remove()
#                         except Exception as e:
#                             pass
#                             #print("Error",e)
#                             
#                         del self.trace_objects[i]
#                         
#                 moving_nodes = set()
#                 for chan_dof in self.chan_dofs:#
#                     chan_, node, az, elev, chan_name  = chan_dof[0:4]+ [chan_dof[-1]]
#                     if node is None:
#                         continue
#                     if not node in self.geometry_data.nodes.keys():
#                         continue
#                     moving_nodes.add(node)
#                 
#                 clist = itertools.cycle(list(matplotlib.cm.jet(np.linspace(0, 1, len(moving_nodes)))))#@UndefinedVariable
#                 for node in moving_nodes:
#                     self.trace_objects.append(self.subplot.plot(xs=self.geometry_data.nodes[node][0] + self.disp_nodes[node][0]
#                          * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][0]),
#                          ys=self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
#                          * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][1]),
#                          zs=self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
#                          * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][2]), 
#                          #marker = ',', s=1, edgecolor='none', 
#                          color = next(clist)))
        #print(self.disp_nodes, self.phi_nodes)       
        self.refresh_nodes()
        self.refresh_lines()
        self.refresh_chan_dofs(False)
        self.refresh_master_slaves(False)
        if self.animated:
            self.stop_ani()
            self.animate()
        self.canvas.draw()

    #@pyqtSlot()
    def stop_ani(self):
        '''
        convenience method to stop the animation and restore the still plot
        '''
        if self.animated or self.data_animated:
            self.seq_num = next(self.line_ani.frame_seq)
            self.line_ani._stop()
            self.animated = False
            self.data_animated = False
            for c in self.connect_handles:
                self.canvas.mpl_disconnect(c)
            self.draw_nodes()
            self.refresh_nodes()
            self.draw_lines()
            self.refresh_lines()
            self.refresh_nd_lines()
            #self.draw_msh()
            
            
    #@pyqtSlot()
    def animate(self):
        '''
        create necessary objects to animate the currently displayed
        deformed structure
        '''
        
        self.save_ani = False
        '''
        !!!
        
        The NUMBERING of the FILES follows the order in which the modes were selected in the stabilization diagramm
        
        !!!
        '''
        if self.save_ani:
            self.cwd = '/vegas/users/staff/womo1998/Projects/2019_Schwabach/tex/figures/ani_high/'#os.getcwd()
            #for i in range(len(self.select_modes)):
            #    os.makedirs(os.path.join(self.cwd,str(i)), exist_ok=True)
                
        self.draw_trace=True
        def init_lines():
            #print('init')
            #self.clear_plot()
            minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()
            
            self.subplot.cla()
            self.subplot.grid(False)
            self.subplot.set_axis_off()
            self.canvas.draw()
            #return self.lines_objects
            self.draw_lines()
            self.draw_axis()
            for line in self.lines_objects:
                line.set_visible(False)
            for line in self.nd_lines_objects:
                line.set_visible(False)
            for line in self.cn_lines_objects.values():
                line.set_visible(False)
            
            self.subplot.set_xlim3d(minx,maxx)
            self.subplot.set_ylim3d(miny, maxy)
            self.subplot.set_zlim3d(minz, maxz)
            
#             this_dirs={}
#             
#             for node in ['1','2','3','4','5','6','7']:
#                 this_chans, this_az = [],[]
#                 for chan, node_, az,elev,header in self.chan_dofs:
#                     if node == node_:
#                         this_chans.append(chan)
#                         this_az.append(az)
#                 if len(this_chans) != 2:
#                     continue    
#                 
#                 this_dirs[node]={}    
#                 
#                 x,y=[],[]
#                 for t in np.linspace(-np.pi,np.pi,359):
#                     x.append(0)
#                     y.append(0)
#                     for j,az in enumerate(this_az):
#                         x_,y_= self.calc_xy(np.radians(az))
#                         x[-1]+= np.abs(msh[j])*x_* np.cos(t  + np.angle(msh[j]))
#                         y[-1]+= np.abs(msh[j])*y_* np.cos(t  + np.angle(msh[j]))
#                 #plot.figure(figsize=(8,8))
#                 if i == 1 and k==0:
#                     ind = ['1','4','5','6','3','2'].index(node)
#                     import matplotlib.cm
#                     color=list(matplotlib.cm.hsv(np.linspace(0, 1, 7)))[ind]
#                     plot.plot(x,y, label=['108','126','145','160','188','TMD'][ind], color=color)


            if self.draw_trace:
                if self.trace_objects:
                    for i in range(len(self.trace_objects)-1,-1,-1):
                        try:
                            self.trace_objects[i].remove()
                        except:
                            pass
                            
                        del self.trace_objects[i]
                        
                moving_nodes = set()
                for chan_dof in self.chan_dofs:#
                    chan_, node, az, elev, = chan_dof[0:4]
                    if node is None:
                        continue
                    if not node in self.geometry_data.nodes.keys():
                        continue
                    moving_nodes.add(node)
                
                #clist = itertools.cycle(list(matplotlib.cm.jet(np.linspace(0, 1, len(moving_nodes)))))#@UndefinedVariable
                clist = itertools.cycle(['darkgray' for i in range(len(moving_nodes))])
                for node in moving_nodes:
                    self.trace_objects.append(self.subplot.plot(xs=self.geometry_data.nodes[node][0] + self.disp_nodes[node][0]
                         * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][0]),
                         ys=self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
                         * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][1]),
                         zs=self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
                         * np.cos(np.linspace(0,359,360) / 360 * 2 * np.pi  + self.phi_nodes[node][2]), 
                         #marker = ',', s=1, edgecolor='none', 
                         color = next(clist), linewidth = self.linewidth, linestyle=(0,(1,1))))
                
            return self.lines_objects + \
            self.nd_lines_objects + \
            list(self.cn_lines_objects.values())
            #return self.lines_objects#, self.nd_lines_objects
        
        def update_lines(num):
            '''
            subfunction to calculate displacements based on magnitude and phase angle
            '''
            #print(num)
            
#             if not self.traced: clist = itertools.cycle(matplotlib.rcParams['axes.color_cycle'])
            
            for i,(line, line_node) in enumerate(zip(self.lines_objects, self.geometry_data.lines)):
                x = [self.geometry_data.nodes[node][0] + self.disp_nodes[node][0]
                     * np.cos(num / 25 * 2 * np.pi  + self.phi_nodes[node][0]) 
                     for node in line_node]
                y = [self.geometry_data.nodes[node][1] + self.disp_nodes[node][1]
                     * np.cos(num / 25 * 2 * np.pi  + self.phi_nodes[node][1]) 
                     for node in line_node]
                z = [self.geometry_data.nodes[node][2] + self.disp_nodes[node][2]
                     * np.cos(num / 25 * 2 * np.pi  + self.phi_nodes[node][2]) 
                     for node in line_node]

                # NOTE: there is no .set_data() for 3 dim data...
                line.set_visible(self.show_lines)
                line.set_data([x, y])
                if isinstance(self.beamcolor, (list, tuple, np.ndarray)):
                    beamcolor = self.beamcolor[i]
                else:
                    beamcolor = self.beamcolor
                if isinstance(self.beamstyle, (list, tuple, np.ndarray)):
                    beamstyle = self.beamstyle[i]
                else:
                    beamstyle = self.beamstyle
                line.set_color(beamcolor)
                line.set_linestyle(beamstyle)
                line.set_3d_properties(z)
                
            for line in self.nd_lines_objects:
                line.set_visible(self.show_nd_lines)
                
            for objs in self.axis_obj.values():
                for obj in objs:
                    obj.set_visible(self.show_axis)
                    
            if self.save_ani:        
                self.fig.savefig(self.cwd + '/{}/ani_{}.pdf'.format(self.select_modes.index(self.mode_index), num))
                print(self.cwd + '/{}/ani_{}.pdf'.format(self.select_modes.index(self.mode_index), num))
            return self.lines_objects + \
            self.nd_lines_objects + \
            list(self.cn_lines_objects.values())
        
        
        #self.cla()
        #self.patches_objects = {}
        self.lines_objects = []
        self.nd_lines_objects = []
        self.cn_lines_objects = {}
        self.arrows_objects = []
        self.channels_objects = []
        self.axis_obj = {}
        
        if self.animated:
            return self.stop_ani()
        else:
            if self.data_animated:
                self.stop_ani()
            self.animated = True        
        
        c1 = self.canvas.mpl_connect('motion_notify_event', self._on_move)
        c2 = self.canvas.mpl_connect('button_press_event', self._button_press)
        c3 = self.canvas.mpl_connect('button_release_event', self._button_release)
        self.connect_handles=[c1,c2,c3]
        self.button_pressed = None
        
        
        
        self.line_ani = matplotlib.animation.FuncAnimation(fig=self.fig,
                                      func=update_lines,
                                      init_func=init_lines,
                                      interval=50,
                                      save_count=50,
                                      blit=True)
        #for i in range(self.seq_num+1): next(self.line_ani.frame_seq)
        
        #print(self.fig.get_size_inches())
        # Set up formatting for the movie files
        #Writer = matplotlib.animation.writers['ffmpeg']
        #writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
        #self.line_ani.save('lines.mp4', writer=writer)
        #print(self.fig.get_size_inches())
        self.canvas.draw()
        
        #self.line_ani._start()
        #print(self.animated, self.line_ani)
        
    #@pyqtSlot()
    def filter_and_animate_data(self, callback=None):
        '''
        create necessary objects to animate the currently displayed
        deformed structure
        '''
        def init_lines():
            #print('init')
            #self.clear_plot()
            minx, maxx, miny, maxy, minz, maxz = self.subplot.get_w_lims()
            
            self.subplot.cla()
            #return self.lines_objects
            self.draw_lines()
            for line in self.lines_objects:
                line.set_visible(False)
            for line in self.nd_lines_objects:
                line.set_visible(False)
            for line in self.cn_lines_objects.values():
                line.set_visible(False)
            
            self.subplot.set_xlim3d(minx,maxx)
            self.subplot.set_ylim3d(miny, maxy)
            self.subplot.set_zlim3d(minz, maxz)
            
            return self.lines_objects + \
            self.nd_lines_objects + \
            list(self.cn_lines_objects.values())
            #return self.lines_objects#, self.nd_lines_objects
        def update_lines(num):
            '''
            subfunction to calculate displacements based on magnitude and phase angle
            '''
            self.callback(f'{num/self.prep_data.sampling_rate:.4f}')
            disp_nodes={ i : [0,0,0] for i in self.geometry_data.nodes.keys() } 
            for chan_dof in self.chan_dofs:
                chan_, node, az, elev, = chan_dof[0:4]
                
                if node is None:
                    continue
                if not node in self.geometry_data.nodes.keys():
                    continue
                x,y,z = self.calc_xyz(az*np.pi/180, elev*np.pi/180)
                disp_nodes[node][0]+=self.prep_data.measurement_filt[num,chan_]*x*self.amplitude
                disp_nodes[node][1]+=self.prep_data.measurement_filt[num,chan_]*y*self.amplitude
                disp_nodes[node][2]+=self.prep_data.measurement_filt[num,chan_]*z*self.amplitude
                
            #print(num)
            for line, line_node in zip(self.lines_objects, self.geometry_data.lines):
                x = [self.geometry_data.nodes[node][0] + disp_nodes[node][0]
                     for node in line_node]
                y = [self.geometry_data.nodes[node][1] + disp_nodes[node][1]
                     for node in line_node]
                z = [self.geometry_data.nodes[node][2] + disp_nodes[node][2]
                     for node in line_node]
                # NOTE: there is no .set_data() for 3 dim data...
                line.set_visible(self.show_lines)
                line.set_data([x, y])
                line.set_color('b')
                line.set_3d_properties(z)
                
            for line in self.nd_lines_objects:
                line.set_visible(self.show_nd_lines)
                
            for key in self.geometry_data.nodes.keys():
                node = self.geometry_data.nodes[key]
                disp_node = disp_nodes.get(key,[0,0,0])
                x = [node[0],node[0] + disp_node[0]]
                y = [node[1],node[1] + disp_node[1]]
                z = [node[2],node[2] + disp_node[2]]
                line = self.cn_lines_objects.get(key,None)
                if line is not None:
                    line.set_data([x, y])
                    line.set_visible(self.show_nd_lines)
                    line.set_3d_properties(z)
                    
            return self.lines_objects + \
            self.nd_lines_objects + \
            list(self.cn_lines_objects.values())        

        #self.cla()
        #self.patches_objects = {}
        self.lines_objects = []
        self.nd_lines_objects = []
        self.cn_lines_objects = {}
        self.arrows_objects = []
        self.channels_objects = []
        self.axis_obj = {}
        
        if self.data_animated:
            return self.stop_ani()
        else:
            if self.animated:
                self.stop_ani()
            self.data_animated = True
        
        
        c1 = self.canvas.mpl_connect('motion_notify_event', self._on_move)
        c2 = self.canvas.mpl_connect('button_press_event', self._button_press)
        c3 = self.canvas.mpl_connect('button_release_event', self._button_release)
        self.connect_handles=[c1,c2,c3]
        self.button_pressed = None
        
        #self.prep_data.filter_data(lowpass, highpass)
        if callback is not None:
            self.callback = callback
        self.line_ani = matplotlib.animation.FuncAnimation(fig=self.fig,
                                      func=update_lines,
                                      frames=range(self.prep_data.measurement_filt.shape[0]),
                                      init_func=init_lines,
                                      interval=1/self.prep_data.sampling_rate,
                                      save_count=0,
                                      blit=True)
        
        self.canvas.draw()
        
        #self.line_ani._start()
        #print(self.animated, self.line_ani)        
    def _button_press(self, event):
        if event.inaxes == self.subplot:
            self.button_pressed = event.button

    def _button_release(self, event):
        self.button_pressed = None
        
    def _on_move(self, event):

        if not self.button_pressed:
            return
        
        for line in self.lines_objects:
            line.set_visible(False)
        for line in self.nd_lines_objects:
            line.set_visible(False)
        for line in self.cn_lines_objects.values():
            line.set_visible(False)
        #self.canvas.draw()
        self.line_ani._setup_blit()
        #self.line_ani._start()   
        

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


if __name__ == "__main__":
    pass
