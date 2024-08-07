'''pyOMA - A toolbox for Operational Modal Analysis
Copyright (C) 2015 - 2021  Simon Marwitz, Volkmar Zabel, Andrei Udrea et al.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


General documentation of the core package
'''
import logging
import sys
logging.basicConfig(stream=sys.stdout)

from . import PlotMSH
from . import PLSCF
from . import PostProcessingTools
from . import PRCE
from . import SSICovRef
from . import SSIData
from . import StabilDiagram
from . import VarSSIRef
from . import PreProcessingTools