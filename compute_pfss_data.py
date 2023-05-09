"""
    Script for reading in map of Br(theta, phi) on the solar surface, computing a
    PFSS extrapolation, and outputting to a netcdf file.
    
    Copyright (C) Anthony R. Yeates, Durham University 29/8/17

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pfss import pfss
from sunpy.coordinates.sun import carrington_rotation_number
from data import readcrmap_hmi, readcrmap_gong
import datetime

# DEFINE GRID FOR PFSS COMPUTATION
# - equally spaced in rho=log(r/Rsun), s=cos(theta) and phi.
# - specify number of grid points:
nr = 60
ns = 180
nph = 360
if ((nph%2)!=0):
    sys.exit('ERROR: np must be even (for polar boundary conditions)')
# - choose source surface radius r (in Rsun):
rss = 2.5

# TIME REQUIRED
date = datetime.datetime(2014, 11, 10)

# READ MAP OF br(theta,phi) FROM SYNOPTIC DATA:
cr_snap = int(carrington_rotation_number(date))
# - to use HMI data:
# br0 = readcrmap_hmi(cr_snap, ns, nph, smooth=0)
# - to use GONG data:
br0 = readcrmap_gong(cr_snap, ns, nph, smooth=0)

# PLOT INPUT MAP:
# - threshold for colour scales:
bmax = 10

plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax = plt.subplot(111)
ds = 2/ns
dph = 2*np.pi/nph
s0 = np.linspace(-1+0.5*ds, 1-0.5*ds,ns)
ph0 = np.linspace(0.5*dph, 2*np.pi-0.5*dph, nph)
lat0 = 0.5*np.pi - np.arccos(s0)
pm = ax.pcolormesh(np.rad2deg(ph0), np.rad2deg(lat0), br0, cmap='bwr', vmin=-bmax, vmax=bmax)
cb1 = plt.colorbar(pm)
ax.set_xlabel('Carrington longitude')
ax.set_ylabel('Latitude')
ax.set_title('Magnetogram')

plt.show()

# COMPUTE POTENTIAL FIELD:
# (note that output='bg' gives the output B averaged to grid points; output='bc' would give B components on the staggered grid)
print('Computing PFSS...')
pfss(br0, nr, ns, nph, rss, filename='./b_pfss'+date.strftime('%Y%m%d.%H')+'.nc', output='bg', testQ=False)
