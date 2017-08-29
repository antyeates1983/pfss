"""
    Routines for writing netcdf files.
    
    Copyright (C) Anthony R. Yeates, Durham University 25/8/17

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

from scipy.io import netcdf
import numpy as n

def a(filename, r, th, ph, apr, aps, app):
    """
        Vector potential * edge lengths on cell edges.
    """

    nr = n.size(r) - 1
    ns = n.size(th) - 1
    np = n.size(ph) - 1

    fid = netcdf.netcdf_file(filename, 'w')
    fid.createDimension('rc', nr)
    fid.createDimension('r', nr+1)
    fid.createDimension('thc', ns)
    fid.createDimension('th', ns+1)   
    fid.createDimension('phc', np)
    fid.createDimension('ph', np+1)      
    vid = fid.createVariable('r', 'd', ('r',))
    vid[:] = r
    vid = fid.createVariable('th', 'd', ('th',))
    vid[:] = th
    vid = fid.createVariable('ph', 'd', ('ph',))
    vid[:] = ph  
    vid = fid.createVariable('ar', 'd', ('ph','th','rc'))
    vid[:] = apr
    vid = fid.createVariable('as', 'd', ('ph','thc','r'))
    vid[:] = aps    
    vid = fid.createVariable('ap', 'd', ('phc','th','r'))
    vid[:] = app     
    fid.close()
    print('Wrote A*L to file '+filename)
    
    
def bc(filename, r, th, ph, rc, thc, phc, br, bs, bp):
    """
        Magnetic field components on cell faces, including ghost cells.
    """
    
    nr = n.size(r) - 1
    ns = n.size(th) - 1
    np = n.size(ph) - 1
    
    fid = netcdf.netcdf_file(filename, 'w')
    fid.createDimension('rc', nr+2)
    fid.createDimension('r', nr+1)
    fid.createDimension('thc', ns+2)
    fid.createDimension('th', ns+1)   
    fid.createDimension('phc', np+2)
    fid.createDimension('ph', np+1)      
    vid = fid.createVariable('r', 'd', ('r',))
    vid[:] = r
    vid = fid.createVariable('th', 'd', ('th',))
    vid[:] = th
    vid = fid.createVariable('ph', 'd', ('ph',))
    vid[:] = ph
    vid = fid.createVariable('rc', 'd', ('rc',))
    vid[:] = rc
    vid = fid.createVariable('thc', 'd', ('thc',))
    vid[:] = thc
    vid = fid.createVariable('phc', 'd', ('phc',))
    vid[:] = phc       
    vid = fid.createVariable('br', 'd', ('phc','thc','r'))
    vid[:] = br
    vid = fid.createVariable('bth', 'd', ('phc','th','rc'))
    vid[:] = -bs   
    vid = fid.createVariable('bph', 'd', ('ph','thc','rc'))
    vid[:] = bp     
    fid.close()
    print('Wrote B on faces to file '+filename)
    
    
def bg(filename, r, th, ph, brg, bsg, bpg):
    """
        Magnetic field components co-located at grid points.
    """
    
    nr = n.size(r) - 1
    ns = n.size(th) - 1
    np = n.size(ph) - 1   
    
    fid = netcdf.netcdf_file(filename, 'w')
    fid.createDimension('r', nr+1)
    fid.createDimension('th', ns+1)   
    fid.createDimension('ph', np+1)      
    vid = fid.createVariable('r', 'd', ('r',))
    vid[:] = r
    vid = fid.createVariable('th', 'd', ('th',))
    vid[:] = th
    vid = fid.createVariable('ph', 'd', ('ph',))
    vid[:] = ph     
    vid = fid.createVariable('br', 'd', ('ph','th','r'))
    vid[:] = brg
    vid = fid.createVariable('bth', 'd', ('ph','th','r'))
    vid[:] = -bsg   
    vid = fid.createVariable('bph', 'd', ('ph','th','r'))
    vid[:] = bpg    
    fid.close()
    print('Wrote B at grid points to file '+filename)
    
