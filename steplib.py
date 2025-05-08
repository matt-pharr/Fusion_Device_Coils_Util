"""
Library of basic functionality for extracting coil data
from STPs in python. Currently tested on coils from SPARC and ITER. 

Note: If you wish to use this library for your own coils, you may
need to tweak some of these algorithms! STEP files have many different
types of objects and the extraction algorithm is capable of dealing with
two of them. This code can currently not deal with any coils that have
jacketing or other features that are not interpolated wires or edges. 
It also cannot yet deal with thick coils (i.e. coils with a cross section
that is not a single point). This may be added in the future, but for now
it is workable with centerline traces of coils.

Author: Matthew Pharr
Date: 25 August 2023
matthew.pharr@columbia.edu

Copyright (c) 2023, Matthew Pharr.
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree. 
"""



# from multiprocessing import Value
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
# from steputils import p21
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Extend.DataExchange import read_step_file
#from OCC.Core.gp import gp_Pnt

from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods, TopoDS_Iterator#, TopoDS_Wire
#from OCC.Core.TopExp import TopExp_Explorer
# %matplotlib qt

debug_level = 0;

# Function to sample points along the curve
def sample_curve_points(curve, u_start, u_end, num_samples):
    """Sample points along a curve"""
    usample = np.linspace(u_start,u_end,int(num_samples))
    # du = (u_end-u_start)/100
    xyzwire = []
    for ui in usample:
        xyzi = np.asarray(curve.Value(ui).Coord())/1000
        xyzwire.append(xyzi)
    return np.asarray(xyzwire)

# Function to get the arc length of a curve
def get_arc_length(curve, u_start, u_end, num_samples=100):
    """Get the arc length of a curve"""
    arclength = np.sum(np.linalg.norm(np.diff(sample_curve_points(curve, u_start, u_end, int(num_samples)),axis=0),axis=1))
    return arclength

# Function to test if the new segment is connected to the previous segment
def test_connection(xyz, xyzadd, tolerance):
    """
    Test if the new segment is connected to the previous segment
    xyz: Established coil points.
    xyzadd: Coil segment to test.
    tolerance: Tolerance for testing segments. Given in meters.
    """
    tail_to_tail_dist = euc_dist(xyz[-1],xyzadd[-1])
    tail_to_head_dist = euc_dist(xyz[-1],xyzadd[0])
    if (
        tail_to_tail_dist < tolerance
        and not 
        tail_to_head_dist < tolerance
    ):
        if debug_level > 0:
            print("reversing segment: found backwards")
            print("tail to tail dist:",tail_to_tail_dist)
            print("tail to head dist:",tail_to_head_dist)
        isvalid = False
    elif (
        tail_to_tail_dist < tolerance
        and 
        tail_to_head_dist < tolerance
    ):
        # print("Error: segments are too close. Change tolerance.")
        # raise ValueError
        isvalid = False
    elif (
        not tail_to_tail_dist < tolerance
        and 
        tail_to_head_dist < tolerance
    ):
        if debug_level > 1:
            print("found forward")
            print("tail to tail dist:",tail_to_tail_dist)
            print("tail to head dist:",tail_to_head_dist)
        isvalid = True
    elif len(xyzadd) == 1:
        if tail_to_tail_dist < tolerance:
            isvalid = True
    else:
        # print("Error: segments are too far apart. Change tolerance.")
        # print("xyz[-1]:",xyz[-1])
        # print("xyzadd[-1]:",xyzadd[-1])
        # print("xyz[0]:",xyzadd[0])
        # print("tail to head dist:",tail_to_head_dist)
        # raise ValueError
        isvalid = False
    return isvalid  

def euc_dist(xyz1, xyz2):
    """Euclidean distance between two points"""
    return np.linalg.norm(xyz1-xyz2)

def wire_connecter(splices:list, tolerance, pointspermeter):
    """Find and join connected wires"""
    
    starts = np.array([splices[i][0] for i in range(len(splices))])
    ends = np.array([splices[i][-1] for i in range(len(splices))])

    head_to_tail = np.linalg.norm(starts[:,None,:] - ends[None,:,:], axis=2) + 3*np.eye(len(splices))
    tail_to_tail = np.linalg.norm(ends[:,None,:] - ends[None,:,:], axis=2) + 3*np.eye(len(splices))
    head_to_head = np.linalg.norm(starts[:,None,:] - starts[None,:,:], axis=2) + 3*np.eye(len(splices))
    tail_to_head = np.linalg.norm(ends[:,None,:] - starts[None,:,:], axis=2) + 3*np.eye(len(splices))
    noconnect = []
    for row in range(len(starts)):
        rowsum1,rowsum2,rowsum3,rowsum4 = np.sum(head_to_head[row,:] < tolerance),np.sum(tail_to_tail[row,:] < tolerance),np.sum(head_to_tail[row,:] < tolerance), np.sum(tail_to_head[row,:] < tolerance)
        if rowsum1 > 1:
            raise ValueError(f'row {row} has {rowsum1} connections in head_to_head. Try a smaller tolerance.')
        if rowsum2 > 1:
            raise ValueError(f'splice {row} has {rowsum2} connections in tail_to_tail. Try a smaller tolerance.')
        if rowsum3 > 1:
            raise ValueError(f'splice {row} has {rowsum3} connections in head_to_tail. Try a smaller tolerance.')
        if rowsum4 > 1:
            raise ValueError(f'splice {row} has {rowsum4} connections in tail_to_head. Try a smaller tolerance.')
        if rowsum1 + rowsum2 + rowsum3 + rowsum4 == 0:
            raise Warning(f'splice {row} has no connections. Try a larger tolerance.')
            noconnect.append(row)
    if len(noconnect) > 0:
        raise Warning(f"Found {len(noconnect)} splices with no connections.")
    
    connections = set()
    connections_dict = {}
    for i in range(len(splices)):
        htts = np.where(head_to_tail[i] < tolerance)[0]
        hths = np.where(head_to_head[i] < tolerance)[0]
        ttts = np.where(tail_to_tail[i] < tolerance)[0]
        for j in htts:
            connections.add(((i,0),(j,1)))
        for j in hths:
            connections.add(((i,0),(j,0)))
        for j in ttts:
            connections.add(((i,1),(j,1)))
    
    for c in connections:
        connections_dict[c[0]] = c[1]
        connections_dict[c[1]] = c[0]
    
    # Order all the splices based on the connections
    splicenums = list(range(len(splices)))
    coils = []
    while len(splicenums) > 0:
        # list of tuples (index, indexdirection: -1 or 1)
        coil = []
        coilheaddone = False
        coiltaildone = False
        coil.append((splicenums.pop(0), 1))
        nextend = (coil[0][0], 1)
        while not coiltaildone:
            try:
                nextitem = connections_dict[nextend]
            except KeyError:
                coiltaildone = True
                continue

            if nextitem[1] == 0:
                coil.append((nextitem[0],1))
                nextend = (nextitem[0], 1)
            elif nextitem[1] == 1:
                coil.append((nextitem[0],-1))
                nextend = (nextitem[0], 0)
            else:
                print('Error: next item not 0 or 1')
                break
            
            splicenums.remove(nextitem[0])
        
        nextend = (coil[0][0], 0)
        while not coilheaddone:
            try:
                nextitem = connections_dict[nextend]
            except KeyError:
                coilheaddone = True
                continue

            if nextitem[1] == 1:
                coil.insert(0, (nextitem[0],1))
                nextend = (nextitem[0], 0)
            elif nextitem[1] == 0:
                coil.insert(0, (nextitem[0],-1))
                nextend = (nextitem[0], 1)
            else:
                print('Error: next item not 0 or 1')
                break

            splicenums.remove(nextitem[0])

        coils.append(coil)
        if debug_level > 0:
            print(f"Found coil with {len(coil)} splices.")
        
    # Make the actual point clouds

    xyz = []
    for coil in coils:
        newcoil = []
        for i in range(len(coil)):
            newcoil.extend(splices[coil[i][0]][::coil[i][1]])

        coillen = np.sum(np.linalg.norm(np.diff(newcoil, axis=0), axis=1))
        lengthfunc = np.cumsum(np.linalg.norm(np.diff(newcoil, axis=0, prepend=0), axis=1))/coillen
        mask = np.diff(lengthfunc, prepend=-0.1) > 0
        lengthfuncinterp = lengthfunc[mask]-lengthfunc[mask][0]
        # print(len(lengthfunc), len(newcoil))
        coilinterp = CubicSpline(x=lengthfuncinterp, y=np.asarray(newcoil)[mask], axis=0)
        numpoints = int(coillen*pointspermeter)

        newcoil = coilinterp(np.linspace(0, 1, numpoints))
        newcoil = np.asarray(newcoil)

        xyz.append(np.asarray(newcoil))

    return xyz

# Function to extract a coil from a wire
def coil_extract(wire_iterator, pointspermeter:float, tolerance:float = -1., force:bool=False, startfunc=lambda x: False, cutoffsphere:float =-1):
    """
    Extract a coil from a wire.
    wire_iterator: TopoDS_Iterator
    pointspermeter: Fideliy with wich to extract discrete coil geometry. Given in points per meter.
    tolerance: Tolerance for connecting segments. Given in meters.
    force: Force the coil to be extracted even if segments are not connected.
    startfunc: Function to determine if the coil should be reversed.
    cutoffsphere: Radius outside of which to exclude points. Given in meters.
    """
    # Set default values

    # Deal with tolerance unspecified
    if tolerance < 0:
        tolerance = 2/pointspermeter
        if debug_level > 0:
            print("Tolerance unspecified or negative. Defaulting to double spacing.")
    # Deal with cutoffsphere unspecified
    if cutoffsphere < 0:
        cutoffsphere = 1e6 # Set to a large number to include whole coil


    b = False
    try:
        b = wire_iterator.More()
    except Exception as e:
        print("Wire_iterator is not a TopoDS_Iterator. Trying edges list:")
        pass
    if b:
        xyz = []
        j = 0
        while wire_iterator.More():
            if debug_level > 1:
                print("segment:",j)
            edge = topods.Edge(wire_iterator.Value())
            curve, u_start, u_end = BRep_Tool().Curve(edge)
            
            # Set even spacing of points between coils
            arclength = get_arc_length(curve, u_start, u_end)
            numpoints = int(arclength*pointspermeter)

            if debug_level > 1:
                print(numpoints)
            # Rasterize points on the curve
            xyzadd = sample_curve_points(curve, u_start, u_end, numpoints)
            # Check if the new segment is connected to the previous segment and append
            if len(xyz) == 0:
                xyz.extend(xyzadd)
                pass
            else:
                print(xyz)

                isvalid = test_connection(xyz, xyzadd, tolerance)
                if isvalid:
                    xyz.extend(xyzadd)
                    
                else:
                    newisvalid = test_connection(xyz, xyzadd[::-1], tolerance)
                    if newisvalid:
                        xyz.extend(xyzadd[::-1])
                        
                    elif j == 1:
                        finalisvalid_f = test_connection(xyz[::-1],xyzadd,tolerance)
                        finalisvalid_b = test_connection(xyz[::-1],xyzadd[::-1],tolerance)
                        if finalisvalid_f:
                            xyz = xyz[::-1]
                            xyz.extend(xyzadd)
                            
                        elif finalisvalid_b:
                            xyz = xyz[::-1]
                            xyz.extend(xyzadd[::-1])
                        else:
                            pass
                    elif force:
                        xyz.extend(xyzadd)
                        
                    else:
                        print(f"Error: segments are discontinuous between segments {j} and {j-1}. Change tolerance.")
                        raise ValueError
                    
            j+=1
            wire_iterator.Next()
        xyz = np.asarray(xyz)

        # Check if the coil should be reversed
        if startfunc(xyz):
            xyz = xyz[::-1]

        # Remove points outside of the cutoff sphere
        dist = np.linalg.norm(xyz, axis=1)
        mask = dist <= cutoffsphere
        xyz = xyz[mask]

    else:
        # If wire_iterator is not a TopoDS_Iterator, assume it is a list of edges
        splices = []

        for j, edge in enumerate(list(wire_iterator)):
            if debug_level > 1:
                print("segment:",j)
            curve, u_start, u_end = BRep_Tool().Curve(edge)

            # Set even spacing of points between coils
            arclength = get_arc_length(curve, u_start, u_end)
            if arclength > 1:
                arclengthold = arclength
                arclength = get_arc_length(curve, u_start, u_end, 100*arclength)
            if arclength * pointspermeter < 15:
                numpoints = 15
            else:
                numpoints = int(arclength*pointspermeter)
            if debug_level > 1:
                print(numpoints)
            # Rasterize points on the curve
            xyzwire = sample_curve_points(curve, u_start, u_end, numpoints)
            if len(xyzwire) == 0:
                continue
            # Remove points outside of the cutoff sphere
            # print(xyzwire.shape)
            if cutoffsphere != -1:
                dist = np.linalg.norm(xyzwire, axis=1)
                mask = dist <= cutoffsphere
                xyzwire = xyzwire[mask]

            splices.append(np.asarray(xyzwire))
        
        # Clean up duplicate splices if present
        starts = [splices[i][0] for i in range(len(splices))]
        ends = [splices[i][-1] for i in range(len(splices))]

        same = []
        
        for i in range(len(splices)):
            for j in range(i+1, len(splices)):
                if np.linalg.norm(starts[i]-starts[j])< 1e-4 and np.linalg.norm(ends[i]-ends[j]) < 1e-4:
                    # print(f'Splices {i} and {j} are the same... {np.linalg.norm(starts[i]-starts[j])} {np.linalg.norm(ends[i]-ends[j])}')
                    # # print(f'  {len(splices[i])} {len(splices[j])}')
                    # print()
                    same.append((i,j))
                elif np.linalg.norm(starts[i]-ends[j])< 1e-4 and np.linalg.norm(ends[i]-starts[j]) < 1e-4:
                    # print(f'Splices {i} and {j} seem the same... {np.linalg.norm(starts[i]-ends[j])} {np.linalg.norm(ends[i]-starts[j])}')
                    # print(f'  {len(splices[i])} {len(splices[j])}')
                    # print()
                    same.append((i,j))
        
        same = sorted(same, key=lambda x: max(x[0], x[1]))

        for s in same[::-1]:
            maxindex = max(s)
            # print(f'popping splice {maxindex}')
            splices.pop(maxindex)

        # Find connected wires and join them
        xyz = wire_connecter(splices, tolerance, pointspermeter)
        
        # Check if the coil should be reversed
        for i in range(len(xyz)):
            if startfunc(xyz[i]):
                xyz[i] = xyz[i][::-1]
    
    return xyz

def coil_read(sourcefiles:list[str] = [''],
              pointspermeter:float = 50,
               tolerance:float = -1., 
               force:bool=False,
               startfunc=lambda x: False, 
               cutoffsphere:float = 1e6) -> list[np.ndarray]:
    """
    Read in a coil from a STEP file.
    sourcefiles: list of strings containing the path to the STEP file.
    """
    coils = []
    for sourcefile in sourcefiles:
        compound = read_step_file(sourcefile)
        t = TopologyExplorer(compound) # type: ignore
        try:
            cenum = t.wires()
            wires = True
        except:
            wires = False
        if wires:
            # Initial sample t measure the length of 
            # the curve via numerical integration
            for i, coil in enumerate(t.wires()):
                # coil = t.wires()
                print(f"processing coil {i}")
                wire_iterator = TopoDS_Iterator(coil)
                xyz = coil_extract(wire_iterator, pointspermeter, 
                                tolerance, force, startfunc)
                coils.append(xyz)
            if len(coils) == 0:
                print("No wires found in file.")
                try: 
                    coils.extend(coil_extract(t.edges(), pointspermeter, tolerance, force, startfunc, cutoffsphere))
                except Exception as e:
                    print("Error: no wires or edges found in file.")
                    print(e)
                    raise ValueError(e)

        else:
            coils.extend(coil_extract(t.edges(), pointspermeter, tolerance, force, startfunc, cutoffsphere))
    return coils


if __name__ == "__main__":
    coils = []
    compound = read_step_file("/Users/pharr/Downloads/" + \
                              "CS_STACK_FILAMENT_SKE#4BTN4M --C_08-07-2023.stp")
    t = TopologyExplorer(compound) # type: ignore
    pointspermeter = 50
    tolerance = 5/pointspermeter

    # Initial sample t measure the length of the curve via numerical integration
    for i, coil in enumerate(t.wires()):
        # coil = t.wires()
        print(f"processing coil {i}")
        wire_iterator = TopoDS_Iterator(coil)
        xyz = coil_extract(wire_iterator, pointspermeter, tolerance, force=False)
        coils.append(xyz)
        if i>0: break
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    interval = 50
    ax.plot(*coils[1][::interval].T)
    plt.show()

    