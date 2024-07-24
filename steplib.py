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



import numpy as np
import matplotlib.pyplot as plt
# from steputils import p21
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.gp import gp_Pnt

from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import TopoDS_Wire, topods, TopoDS_Iterator
from OCC.Core.TopExp import TopExp_Explorer
# %matplotlib qt

debug_level = 0;

# Function to sample points along the curve
def sample_curve_points(curve, u_start, u_end, num_samples):
    """Sample points along a curve"""
    usample = np.linspace(u_start,u_end,num_samples)
    # du = (u_end-u_start)/100
    xyzwire = []
    for ui in usample:
        xyzi = np.asarray(curve.Value(ui).Coord())/1000
        xyzwire.append(xyzi)
    return np.asarray(xyzwire)

# Function to get the arc length of a curve
def get_arc_length(curve, u_start, u_end):
    """Get the arc length of a curve"""
    arclength = np.sum(np.linalg.norm(np.diff(sample_curve_points(curve, u_start, u_end, 100),axis=0),axis=1))
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

def wire_connecter(xyz:list, tolerance):
    """Find and join connected wires"""
    connected_remaining = True
    while connected_remaining:
        starts = [xyz[i][0] for i in range(len(xyz))]
        ends = [xyz[i][-1] for i in range(len(xyz))]
        break_flag = False
        for i in range(len(xyz)):
            for k in range(len(xyz)):
                if i != k:
                    if euc_dist(ends[i], starts[k]) < tolerance:
                        if debug_level > 0:
                            print("Found connected xyz.")
                        xyz[i] = np.concatenate((xyz[i],xyz[k]))
                        xyz.pop(k)
                        starts.pop(k)
                        ends.pop(k)
                        break_flag = True
                        break
                    elif euc_dist(ends[i], ends[k]) < tolerance:
                        xyz[i] = np.concatenate((xyz[i],xyz[k][::-1]))
                        xyz.pop(k)
                        starts.pop(k)
                        ends.pop(k)
                        break_flag = True
                        break
                    elif euc_dist(starts[i], starts[k]) < tolerance:
                        xyz[i] = np.concatenate((xyz[k][::-1],xyz[i]))
                        xyz.pop(k)
                        starts.pop(k)
                        ends.pop(k)
                        break_flag = True
                        break
            if break_flag:
                break
        if debug_level > 0:
            print("break_flag:",break_flag)
        
        # If no connections are found, break the loop
        if not break_flag:
            connected_remaining = False

    starts = [xyz[i][0] for i in range(len(xyz))]
    ends = [xyz[i][-1] for i in range(len(xyz))]
    for i in range(len(starts)):
        if starts[i][2] > ends[i][2]:
            xyz[i] = xyz[i][::-1]
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
        xyzpre = []

        for j, edge in list(enumerate(wire_iterator)):
            if debug_level > 1:
                print("segment:",j)
            curve, u_start, u_end = BRep_Tool().Curve(edge)

            # Set even spacing of points between coils
            arclength = get_arc_length(curve, u_start, u_end)
            numpoints = int(arclength*pointspermeter)
            if debug_level > 1:
                print(numpoints)
            
            # Rasterize points on the curve
            xyzwire = sample_curve_points(curve, u_start, u_end, numpoints)
            if len(xyzwire) == 0:
                continue
            # Remove points outside of the cutoff sphere
            # print(xyzwire.shape)
            dist = np.linalg.norm(xyzwire, axis=1)
            mask = dist <= cutoffsphere
            xyzwire = xyzwire[mask]

            xyzpre.append(np.asarray(xyzwire))

        # Find connected wires and join them
        xyz = wire_connecter(xyzpre, tolerance)
        
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
                    raise ValueError

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

    