"""
Library of basic functionality for extracting coil data
from STPs in python.
Author: Matthew Pharr
Date: 25 August 2023
matthew.pharr@columbia.edu
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
    tail_to_tail_dist = np.linalg.norm(xyz[-1] - xyzadd[-1])
    tail_to_head_dist = np.linalg.norm(xyz[-1] - xyzadd[0])
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
    else:
        # print("Error: segments are too far apart. Change tolerance.")
        # print("xyz[-1]:",xyz[-1])
        # print("xyzadd[-1]:",xyzadd[-1])
        # print("xyz[0]:",xyzadd[0])
        # print("tail to head dist:",tail_to_head_dist)
        # raise ValueError
        isvalid = False
    return isvalid  

# Function to extract a coil from a wire
def coil_extract(wire_iterator, pointspermeter:float, tolerance:float = -1., force:bool=False):
    """
    Extract a coil from a wire.
    wire_iterator: TopoDS_Iterator
    pointspermeter: Fideliy with wich to extract discrete coil geometry. Given in points per meter.
    tolerance: Tolerance for connecting segments. Given in meters.
    """
    if tolerance < 0:
        tolerance = 2/pointspermeter
        if debug_level > 0:
            print("Tolerance unspecified or negative. Defaulting to double spacing.")


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
    
    return np.asarray(xyz)

def coil_read(sourcefiles:list[str] = [''],
              pointspermeter:float = 50,
               tolerance:float = -1., 
               force:bool=False) -> list[np.ndarray]:
    """
    Read in a coil from a STEP file.
    sourcefiles: list of strings containing the path to the STEP file.
    """
    coils = []
    for sourcefile in sourcefiles:
        compound = read_step_file(sourcefile)
        t = TopologyExplorer(compound) # type: ignore
        pointspermeter = 50
        tolerance = 5/pointspermeter

        # Initial sample t measure the length of 
        # the curve via numerical integration
        for i, coil in enumerate(t.wires()):
            # coil = t.wires()
            print(f"processing coil {i}")
            wire_iterator = TopoDS_Iterator(coil)
            xyz = coil_extract(wire_iterator, pointspermeter, 
                               tolerance, force)
            coils.append(xyz)
    return coils


if __name__ == "main":
    coils = []
    compound = read_step_file("/Users/pharr/Downloads/\
                              CS_STACK_FILAMENT_SKE#4BTN4M\
                               --C_08-07-2023.stp")
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

    