from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from typing import Union
from io import StringIO


def Rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.asarray([[1, 0, 0, 0], 
                    [0, np.cos(theta), -np.sin(theta), 0], 
                    [0, np.sin(theta), np.cos(theta), 0], 
                    [0, 0, 0, 1]], dtype = np.float64)
    return R

def Ry(theta):
    R = np.asarray([[np.cos(theta), 0, np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 0],
                    [0, 0, 0, 1]], dtype = np.float64)
    return R

def Rz(theta):
    R = np.asarray([[np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0,],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype = np.float64)
    return R


class coildata:
    def __init__(self, sourcefile = None):
        if sourcefile is not None:
            with open(sourcefile) as f:
                data = f.read()

            dl = data.split(' 1 ')
            dlines = data.splitlines()
            coils = defaultdict(list)
            groups = defaultdict(list)
            i = 0
            worker = []
            for line in dlines:
                if len(line.split()) != 4:
                    if len(line.split()) > 4:
                        worker.append([float(a) for a in line.split()[:4] ])
                        coils[line.split()[-1]].append(np.asarray(worker))
                        if len(line.split()) == 1:
                            groups[line.split()[-1]].append(0)
                        else:
                            groups[line.split()[-1]].append(line.split()[0])
                        worker = []
                    else:
                        continue
                else:
                    worker.append([float(a) for a in line.split()])
            
            self.coilsdict = coils
            self.groupsdict = groups

        else:
            coils = defaultdict(list)
            groups = defaultdict(list)
            self.coilsdict = coils
            self.groupsdict = groups

    def __str__(self):
        s = \
        f"""OMFIT COILS OBJECT\n{len(self.coilsdict.keys())} coils:\n"""
        for coil in self.coilsdict.keys():
            s += f"    {coil}: {len(self.groupsdict[coil])} groups\n"
        if len(self.coilsdict.keys()) == 0:
            s += "    No coils, object empty\n"

        return s
        
    def plot(self, key:Union[str,list]='all', rmax:float=3.,zmax:float=3,points='-'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if type(key) == str:            
            if key == 'all':

                for key in reversed(list(self.coilsdict.keys())):

                    for i in range(len(self.coilsdict[key])):
                        c1 = np.asarray(self.coilsdict[key][i])
                        x = c1.T[0]
                        y = c1.T[1]
                        z = c1.T[2]
                        ax.plot(x,y,z,points)

            else:
                for i in range(len(self.coilsdict[key])):
                    c1 = np.asarray(self.coilsdict[key][i])
                    x = c1.T[0]
                    y = c1.T[1]
                    z = c1.T[2]
                    ax.plot(x,y,z)
                print(len(self.coilsdict[key]))

        else:
            for k in key:
                for i in range(len(self.coilsdict[k])):
                    c1 = np.asarray(self.coilsdict[k][i])
                    x = c1.T[0]
                    y = c1.T[1]
                    z = c1.T[2]
                    ax.plot(x,y,z)
                
        ax.set_xlim(-rmax,rmax)
        ax.set_ylim(-rmax,rmax)
#         ax.set_zlim(-3,3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.tight_layout()
        plt.show()
        
        return

    def write(self,fname:str,direc='',coilnames=None,periods=12,coilformat='omfit',numcoils=1):
            
        if coilformat == 'omfit':
            header = f"periods {periods}\nbegin filaments\nmirror NUL\n"
            footer = "end\n"
            fxyz = '{:#16.8g}'
            fi = '{:#13.6g}'
            coils = self.coilsdict
            if coilnames is None:
                coilnames = list(coils.keys())       
            with open(f'{direc}/{fname}.coils','w') as f:
                f.write(header)
                for i in range(len(list(coils.keys()))):
    #                 print(coilnames[i])
                    if i == 0:
                        n = 1
                    else:
                        n = 2
                    key = list(coils.keys())[i]
                    for snip in coils[key]:
                        for j in range(len(snip)):
                            if j == len(snip)-1:
                                outline = fxyz.format(snip[j][0]) + fxyz.format(snip[j][1]) + \
                                fxyz.format(snip[j][2]) + fi.format(snip[j][3]) + (n*' ') + str(i+1) + ' ' + str(coilnames[i]) + '\n'
                            else:
                                outline = fxyz.format(snip[j][0]) + fxyz.format(snip[j][1]) + \
                                fxyz.format(snip[j][2]) + fi.format(snip[j][3]) + '\n'
                            f.write(outline)
                f.write(footer)
            
        elif coilformat == 'gpec':
            # does not yet support 'none' as coilnames, will default to writing all coils.
            for name in self.coilsdict.keys():
                fulllen = 0
                firstlen = len(self.coilsdict[name][0])
                for i in range(len(self.coilsdict[name])):
                    fulllen += len(self.coilsdict[name][i])
                # reminder: header vars mean 
                # | number of evenly divided coils in this file |
                # | not sure? Just keep 1 |
                # | total number of points in the file |
                # | number of windings about these coils |
                header = f" {str(numcoils).rjust(4)} {'1'.rjust(4)} {str(firstlen).rjust(4)} {str(periods).rjust(4)}.00\n"
                # header = f" {str(numcoils).rjust(4)} {'1'.rjust(4)} {str(fulllen).rjust(4)} {str(periods).rjust(4)}.00\n"
                footer = ''
                fxyz = '{:-13.4e}'

                with open(f"{direc}/gpeccoils/{fname}_{name.replace(' ','')}.dat",'w') as f:
                    f.write(header)
                    body = ''
                    for coil in self.coilsdict[name]:
                        for i in range(len(coil)):
                            line = coil[i]
                            linetxt = fxyz.format(line[0]) + fxyz.format(line[1]) + fxyz.format(line[2]) + '\n'
                            body += linetxt
                    f.write(body[:-1])


        else:
            raise ValueError('Unsupported coil format')

        return

    def addcoil(self, name, group, xyzi):
        self.coilsdict[name].append(np.array(xyzi))
        self.groupsdict[name].append(group)

        return

    def coilsgenerate(self, R, Z, dR, dZ, name, coilsshape = (1,1), numpoints = 500):

    

        diffr = diffz = 0

        if coilsshape[0] > 1:
            diffr = dR/(coilsshape[0] - 1)
        if coilsshape[1] > 1:
            diffz = dZ/(coilsshape[1] - 1)
        phi = np.linspace(0,2*np.pi,numpoints)

        for i in range(coilsshape[0]):
            for j in range(coilsshape[1]):
                rij = R + i*diffr
                zij = Z + j*diffz

                xyzi = np.asarray([rij*np.cos(phi), rij*np.sin(phi), zij*np.ones(numpoints), np.ones(numpoints)]).T

                self.addcoil(name, 0, xyzi)

        return    

    def shift(self, part:str = 'CP', direc:str = 'x', dx:float = 0.01):
        """
        DO NOT TRY TO SHIFT CENTER POST AND ITS INDIVIDUAL COMPONENTS IN DIFFERENT WAYS AT ONCE!
        """
        if part in self.coilsdict.keys():
            for i in range(len(self.coilsdict[part])):
                if direc == 'x':
                    self.coilsdict[part][i] += np.asarray([dx, 0, 0, 0])
                elif direc == 'y':
                    self.coilsdict[part][i] += np.asarray([0, dx, 0, 0])
                elif direc == 'z':
                    self.coilsdict[part][i] += np.asarray([0, 0, dx, 0])
                else:
                    print('invalid direction')
                    return
        
        elif part == 'CP':
            CP_fullparts = ['OH','PF1AU','PF1BU','PF1CU','PF1CL','PF1BL','PF1AL']
#             CP_fullparts = ['OH', 'PF1AU', 'PF1AL', 'PF1B']
            #non-TF
            for parta in CP_fullparts:
                for i in range(len(self.coilsdict[parta])):
                    if direc == 'x':
                        self.coilsdict[parta][i] += np.asarray([dx, 0, 0, 0])
                    elif direc == 'y':
                        self.coilsdict[parta][i] += np.asarray([0, dx, 0, 0])
                    elif direc == 'z':
                        self.coilsdict[parta][i] += np.asarray([0, 0, dx, 0])
                    else:
                        print('invalid direction')
                        return

            #TF
            #rlim^2 = 0.01000001 for TF coils
            for i in range(len(self.coilsdict['TF'])):
                for j in range(len(self.coilsdict['TF'][i])):
                    if self.coilsdict['TF'][i][j][0]**2 + self.coilsdict['TF'][i][j][1]**2 <= 0.01000001:
                        if direc == 'x':
                            self.coilsdict['TF'][i][j] += np.asarray([dx, 0, 0, 0])
                        elif direc == 'y':
                            self.coilsdict['TF'][i][j] += np.asarray([0, dx, 0, 0])
                        elif direc == 'z':
                            self.coilsdict['TF'][i][j] += np.asarray([0, 0, dx, 0])
                        else:
                            print('invalid direction')
                            return
            
        elif part == 'PF':
            PF_fullparts = [a for a in self.coilsdict.keys() if (a[:2] == 'PF')]
            for parta in PF_fullparts:
                for i in range(len(self.coilsdict[parta])):
                    if direc == 'x':
                        self.coilsdict[parta][i] += np.asarray([dx, 0, 0, 0])
                    elif direc == 'y':
                        self.coilsdict[parta][i] += np.asarray([0, dx, 0, 0])
                    elif direc == 'z':
                        self.coilsdict[parta][i] += np.asarray([0, 0, dx, 0])
                    else:
                        print('invalid direction')
                        return

            
        else:
            print('invalid part')
        
        print(f'Shifted {part} by {100*dx} cm in the +{direc}-direction.')
        return 
        
    def rotate(self, part:str = 'CP', axis = 'y', centerpoint = 'center', dtheta:float = 0.005):
        
        if part in self.coilsdict.keys():

            if centerpoint == 'center':
                centerpoint = np.mean(self.coilsdict[part], axis=(0,1))
            realcenter = np.asarray([centerpoint[0],centerpoint[1],centerpoint[2],0])


            for i in range(len(self.coilsdict[part])):
                if axis == 'x':
                    self.coilsdict[part][i] = np.matmul(Rx(dtheta), (self.coilsdict[part][i] - realcenter).T).T + realcenter
                elif axis == 'y':
                    self.coilsdict[part][i] = np.matmul(Ry(dtheta), (self.coilsdict[part][i] - realcenter).T).T + realcenter
                elif axis == 'z':
                    self.coilsdict[part][i] = np.matmul(Rz(dtheta), (self.coilsdict[part][i] - realcenter).T).T + realcenter
                else:
                    print('invalid axis')
                    
        elif part == 'CP':          
            
#             CP_fullparts = ['OH', 'PF1AU', 'PF1AL', 'PF1B']
            CP_fullparts = ['OH','PF1AU','PF1BU','PF1CU','PF1CL','PF1BL','PF1AL']
            #non-TF
            for parta in CP_fullparts:
                for i in range(len(self.coilsdict[parta])):
                    if axis == 'x':
                        self.coilsdict[parta][i] = np.matmul(Rx(dtheta), self.coilsdict[parta][i].T).T
                    elif axis == 'y':
                        self.coilsdict[parta][i] = np.matmul(Ry(dtheta), self.coilsdict[parta][i].T).T
                    elif axis == 'z':
                        self.coilsdict[parta][i] = np.matmul(Rz(dtheta), self.coilsdict[parta][i].T).T
                    else:
                        print('invalid axis')

            #TF
            #rlim^2 = 0.01000001 for TF coils
            for i in range(len(self.coilsdict['TF'])):
                for j in range(len(self.coilsdict['TF'][i])):
                    if self.coilsdict['TF'][i][j][0]**2 + self.coilsdict['TF'][i][j][1]**2 <= 0.01000001:
                        if axis == 'x':
                            self.coilsdict['TF'][i][j] = np.matmul(Rx(dtheta), self.coilsdict['TF'][i][j])
                        elif axis == 'y':
                            self.coilsdict['TF'][i][j] = np.matmul(Ry(dtheta), self.coilsdict['TF'][i][j])
                        elif axis == 'z':
                            self.coilsdict['TF'][i][j] = np.matmul(Rz(dtheta), self.coilsdict['TF'][i][j])
                        else:
                            print('invalid axis')
                            return
                        
        elif part == 'PF':
            PF_fullparts = [a for a in self.coilsdict.keys() if (a[:2] == 'PF')]
            for parta in PF_fullparts:
                for i in range(len(self.coilsdict[parta])):
                    if axis == 'x':
                        self.coilsdict[parta][i] = np.matmul(Rx(dtheta), self.coilsdict[parta][i].T).T
                    elif axis == 'y':
                        self.coilsdict[parta][i] = np.matmul(Ry(dtheta), self.coilsdict[parta][i].T).T
                    elif axis == 'z':
                        self.coilsdict[parta][i] = np.matmul(Rz(dtheta), self.coilsdict[parta][i].T).T
                    else:
                        print('invalid axis')
                        
        else:
            print('invalid part')
        
        print(f'Rotated {part} by {dtheta} radians about the {axis}-axis.')
        return