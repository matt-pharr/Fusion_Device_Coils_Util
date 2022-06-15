from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from typing import Union
from io import StringIO

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
  
    def write(self,fname:str,direc='',coilnames=None,periods=12,coilformat='omfit'):
            
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
                for i in range(len(self.coilsdict[name])):
                    fulllen += len(self.coilsdict[name][i])
                header = f" {str(1).rjust(4)} {'1'.rjust(4)} {str(fulllen).rjust(4)} {str(periods).rjust(4)}.00\n"
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

    def addcoil(self, name, group, xyzi):
        self.coilsdict[name].append(np.array(xyzi))
        self.groupsdict[name].append(group)
