"""
Library of basic functionality for working with coils for tokamaks or
stellarators in python. 


Author: Matthew Pharr
Date: 25 August 2023
matthew.pharr@columbia.edu

Copyright (c) 2023, Matthew Pharr.
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree. 
"""

from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from typing import Union

from steplib import coil_read

mayavi_successful = False
try:
	from mayavi import mlab
	mayavi_successful = False
except ImportError:
	print('Mayavi not installed, 3D plotting will use MPL')




def Rx(theta:np.float32):
	c = np.cos(theta)
	s = np.sin(theta)
	R = np.asarray([[1, 0, 0, 0], 
					[0, np.cos(theta), -np.sin(theta), 0], 
					[0, np.sin(theta), np.cos(theta), 0], 
					[0, 0, 0, 1]], dtype = np.float32)
	return R

def Ry(theta:np.float32):
	R = np.asarray([[np.cos(theta), 0, np.sin(theta), 0],
					[0, 1, 0, 0],
					[-np.sin(theta), 0, np.cos(theta), 0],
					[0, 0, 0, 1]], dtype = np.float32)
	return R

def Rz(theta:np.float32):
	R = np.asarray([[np.cos(theta), -np.sin(theta), 0, 0],
					[np.sin(theta), np.cos(theta), 0, 0,],
					[0, 0, 1, 0],
					[0, 0, 0, 1]], dtype = np.float32)
	return R


class coildata:
	"""
	A class for holding coil data and performing operations on it. 
	Coil data is stored in a dictionary of the form 
	{coilname: [coildata1, coildata2, ...]}. 
	coildata1, coildata2, etc. are numpy arrays of shape (n,4), 
	where n is the number of points in the coil. 
	Each row of the numpy array is of the form [x,y,z,I]. 
	The I value is always 1, but is included for 
	compatibility with the OMFIT coil format.
	"""
	def __init__(self, sourcefiles:list[str] = [''], 
				 sftype:str = 'OMFIT', debug:bool=False):
		"""
		sourcefiles: list of strings, each string is a path to a coil file.
		sftype: string, either 'OMFIT', 'GPEC', or 'STEP'.
		debug: boolean, whether to print debug statements.
		"""
		coils = defaultdict(list)
		groups = defaultdict(list)
		self.coilsdict = coils
		self.groupsdict = groups
		
		if sourcefiles != ['']:
			self.readcoils(sourcefiles, sftype, debug)
		else:
			pass

	def __str__(self):
		s = \
		f"""COILS OBJECT\n{len(self.coilsdict.keys())} \
			coils:\n"""
		for coil in self.coilsdict.keys():
			s += f"    {coil}: {len(self.groupsdict[coil])} groups\n"
		if len(self.coilsdict.keys()) == 0:
			s += "    No coils, object empty\n"

		return s
		
	def readcoils(self, sourcefiles:list[str], 
				  sftype:str='OMFIT', debug:bool=False,
				    name='tempname', startfunc = lambda x: x is None,
					 cutoffsphere = 1e6, ppm=(100*np.pi),
					 tol=5/(100*np.pi)) -> None:
		
		for sourcefile in sourcefiles:
			
			if sftype.upper() == 'OMFIT':
				with open(sourcefile, 'r') as f:
					data = f.read()
					print('found coils file')
				dl = data.split(' 1 ')
				dlines = data.splitlines()
				coils = defaultdict(list)
				groups = defaultdict(list)
				i = 0
				worker = []
				if debug:
					print(dlines)
				for line in dlines:
					if len(line.split()) != 4:
						if len(line.split()) > 4:
							worker.append([float(a) for a \
										   in line.split()[:4] ])
							coils[line.split()[-1]].append(
								np.asarray(worker))
							if len(line.split()) == 1:
								groups[line.split()[-1]].append(0)
							else:
								groups[line.split()[-1]].append(
									line.split()[0])
							worker = []
						else:
							continue
					else:
						worker.append([float(a) for a in line.split()])
				
				self.coilsdict.update(coils)
				self.groupsdict.update(groups)
			elif sftype.upper() == 'GPEC':
				with open(sourcefile, 'r') as f:
					data = f.read()
					print('found coils file')
				dlines = data.splitlines()
				header = dlines[0].split()
				xyzlong = np.array([x.split() for x \
									in dlines[1:]]).astype(float)
				if int(header[2]) == np.shape(xyzlong)[0]/\
					int(header[0]) and debug:
					print('correct array lengths')
				elif debug:
					print('error in data file')
				else: 
					pass
				xyz = xyzlong.reshape(int(header[0]), 
									  int(header[2]), 3)
				xyzi = np.empty((int(header[0]), 
								 int(header[2]), 4))
				xyzi[:,:,:-1] = xyz[:,:,:]
				xyzi[:,:,-1] = 1.
				# print(xyzi)
				for i in range(int(header[0])):
					# self.addcoil(sourcefile[-8:-4].upper(),
					# 			 0,xyzi[i,:,:])
					self.addcoil(sourcefile[-8:-4],0,xyzi[i,:,:])
					
			elif sftype.upper() == 'STEP':
				print('found STEP file')
				# ppm = 100*np.pi
				xyzi = coil_read([sourcefile], pointspermeter=ppm, 
								 tolerance=tol, force=False, 
								 startfunc=startfunc, 
								 cutoffsphere=cutoffsphere)
				coilnum = len(xyzi)
				for i in range(coilnum):
					coil = np.empty((len(xyzi[i]),4))
					coil[:,:-1] = xyzi[i]
					coil[:,-1] = 1.
					self.addcoil(f"{name}_{i}", 0, coil)

			else:
				print('unsupported coil file type')
				raise ValueError

	def plot(self, key:Union[str,list]='all', rmax:float=3.,
			 zmax:float=3,points='-', legend=False, 
			 colorlist = None, duallegend=None, fig=None, ax=None):
		
		if not mayavi_successful:
			colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
			if fig is None:
				fig = plt.figure(figsize=(10,6))
			if ax is None:
				ax = fig.add_subplot(111, projection='3d')
			else:
				pass

		
		else:
			colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
			if fig is None:
				fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 500))
			
			return self.__mayavi_plot(key, fig, colorlist)



		if type(key) == str:            
			if key == 'all':
				if type(colorlist) == list:
					colors = colorlist
				elif type(colorlist) == str:
					colors = len(self.coilsdict.keys())*[colorlist]
				for j, keyn in enumerate(reversed(\
					list(self.coilsdict.keys()))):

					for i in range(len(self.coilsdict[keyn])):
						c1 = np.asarray(self.coilsdict[keyn][i])
						x = c1.T[0]
						y = c1.T[1]
						z = c1.T[2]
						ax.plot(x,y,z,points,color=\
								colors[j%len(colors)], 
								label=keyn.upper())

			else:
				if type(colorlist) == list:
					colors = colorlist
				elif type(colorlist) == str:
					colors = [colorlist]
				for i in range(len(self.coilsdict[key])):
					c1 = np.asarray(self.coilsdict[key][i])
					x = c1.T[0]
					y = c1.T[1]
					z = c1.T[2]
					ax.plot(x,y,z,points,color = colors[0], 
			 					label = key.upper())
				print(len(self.coilsdict[key]))

		else:
			if colorlist is not None:
				if type(colorlist) == list:
					colors = colorlist
				elif type(colorlist) == str:
					colors = len(key)*[colorlist]
			for j, k in enumerate(key):
				for i in range(len(self.coilsdict[k])):
					
					c1 = np.asarray(self.coilsdict[k][i])
					x = c1.T[0]
					y = c1.T[1]
					z = c1.T[2]
					if i == 0:
						ax.plot(x,y,z,color=colors[j%len(colors)],label=k.upper())
					else:
						ax.plot(x,y,z,color=colors[j%len(colors)])

				
		ax.set_xlim(-rmax,rmax)
		ax.set_ylim(-rmax,rmax)
		ax.set_zlim(-zmax,zmax)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		if legend:
			l1 = plt.legend(loc=1)
			ax.add_artist(l1)
		if duallegend is not None:
			for key in duallegend.keys():
				ax.plot(np.NaN, np.NaN, c=duallegend[key])
			lines = ax.get_lines()
			l2 = plt.legend(lines[-len(duallegend.keys()):], 
							duallegend.keys(), loc=4)
			ax.add_artist(l2)
		
		return fig, ax

	def __mayavi_plot(self, key:Union[str,list], fig, colors):
		if colors is None:
			colors = [(0,0,0)]*len(self.coilsdict.keys())

		if type(key) == str:
			if key == 'all':
				for j, key in enumerate(self.coilsdict.keys()):
					for i in range(len(self.coilsdict[key])):
						c1 = np.asarray(self.coilsdict[key][i])
						x = c1.T[0]
						y = c1.T[1]
						z = c1.T[2]
						o = mlab.plot3d(x, y, z, tube_radius=0.01, figure=fig, color=colors[j])
			else:
				for i in range(len(self.coilsdict[key])):
					c1 = np.asarray(self.coilsdict[key][i])
					x = c1.T[0]
					y = c1.T[1]
					z = c1.T[2]
					o = mlab.plot3d(x, y, z, tube_radius=0.01, figure=fig, color=colors[0])
		
		elif type(key) == list:
			for key in key:
				for i in range(len(self.coilsdict[key])):
					c1 = np.asarray(self.coilsdict[key][i])
					x = c1.T[0]
					y = c1.T[1]
					z = c1.T[2]
					o = mlab.plot3d(x, y, z, tube_radius=0.01, figure=fig, color=colors[0])
		
		else:
			raise ValueError('Invalid key type')

		return o, None

	def write(self,fname:str,direc='',coilnames=None,periods=1,
			  coilformat='omfit',numcoils=1):
		"""
		Writes coil data to a file.
		fname: string, name of file to write to.
		direc: string, directory to write to.
		coilnames: list of strings, names of coils to write.
		periods: int or dict, number of turns per coil, or dict of coil names to number of turns.
		coilformat: string, either 'omfit' or 'gpec'.
		numcoils: int, number of coils to write.
		"""
			
		if coilformat.upper() == 'OMFIT':
			header = f"periods {periods}\nbegin filaments\
				\nmirror NUL\n"
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
								outline = fxyz.format(snip[j][0])\
									  + fxyz.format(snip[j][1]) + \
								fxyz.format(snip[j][2]) + \
									fi.format(snip[j][3]) + \
										(n*' ') + str(i+1) + ' ' + \
											str(coilnames[i]) + '\n'
							else:
								outline = fxyz.format(snip[j][0])\
									+ fxyz.format(snip[j][1]) + \
									fxyz.format(snip[j][2]) + \
									fi.format(snip[j][3]) + '\n'
								
							f.write(outline)

				f.write(footer)
			
		
		elif coilformat.upper() == 'GPEC':
			# does not yet support 'none' as coilnames, will default to writing all coils.
			for name in self.coilsdict.keys():
				numcoils = len(self.coilsdict[name])
				fulllen = 0
				firstlen = len(self.coilsdict[name][0])
				for i in range(len(self.coilsdict[name])):
					fulllen += len(self.coilsdict[name][i])
				# reminder: header vars mean 
				# | number of evenly divided coils in this file |
				# | not sure? Just keep 1 |
				# | total number of points in the file |
				# | number of windings about these coils |
					
				if type(periods) == int:
					periodsi = periods
				else:
					periodsi = int(periods[name])

				header = f" {str(numcoils).rjust(4)}" +\
					f" {'1'.rjust(4)} {str(firstlen).rjust(4)}" +\
						f" {str(periodsi).rjust(4)}.00\n"
				# header = f" {str(numcoils).rjust(4)} {'1'.rjust(4)}\
				#  {str(fulllen).rjust(4)} \
				# {str(periods).rjust(4)}.00\n"
				footer = ''
				fxyz = '{:-13.4e}'

				with open(f"{direc}/gpeccoils/{fname}_" + \
						  f"{name.replace(' ','')}.dat",'w') as f:
					f.write(header)
					body = ''
					for coil in self.coilsdict[name]:
						for i in range(len(coil)):
							line = coil[i]
							linetxt = fxyz.format(line[0]) + \
								fxyz.format(line[1]) + \
									fxyz.format(line[2]) + '\n'
							body += linetxt
					f.write(body[:-1])


		else:
			raise ValueError('Unsupported coil format')

		return

	def addcoil(self, name, group, xyzi):
		self.coilsdict[name].append(np.array(xyzi))
		self.groupsdict[name].append(group)

		return

	def coilsgenerate(self, R, Z, dR, dZ, name, \
					  coilsshape = (1,1), numpoints = 500, rot=None):

	
		if rot is None:
			rot = 0
		diffr = diffz = 0

		if coilsshape[0] > 1:
			diffr = dR/(coilsshape[0] - 1)
		if coilsshape[1] > 1:
			diffz = dZ/(coilsshape[1] - 1)
		phi = np.linspace(0,2*np.pi,numpoints)

		for i in range(coilsshape[0]):
			for j in range(coilsshape[1]):
				rij = i*diffr
				zij = j*diffz

				rijtemp = rij*np.cos(rot) - zij*np.sin(rot)
				zij = rij*np.sin(rot) + zij*np.cos(rot) + Z
				rij = rijtemp + R

				xyzi = np.asarray([rij*np.cos(phi), \
								   rij*np.sin(phi), \
									zij*np.ones(numpoints), \
										np.ones(numpoints)]).T

				self.addcoil(name, 0, xyzi)

		return    

	def shift(self, part:str = 'CP', direc:str = 'x', dx:float = 0.01):
		"""
		DO NOT TRY TO SHIFT CENTER POST AND ITS \
			INDIVIDUAL COMPONENTS IN DIFFERENT WAYS AT ONCE!
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
		
	def rotate(self, part:str = 'CP', axis = 'y', centerpoint = 'center', dtheta:np.float64 = 0.005):
		
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
	
	def coilscombine(self, testcoilslist:list[str]) -> None:
		"""
		Finds leads associated with coils in testcoilslist and
		combines if adjacent. 
		"""
		return
	
	def join(self, coillist:list[str], name:str, tolmult=2.) -> None:
		"""
		Combines a list of connected coils.
		"""

		starts = {}
		ends = {}
		clist1 = {}

		maxdiff = 0
		for coil in coillist:
			c = self.coilsdict[coil]
			for i, ci in enumerate(c):
				coilname = f"{coil}_{i}"
				starts[coilname] = ci[0]
				ends[coilname] = ci[-1]
				clist1[coilname] = ci
				diff = max(np.linalg.norm(np.diff(ci, axis=0), axis=1))
				if diff > maxdiff:
					maxdiff = diff

		tol = maxdiff*tolmult
		
		connectionmatrix = np.empty((len(clist1),len(clist1)))

		for i, coil in enumerate(clist1.keys()):
			for j, coil2 in enumerate(clist1.keys()):
				if i == j:
					connectionmatrix[i,j] = 0
				else:
					if np.linalg.norm(ends[coil] - starts[coil2]) < tol:
						connectionmatrix[i,j] = 1
					elif np.linalg.norm(ends[coil2] - starts[coil]) < tol:
						connectionmatrix[i,j] = -1
					else:	
						connectionmatrix[i,j] = 0

		print(connectionmatrix)
		totalcoil = np.empty(len(clist1.keys()), dtype=object)
		last = len(totalcoil) + 1
		for i in range(len(clist1.keys())):
			if sum(connectionmatrix[i]) == 1:
				totalcoil[0] = list(clist1.keys())[i]
				last = i
		print(totalcoil)
		for i in range(1,len(clist1.keys())):
			for j in range(len(clist1.keys())):
				if connectionmatrix[last,j] == 1:
					totalcoil[i] = list(clist1.keys())[j]
					print(totalcoil)
					last = j
					break
		
		conclist = []
		for i in range(len(totalcoil)):
			conclist.append(clist1[totalcoil[i]])
		
		finalcoil = np.concatenate(conclist)

		self.coilsdict[name] = [finalcoil]


		return

	def getcoils(self):
		return self.coilsdict.keys()
	


if __name__ == '__main__':
	import h5py
	import xarray as xr

	cobj = coildata()

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')


	a = h5py.File('/Users/pharr/Downloads/ITER_windings_simon_mcintosh/PF.nc','r')
	k = list(a.keys())[0]
	anc = xr.open_dataset('/Users/pharr/Downloads/ITER_windings_simon_mcintosh/PF.nc', group=k)

	for name in anc.coords['coil_name'].values:
		xyz = anc['points'].sel(coil_name=name)

		if name != 'PF3':
			xyzi = xyz[:np.argmax(np.linalg.norm(np.diff(xyz, axis=0), axis=1))+1]
		else:
			xyzi = xyz

		# Add a column of 0s to the end of the array
		xyz0s = np.hstack((np.asarray(xyzi), np.zeros(np.asarray(xyzi).shape[0])[:,np.newaxis]))
		cobj.addcoil(name, group=0, xyzi=xyz0s)
	#     ax.plot(xyzi[:, 0], xyzi[:, 1], xyzi[:, 2], label=name)



	a = h5py.File('/Users/pharr/Downloads/ITER_windings_simon_mcintosh/CS.nc','r')
	k = list(a.keys())[0]
	anc = xr.open_dataset('/Users/pharr/Downloads/ITER_windings_simon_mcintosh/CS.nc', group=k)

	for name in anc.coords['coil_name'].values:
		xyz = anc['points'].sel(coil_name=name)
		if name != 'CS1U':
			xyzi = xyz[:np.argmax(np.linalg.norm(np.diff(xyz, axis=0), axis=1))+1]
		else:
			xyzi = xyz

		# Add a column of 0s to the end of the array
		xyz0s = np.hstack((np.asarray(xyzi), np.zeros(np.asarray(xyzi).shape[0])[:,np.newaxis]))
		cobj.addcoil(name, group=0, xyzi=xyz0s)
	#     ax.plot(xyzi[:, 0], xyzi[:, 1], xyzi[:, 2], label=name)

	a = h5py.File('/Users/pharr/Downloads/ITER_windings_simon_mcintosh/TF.nc','r')
	k = list(a.keys())[0]
	anc = xr.open_dataset('/Users/pharr/Downloads/ITER_windings_simon_mcintosh/TF.nc', group=k)

	for name in anc.coords['coil_name'].values:

		xyz = anc['points'].sel(coil_name=name)[1:]
		xyz = xyz[:np.argmax(np.linalg.norm(np.diff(xyz, axis=0), axis=1))+1]
		xyz0s = np.hstack((np.asarray(xyz), np.zeros(np.asarray(xyz).shape[0])[:,np.newaxis]))
		cobj.addcoil(name, group=1, xyzi=xyz0s)


	# fig, ax = cobj.plot([tfname for tfname in cobj.coilsdict if 'tf' in tfname.lower()])
	fig, ax = cobj.plot()
	mlab.show()