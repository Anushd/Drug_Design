from numpy import *
import numpy as np
from math import *
from sklearn import datasets, svm, metrics 
set_printoptions(threshold='nan')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from decimal import *
from itertools import product, combinations
import os

#Set precision of decimal calulations
getcontext().prec = 5

###Import_Structures###
path = '/Users/anush/Desktop/Final_Ligands/'

structures_min = []
count=0
for i in os.listdir(path):
	if count != 0:
		file1 = open("/Users/anush/Desktop/Final_Ligands/"+i, "r")
		structures_min.append(file1)
	count += 1

structures = []	
count=0
for i in os.listdir(path):
	if count != 0:
		file2 = open("/Users/anush/Desktop/Final_Ligands/"+i, "r")
		structures.append(file2)
	count += 1
count =0

"""
structures_min = [open("/Users/anush/Desktop/Final_Ligands/713182_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/6605027_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/5389521_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3247284_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3236943_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3236517_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/654602_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/653221_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/8548995_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/49786375_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/4911035_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3247293_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3147229_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/2982527_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/2508673_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/16231644_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/16031807_out.pdb", "r")]

structures = [open("/Users/anush/Desktop/Final_Ligands/713182_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/6605027_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/5389521_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3247284_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3236943_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3236517_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/654602_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/653221_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/8548995_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/49786375_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/4911035_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3247293_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3147229_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/2982527_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/2508673_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/16231644_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/16031807_out.pdb", "r")]"""


#[binding affinity]
data = [[-7.3],[-7.9],[-8.6],[-8.2],[-7.4],[-8.7],[-8.2],[-8.6],[-9.4],[-8.7],[-8.6],[-8.7],[-9.1],[-9.5],[-7.3],[-8.6],[-9.7]]

###Parse_PDB###
def Coords(structure):
	
	coord_x = []
	coord_y = []
	coord_z = []
	count = 0
	
	for i in structure:
		if i[:5] == 'MODEL':
			count += 1
		
		if i[:4] == 'ATOM' and count < 2:
			coord_x.append(float(i[31:38])*10)
			coord_y.append(float(i[39:46])*10)
			coord_z.append(float(i[47:54])*10)
		
		if i[:6] == 'HETATM' and count < 2:
			coord_x.append(float(i[31:38])*10)
			coord_y.append(float(i[39:46])*10)
			coord_z.append(float(i[47:54])*10)
	
	return coord_x, coord_y, coord_z

###Calculate_Global_Minimum###
def Min(structures):
	
	structures_x = []
	structures_y = []
	structures_z = []
	
	#Extract coordinates and seperate by dimension for each structure
	for i in structures:
		coords = Coords(i)
		structures_x.append(coords[0])
		structures_y.append(coords[1])
		structures_z.append(coords[2])
	#Find minimum value in each of x/y/z sets
	minsx = []
	for i in structures_x:
		minsx.append(min(i))
	minsy = []
	for i in structures_y:
		minsy.append(min(i))
	minsz = []
	for i in structures_z:
		minsz.append(min(i))
	
	#Set initial translation values to 0
	add_x = 0
	add_y = 0
	add_z = 0
	
	#If minimum in each of x/y/z negative, then change value of add_x/y/z; else set equal to 0
	if min(minsx) < 0:
		add_x = abs(min(minsx))
	elif min(minsx) > 0:
		add_x = -1*min(minsx)
	if min(minsy) < 0:
		add_y = abs(min(minsy))
	elif min(minsy) > 0:
		add_y = -1*min(minsy)
	if min(minsz) < 0:
		add_z = abs(min(minsz))
	elif min(minsz) > 0:
		add_z = -1*min(minsz)
	
	return add_x, add_y, add_z

###Translate_Atom_Coordinates
def Translate(add_x, add_y, add_z, coord_x, coord_y, coord_z):
	
	x = []
	y = []
	z = []
	
	#Add minimums to each of x/y/z
	for i in coord_x:
		x.append(i + add_x)
	
	for i in coord_y:
		y.append(i + add_y)
	
	for i in coord_z:
		z.append(i + add_z)
	
	return x, y, z
	
###Calculate_Global_Maximums###
def Max(structures):
	structures_x = []
	structures_y = []
	structures_z = []
	
	#Extract coordinates and seperate by dimension for each structure
	for i in structures:
		structures_x.append(i[0])
		structures_y.append(i[1])
		structures_z.append(i[2])
	#Find maximum value in each of x/y/z sets
	maxsx = []
	for i in structures_x:
		maxsx.append(max(i))
	maxsy = []
	for i in structures_y:
		maxsy.append(max(i))
	maxsz = []
	for i in structures_z:
		maxsz.append(max(i))
	x_lim = ceil(max(maxsx))+1
	y_lim = ceil(max(maxsy))+1
	z_lim = ceil(max(maxsz))+1
	model = zeros((x_lim, y_lim, z_lim))
	return model

###Construct_Models###
def Construct(coord_x, coord_y, coord_z, blank_model):
	
	#Recieve blank model with appropriate dimensions
	model = blank_model
	count = 0
	
	#Define seperate x,y,z arrays for actual coordinate values (round+int)
	real_x = []
	real_y = []
	real_z = []
	
	#Append 1 to zeroes model at given coordinates
	for i in coord_x:
		model[int(round(i))][int(round(coord_y[count]))][int(round(coord_z[count]))] = 1
		
		#Add indices used before as new x,y,z values
		real_x.append(int(round(i)))
		real_y.append(int(round(coord_y[count])))
		real_z.append(int(round(coord_z[count])))
		count +=1
	
	return model, real_x, real_y, real_z

###Partition###
def Partition(model1):
	
	#Find shape of array
	shape = model1.shape
	x_new = []
	
	#I lost myself at this point
	if shape[2]%2 == 0:
		for i in model1:
			for j in i:
				new_shape1 = shape[2]
				x_new.append(j)
				
	elif shape[2]%2 == 1:
		for i in model1:
			for j in i:
				new_shape1 = shape[2]-1
				x_new.append(j[0:new_shape1])

	model2 = reshape(x_new,(shape[0],shape[1],new_shape1))
	y_new = []

	if shape[1]%2 == 0:
		for i in model2:
			new_shape2 = shape[1]
			y_new.append(i)
			
	elif shape[1]%2 == 1:
		for i in model2:
			new_shape2 = shape[1]-1
			y_new.append(i[0:new_shape2])

	model3 = reshape(y_new, (shape[0],new_shape2,new_shape1))
	z_new = []

	if shape[0]%2 == 0:
		new_shape3 = shape[0]
		z_new.append(model3)
		
	elif shape[0]%2 == 1:
		new_shape3 = shape[0]-1
		z_new.append(model3[0:new_shape3])
		
	final_model = reshape(z_new, (new_shape3, new_shape2, new_shape1))
	
	split_x = hsplit(final_model,2)
	split_y = []

	for i in split_x:
		split_y.append(vsplit(i,2))

	split_z = []
	for i in split_y:
		for j in i:
			split_z.append(dsplit(j,2))

	models_out = []			

	for i in split_z:
		for j in i:
			models_out.append(j)
	
	return models_out

###Calculate_Probabilities###
def Probability(models):

	counts = [0,0,0,0,0,0,0,0]
	num = 0
	for i in models:
		for j in i:
			for k in j:
				for l in k:
					if l == 1:
						counts[num] += 1
		num += 1	

	sum1 = sum(counts)
	
	if sum1 == 0:
		return counts
	else:
		index = 0
		for i in counts:
			counts[index] = Decimal(i)/Decimal(sum1)
			index += 1
	
		return counts

###SVM_Classification###
def SVM(probs,data,quadrant,to_predict):
	label_set = []
	count=0
	for i in probs:
		label_set.append(i[quadrant])
		count+=1	
	label_set = array(label_set)
	data_set = array(data)
	clf = svm.SVR(kernel='rbf', gamma=1e-3,C=1e2)
	clf.fit(data_set, label_set)  
	
	prediction = clf.predict([to_predict])
	return prediction

###Run_Functions###		
min = Min(structures_min)

s = []
structures_max = []
count=0
for i in structures:
	s.append(Coords(i))
	structures_max.append(Translate(min[0], min[1], min[2], (s[count])[0], (s[count])[1], (s[count])[2]))
	count+=1

maxs = []
count=0
for i in structures_max:
	maxs.append(Max(structures_max))
	count+=1

models = []
count=0
for i in structures_max:
	models.append(Construct(i[0],i[1],i[2],maxs[count]))
	count+=1

partitions = []
count=0
for i in models:
	partitions.append(Partition(i[0]))
	count+=1
probs = []
count=0
for i in partitions:
	probs.append(Probability(i))
	count+=1

quadrant = [0,1,2,3,4,5,6,7]
to_predict = -8.0

finalA = []
for i in quadrant:
	finalA.append(SVM(probs,data,i,to_predict))
print "First Iteration:"
print
print "Predicted:"
print finalA
print
print
print "Second Iteration:"
print 

#Second iteration 
quad_count=0
final_graph=[]
for i in quadrant:
	print "Quadrant:"
	print i+1
	print
	
	partitionsb = []
	count=0
	for j in partitions:
		partitionsb.append(Partition(j[i]))
		count+=1
	
	probsb = []
	count=0
	for j in partitionsb:
		probsb.append(Probability(j))
		count+=1

	finalB = []
	for l in quadrant:
		finalB.append(SVM(probsb,data,l,to_predict))
	final_graph.append(finalB)
	print "Predicted:"
	print finalB
	print
	print

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(model1[1], model1[2], model1[3])
plt.show()'''
'''
x=0
for i in final_graph:
	y=min(0,5,3,2)
	z=0
	for l in i:
		if y<0:
			final_graph[x][z]+=abs(y)
		z+=1
	x+=1	
print
print final_graph'''

###Plot_Probabilities###
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
ax.grid(True)
ax.set_axis_on()

#Quadrant_1
u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[0][0]+6
y = np.sin(u)*np.sin(v)*final_graph[0][0]+6
z = np.cos(v)*final_graph[0][0]
ax.plot_wireframe(x, y, z, color="r")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[0][1]+4
y = np.sin(u)*np.sin(v)*final_graph[0][1]+6
z = np.cos(v)*final_graph[0][1]
ax.plot_wireframe(x, y, z, color="r")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[0][2]+4
y = np.sin(u)*np.sin(v)*final_graph[0][2]+4
z = np.cos(v)*final_graph[0][2]
ax.plot_wireframe(x, y, z, color="r")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[0][3]+6
y = np.sin(u)*np.sin(v)*final_graph[0][3]+4
z = np.cos(v)*final_graph[0][3]
ax.plot_wireframe(x, y, z, color="r")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[0][4]+6
y = np.sin(u)*np.sin(v)*final_graph[0][4]+6
z = np.cos(v)*final_graph[0][4]+2
ax.plot_wireframe(x, y, z, color="r")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[0][5]+4
y = np.sin(u)*np.sin(v)*final_graph[0][5]+6
z = np.cos(v)*final_graph[0][5]+2
ax.plot_wireframe(x, y, z, color="r")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[0][6]+4
y = np.sin(u)*np.sin(v)*final_graph[0][6]+4
z = np.cos(v)*final_graph[0][6]+2
ax.plot_wireframe(x, y, z, color="r")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[0][7]+6
y = np.sin(u)*np.sin(v)*final_graph[0][7]+4
z = np.cos(v)*final_graph[0][7]+2
ax.plot_wireframe(x, y, z, color="r")

#Quadrant_2
u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[1][0]+2
y = np.sin(u)*np.sin(v)*final_graph[1][0]+6
z = np.cos(v)*final_graph[1][0]
ax.plot_wireframe(x, y, z, color="c")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[1][1]
y = np.sin(u)*np.sin(v)*final_graph[1][1]+6
z = np.cos(v)*final_graph[1][1]
ax.plot_wireframe(x, y, z, color="c")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[1][2]
y = np.sin(u)*np.sin(v)*final_graph[1][2]+4
z = np.cos(v)*final_graph[1][2]
ax.plot_wireframe(x, y, z, color="c")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[1][3]+2
y = np.sin(u)*np.sin(v)*final_graph[1][3]+4
z = np.cos(v)*final_graph[1][3]
ax.plot_wireframe(x, y, z, color="c")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[1][4]+2
y = np.sin(u)*np.sin(v)*final_graph[1][4]+6
z = np.cos(v)*final_graph[1][4]+2
ax.plot_wireframe(x, y, z, color="c")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[1][5]
y = np.sin(u)*np.sin(v)*final_graph[1][5]+6
z = np.cos(v)*final_graph[1][5]+2
ax.plot_wireframe(x, y, z, color="c")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[1][6]
y = np.sin(u)*np.sin(v)*final_graph[1][6]+4
z = np.cos(v)*final_graph[1][6]+2
ax.plot_wireframe(x, y, z, color="c")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[1][7]+2
y = np.sin(u)*np.sin(v)*final_graph[1][7]+4
z = np.cos(v)*final_graph[1][7]+2
ax.plot_wireframe(x, y, z, color="c")

#Quadrant_3
u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[2][0]+2
y = np.sin(u)*np.sin(v)*final_graph[2][0]+2
z = np.cos(v)*final_graph[2][0]
ax.plot_wireframe(x, y, z, color="g")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[2][1]
y = np.sin(u)*np.sin(v)*final_graph[2][1]+2
z = np.cos(v)*final_graph[2][1]
ax.plot_wireframe(x, y, z, color="g")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[2][2]
y = np.sin(u)*np.sin(v)*final_graph[2][2]
z = np.cos(v)*final_graph[2][2]
ax.plot_wireframe(x, y, z, color="g")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[2][3]+2
y = np.sin(u)*np.sin(v)*final_graph[2][3]
z = np.cos(v)*final_graph[2][3]
ax.plot_wireframe(x, y, z, color="g")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[2][4]+2
y = np.sin(u)*np.sin(v)*final_graph[2][4]+2
z = np.cos(v)*final_graph[2][4]+2
ax.plot_wireframe(x, y, z, color="g")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[2][5]
y = np.sin(u)*np.sin(v)*final_graph[2][5]+2
z = np.cos(v)*final_graph[2][5]+2
ax.plot_wireframe(x, y, z, color="g")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[2][6]
y = np.sin(u)*np.sin(v)*final_graph[2][6]
z = np.cos(v)*final_graph[2][6]+2
ax.plot_wireframe(x, y, z, color="g")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[2][7]+2
y = np.sin(u)*np.sin(v)*final_graph[2][7]
z = np.cos(v)*final_graph[2][7]+2
ax.plot_wireframe(x, y, z, color="g")

#Quadrant_4
u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[3][0]+6
y = np.sin(u)*np.sin(v)*final_graph[3][0]+2
z = np.cos(v)*final_graph[3][0]
ax.plot_wireframe(x, y, z, color="m")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[3][1]+4
y = np.sin(u)*np.sin(v)*final_graph[3][1]+2
z = np.cos(v)*final_graph[3][1]
ax.plot_wireframe(x, y, z, color="m")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[3][2]+4
y = np.sin(u)*np.sin(v)*final_graph[3][2]
z = np.cos(v)*final_graph[3][2]
ax.plot_wireframe(x, y, z, color="m")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[3][3]+6
y = np.sin(u)*np.sin(v)*final_graph[3][3]
z = np.cos(v)*final_graph[3][3]
ax.plot_wireframe(x, y, z, color="m")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[3][4]+6
y = np.sin(u)*np.sin(v)*final_graph[3][4]+2
z = np.cos(v)*final_graph[3][4]+2
ax.plot_wireframe(x, y, z, color="m")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[3][5]+4
y = np.sin(u)*np.sin(v)*final_graph[3][5]+2
z = np.cos(v)*final_graph[3][5]+2
ax.plot_wireframe(x, y, z, color="m")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[3][6]+4
y = np.sin(u)*np.sin(v)*final_graph[3][6]
z = np.cos(v)*final_graph[3][6]+2
ax.plot_wireframe(x, y, z, color="m")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[3][7]+6
y = np.sin(u)*np.sin(v)*final_graph[3][7]
z = np.cos(v)*final_graph[3][7]+2
ax.plot_wireframe(x, y, z, color="m")

#Quadrant_5
u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[4][0]+6
y = np.sin(u)*np.sin(v)*final_graph[4][0]+6
z = np.cos(v)*final_graph[4][0]+4
ax.plot_wireframe(x, y, z, color="k")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[4][1]+4
y = np.sin(u)*np.sin(v)*final_graph[4][1]+6
z = np.cos(v)*final_graph[4][1]+4
ax.plot_wireframe(x, y, z, color="k")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[4][2]+4
y = np.sin(u)*np.sin(v)*final_graph[4][2]+4
z = np.cos(v)*final_graph[4][2]+4
ax.plot_wireframe(x, y, z, color="k")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[4][3]+6
y = np.sin(u)*np.sin(v)*final_graph[4][3]+4
z = np.cos(v)*final_graph[4][3]+4
ax.plot_wireframe(x, y, z, color="k")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[4][4]+6
y = np.sin(u)*np.sin(v)*final_graph[4][4]+6
z = np.cos(v)*final_graph[4][4]+6
ax.plot_wireframe(x, y, z, color="k")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[4][5]+4
y = np.sin(u)*np.sin(v)*final_graph[4][5]+6
z = np.cos(v)*final_graph[4][5]+6
ax.plot_wireframe(x, y, z, color="k")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[4][6]+4
y = np.sin(u)*np.sin(v)*final_graph[4][6]+4
z = np.cos(v)*final_graph[4][6]+6
ax.plot_wireframe(x, y, z, color="k")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[4][7]+6
y = np.sin(u)*np.sin(v)*final_graph[4][7]+4
z = np.cos(v)*final_graph[4][7]+6
ax.plot_wireframe(x, y, z, color="k")

#Quadrant_6
u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[5][0]+2
y = np.sin(u)*np.sin(v)*final_graph[5][0]+6
z = np.cos(v)*final_graph[5][0]+4
ax.plot_wireframe(x, y, z, color="0.7")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[5][1]+0
y = np.sin(u)*np.sin(v)*final_graph[5][1]+6
z = np.cos(v)*final_graph[5][1]+4
ax.plot_wireframe(x, y, z, color="0.7")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[5][2]+0
y = np.sin(u)*np.sin(v)*final_graph[5][2]+4
z = np.cos(v)*final_graph[5][2]+4
ax.plot_wireframe(x, y, z, color="0.7")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[5][3]+2
y = np.sin(u)*np.sin(v)*final_graph[5][3]+4
z = np.cos(v)*final_graph[5][3]+4
ax.plot_wireframe(x, y, z, color="0.7")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[5][4]+2
y = np.sin(u)*np.sin(v)*final_graph[5][4]+6
z = np.cos(v)*final_graph[5][4]+6
ax.plot_wireframe(x, y, z, color="0.7")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[5][5]+0
y = np.sin(u)*np.sin(v)*final_graph[5][5]+6
z = np.cos(v)*final_graph[5][5]+6
ax.plot_wireframe(x, y, z, color="0.7")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[5][6]+0
y = np.sin(u)*np.sin(v)*final_graph[5][6]+4
z = np.cos(v)*final_graph[5][6]+6
ax.plot_wireframe(x, y, z, color="0.7")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[5][7]+2
y = np.sin(u)*np.sin(v)*final_graph[5][7]+4
z = np.cos(v)*final_graph[5][7]+6
ax.plot_wireframe(x, y, z, color="0.7")

#Quadrant_7
u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[6][0]+2
y = np.sin(u)*np.sin(v)*final_graph[6][0]+2
z = np.cos(v)*final_graph[6][0]+4
ax.plot_wireframe(x, y, z, color="y")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[6][1]+0
y = np.sin(u)*np.sin(v)*final_graph[6][1]+2
z = np.cos(v)*final_graph[6][1]+4
ax.plot_wireframe(x, y, z, color="y")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[6][2]+0
y = np.sin(u)*np.sin(v)*final_graph[6][2]+0
z = np.cos(v)*final_graph[6][2]+4
ax.plot_wireframe(x, y, z, color="y")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[6][3]+2
y = np.sin(u)*np.sin(v)*final_graph[6][3]+0
z = np.cos(v)*final_graph[6][3]+4
ax.plot_wireframe(x, y, z, color="y")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[6][4]+2
y = np.sin(u)*np.sin(v)*final_graph[6][4]+2
z = np.cos(v)*final_graph[6][4]+6
ax.plot_wireframe(x, y, z, color="y")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[6][5]+0
y = np.sin(u)*np.sin(v)*final_graph[6][5]+2
z = np.cos(v)*final_graph[6][5]+6
ax.plot_wireframe(x, y, z, color="y")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[6][6]+0
y = np.sin(u)*np.sin(v)*final_graph[6][6]+0
z = np.cos(v)*final_graph[6][6]+6
ax.plot_wireframe(x, y, z, color="y")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[6][7]+2
y = np.sin(u)*np.sin(v)*final_graph[6][7]+0
z = np.cos(v)*final_graph[6][7]+6
ax.plot_wireframe(x, y, z, color="y")

#Quadrant_8
u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[7][0]+6
y = np.sin(u)*np.sin(v)*final_graph[7][0]+2
z = np.cos(v)*final_graph[7][0]+4
ax.plot_wireframe(x, y, z, color="b")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[7][1]+4
y = np.sin(u)*np.sin(v)*final_graph[7][1]+2
z = np.cos(v)*final_graph[7][1]+4
ax.plot_wireframe(x, y, z, color="b")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[7][2]+4
y = np.sin(u)*np.sin(v)*final_graph[7][2]+0
z = np.cos(v)*final_graph[7][2]+4
ax.plot_wireframe(x, y, z, color="b")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[7][3]+6
y = np.sin(u)*np.sin(v)*final_graph[7][3]+0
z = np.cos(v)*final_graph[7][3]+4
ax.plot_wireframe(x, y, z, color="b")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[7][4]+6
y = np.sin(u)*np.sin(v)*final_graph[7][4]+2
z = np.cos(v)*final_graph[7][4]+6
ax.plot_wireframe(x, y, z, color="b")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[7][5]+4
y = np.sin(u)*np.sin(v)*final_graph[7][5]+2
z = np.cos(v)*final_graph[7][5]+6
ax.plot_wireframe(x, y, z, color="b")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[7][6]+4
y = np.sin(u)*np.sin(v)*final_graph[7][6]+0
z = np.cos(v)*final_graph[7][6]+6
ax.plot_wireframe(x, y, z, color="b")

u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*final_graph[7][7]+6
y = np.sin(u)*np.sin(v)*final_graph[7][7]+0
z = np.cos(v)*final_graph[7][7]+6
ax.plot_wireframe(x, y, z, color="b")

plt.show()
