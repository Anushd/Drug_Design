from numpy import *
import numpy as np
from math import *
from sklearn import datasets, svm, metrics 
set_printoptions(threshold='nan')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from decimal import *
from itertools import product, combinations

#Set precision of decimal calulations
getcontext().prec = 5

###Import_Structures###
structure1 = open("/Users/anush/Desktop/Final_Ligands/713182_out.pdb", "r")
structure2 = open("/Users/anush/Desktop/Final_Ligands/6605027_out.pdb", "r")
structure3 = open("/Users/anush/Desktop/Final_Ligands/5389521_out.pdb", "r")
structure4 = open("/Users/anush/Desktop/Final_Ligands/3247284_out.pdb", "r")
structure5 = open("/Users/anush/Desktop/Final_Ligands/3236943_out.pdb", "r")
structure6 = open("/Users/anush/Desktop/Final_Ligands/3236517_out.pdb", "r")
structure7 = open("/Users/anush/Desktop/Final_Ligands/654602_out.pdb", "r")
structure8 = open("/Users/anush/Desktop/Final_Ligands/653221_out.pdb", "r")
structure9 = open("/Users/anush/Desktop/Final_Ligands/8548995_out.pdb", "r")
structure10 = open("/Users/anush/Desktop/Final_Ligands/49786375_out.pdb", "r")
structure11 = open("/Users/anush/Desktop/Final_Ligands/4911035_out.pdb", "r")
structure12 = open("/Users/anush/Desktop/Final_Ligands/3247293_out.pdb", "r")
structure13 = open("/Users/anush/Desktop/Final_Ligands/3147229_out.pdb", "r")
structure14 = open("/Users/anush/Desktop/Final_Ligands/2982527_out.pdb", "r")
structure15 = open("/Users/anush/Desktop/Final_Ligands/2508673_out.pdb", "r")
structure16 = open("/Users/anush/Desktop/Final_Ligands/16231644_out.pdb", "r")
structure17 = open("/Users/anush/Desktop/Final_Ligands/16031807_out.pdb", "r")

structures_min = [open("/Users/anush/Desktop/Final_Ligands/713182_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/6605027_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/5389521_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3247284_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3236943_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3236517_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/654602_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/653221_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/8548995_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/49786375_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/4911035_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3247293_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/3147229_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/2982527_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/2508673_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/16231644_out.pdb", "r"),open("/Users/anush/Desktop/Final_Ligands/16031807_out.pdb", "r")]

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
	
	label_set = array([probs[0][quadrant],probs[1][quadrant],probs[2][quadrant],probs[3][quadrant],probs[4][quadrant],probs[5][quadrant],probs[6][quadrant],probs[7][quadrant],probs[8][quadrant],probs[9][quadrant],probs[10][quadrant],probs[11][quadrant],probs[12][quadrant],probs[13][quadrant],probs[14][quadrant],probs[15][quadrant],probs[16][quadrant]])
	data_set = array(data)
	
	clf = svm.SVR(kernel='rbf', gamma=1e-3,C=1e7)
	clf.fit(data_set, label_set)  
	
	prediction = clf.predict([to_predict])
	return prediction

###Run_Functions###		
min = Min(structures_min)
s1 = Coords(structure1)
t1 = Translate(min[0], min[1], min[2], s1[0], s1[1], s1[2])
s2 = Coords(structure2)
t2 = Translate(min[0], min[1], min[2], s2[0], s2[1], s2[2])
s3 = Coords(structure3)
t3 = Translate(min[0], min[1], min[2], s3[0], s3[1], s3[2])
s4 = Coords(structure4)
t4 = Translate(min[0], min[1], min[2], s4[0], s4[1], s4[2])
s5 = Coords(structure5)
t5 = Translate(min[0], min[1], min[2], s5[0], s5[1], s5[2])
s6 = Coords(structure6)
t6 = Translate(min[0], min[1], min[2], s6[0], s6[1], s6[2])
s7 = Coords(structure7)
t7 = Translate(min[0], min[1], min[2], s7[0], s7[1], s7[2])
s8 = Coords(structure8)
t8 = Translate(min[0], min[1], min[2], s8[0], s8[1], s8[2])
s9 = Coords(structure9)
t9 = Translate(min[0], min[1], min[2], s9[0], s9[1], s9[2])
s10 = Coords(structure10)
t10 = Translate(min[0], min[1], min[2], s10[0], s10[1], s10[2])
s11 = Coords(structure11)
t11 = Translate(min[0], min[1], min[2], s11[0], s11[1], s11[2])
s12 = Coords(structure12)
t12 = Translate(min[0], min[1], min[2], s12[0], s12[1], s12[2])
s13 = Coords(structure13)
t13 = Translate(min[0], min[1], min[2], s13[0], s13[1], s13[2])
s14 = Coords(structure14)
t14 = Translate(min[0], min[1], min[2], s14[0], s14[1], s14[2])
s15 = Coords(structure15)
t15 = Translate(min[0], min[1], min[2], s15[0], s15[1], s15[2])
s16 = Coords(structure16)
t16 = Translate(min[0], min[1], min[2], s16[0], s16[1], s16[2])
s17 = Coords(structure17)
t17 = Translate(min[0], min[1], min[2], s17[0], s17[1], s17[2])

structures_max = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17]

max1 = Max(structures_max)
max2 = max1
max3 = max1
max4 = max1
max5 = max1
max6 = max1
max7 = max1
max8 = max1
max9 = max1
max10 = max1
max11 = max1
max12 = max1
max13 = max1
max14 = max1
max15 = max1
max16 = max1
max17 = max1
model1 = Construct(t1[0], t1[1], t1[2], max1)
model2 = Construct(t2[0], t2[1], t2[2], max2)
model3 = Construct(t3[0], t3[1], t3[2], max3)
model4 = Construct(t4[0], t4[1], t4[2], max4)
model5 = Construct(t5[0], t5[1], t5[2], max5)
model6 = Construct(t6[0], t6[1], t6[2], max6)
model7 = Construct(t7[0], t7[1], t7[2], max7)
model8 = Construct(t8[0], t8[1], t8[2], max8)
model9 = Construct(t9[0], t9[1], t9[2], max9)
model10 = Construct(t10[0], t10[1], t10[2], max10)
model11 = Construct(t11[0], t11[1], t11[2], max11)
model12 = Construct(t12[0], t12[1], t12[2], max12)
model13 = Construct(t13[0], t13[1], t13[2], max13)
model14 = Construct(t14[0], t14[1], t14[2], max14)
model15 = Construct(t15[0], t15[1], t15[2], max15)
model16 = Construct(t16[0], t16[1], t16[2], max16)
model17 = Construct(t17[0], t17[1], t17[2], max17)

partition1 = Partition(model1[0])
partition2 = Partition(model2[0])
partition3 = Partition(model3[0])
partition4 = Partition(model4[0])
partition5 = Partition(model5[0])
partition6 = Partition(model6[0])
partition7 = Partition(model7[0])
partition8 = Partition(model8[0])
partition9 = Partition(model9[0])
partition10 = Partition(model10[0])
partition11 = Partition(model11[0])
partition12 = Partition(model12[0])
partition13 = Partition(model13[0])
partition14 = Partition(model14[0])
partition15 = Partition(model15[0])
partition16 = Partition(model16[0])
partition17 = Partition(model17[0])

prob1 = Probability(partition1)
prob2 = Probability(partition2)
prob3 = Probability(partition3)
prob4 = Probability(partition4)
prob5 = Probability(partition5)
prob6 = Probability(partition6)
prob7 = Probability(partition7)
prob8 = Probability(partition8)
prob9 = Probability(partition9)
prob10 = Probability(partition10)
prob11 = Probability(partition11)
prob12 = Probability(partition12)
prob13 = Probability(partition13)
prob14 = Probability(partition14)
prob15 = Probability(partition15)
prob16 = Probability(partition16)
prob17 = Probability(partition17)

probs = [prob1,prob2,prob3,prob4,prob5,prob6,prob7,prob8,prob9,prob10,prob11,prob12,prob13,prob14,prob15,prob16,prob17]

quadrant = [0,1,2,3,4,5,6,7]
to_predict = -6.0

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

#Second iteration on quadrant 5
quad_count=0
final_graph=[]
for i in quadrant:
	print "Quadrant:"
	print i+1
	print
	partition1b = Partition(partition1[i])
	partition2b = Partition(partition2[i])
	partition3b = Partition(partition3[i])
	partition4b = Partition(partition4[i])
	partition5b = Partition(partition5[i])
	partition6b = Partition(partition6[i])
	partition7b = Partition(partition7[i])
	partition8b = Partition(partition8[i])
	partition9b = Partition(partition9[i])
	partition10b = Partition(partition10[i])
	partition11b = Partition(partition11[i])
	partition12b = Partition(partition12[i])
	partition13b = Partition(partition13[i])
	partition14b = Partition(partition14[i])
	partition15b = Partition(partition15[i])
	partition16b = Partition(partition16[i])
	partition17b = Partition(partition17[i])

	prob1b = Probability(partition1b)
	prob2b = Probability(partition2b)
	prob3b = Probability(partition3b)
	prob4b = Probability(partition4b)
	prob5b = Probability(partition5b)
	prob6b = Probability(partition6b)
	prob7b = Probability(partition7b)
	prob8b = Probability(partition8b)
	prob9b = Probability(partition9b)
	prob10b = Probability(partition10b)
	prob11b = Probability(partition11b)
	prob12b = Probability(partition12b)
	prob13b = Probability(partition13b)
	prob14b = Probability(partition14b)
	prob15b = Probability(partition15b)
	prob16b = Probability(partition16b)
	prob17b = Probability(partition17b)
	
	probsb = [prob1b,prob2b,prob3b,prob4b,prob5b,prob6b,prob7b,prob8b,prob9b,prob10b,prob11b,prob12b,prob13b,prob14b,prob15b,prob16b,prob17b]	

	finalB = []
	for i in quadrant:
		finalB.append(SVM(probsb,data,i,to_predict))
	final_graph.append(finalB)
	print "Predicted:"
	print finalB
	print
	print 

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(model1[1], model1[2], model1[3])
plt.show()'''

###Plot_Probabilities###
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

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
