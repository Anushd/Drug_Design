from numpy import *
from math import *
from sklearn import datasets, svm, metrics 
set_printoptions(threshold='nan')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from decimal import *

#Set precision of decimal calulations
getcontext().prec = 5

###Import_Structures###
structure1 = open("/Users/anush/Desktop/Tris_out.pdb", "r")
structure2 = open("/Users/anush/Desktop/Cysteamine_out.pdb", "r")
structure3 = open("/Users/anush/Desktop/Miglustat_out.pdb", "r")
structure4 = open("/Users/anush/Desktop/Acetic_Acid_out.pdb", "r")
structure5 = open("/Users/anush/Desktop/Pentosan_Polysulfate_out.pdb", "r")
structure6 = open("/Users/anush/Desktop/lsd_out.pdb", "r")
structures_min = [open("/Users/anush/Desktop/Tris_out.pdb", "r"), open("/Users/anush/Desktop/Cysteamine_out.pdb", "r"), open("/Users/anush/Desktop/Miglustat_out.pdb", "r"), open("/Users/anush/Desktop/Acetic_Acid_out.pdb", "r"), open("/Users/anush/Desktop/Pentosan_Polysulfate_out.pdb", "r"), open("/Users/anush/Desktop/lsd_out.pdb", "r")
]

#[binding affinity]
data = [[-4.0],[-2.1],[-4.5],[-2.2],[-7.6]]

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
	if min(minsy) < 0:
		add_y = abs(min(minsy))
	if min(minsz) < 0:
		add_z = abs(min(minsz))
	
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
	model = zeros((ceil(max(maxsx))+1, ceil(max(maxsy))+1, ceil(max(maxsz))+1))
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

def SVM(probs,data,quadrant,to_predict):
	
	label_set = array([probs[0][quadrant],probs[1][quadrant],probs[2][quadrant],probs[3][quadrant],probs[4][quadrant]])
	data_set = array(data)
	
	clf = svm.SVR(kernel='rbf', gamma=1e-3,C=1e7)
	clf.fit(data_set, label_set)  
	
	prediction = clf.predict([to_predict])
	return prediction
		
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

structures_max = [t1,t2,t3,t4,t5,t6]

max1 = Max(structures_max)
max2 = Max(structures_max)
max3 = Max(structures_max)
max4 = Max(structures_max)
max5 = Max(structures_max)
max6 = Max(structures_max)

model1 = Construct(t1[0], t1[1], t1[2], max1)
model2 = Construct(t2[0], t2[1], t2[2], max2)
model3 = Construct(t3[0], t3[1], t3[2], max3)
model4 = Construct(t4[0], t4[1], t4[2], max4)
model5 = Construct(t5[0], t5[1], t5[2], max5)
model6 = Construct(t6[0], t6[1], t6[2], max6)

partition1 = Partition(model1[0])
partition2 = Partition(model2[0])
partition3 = Partition(model3[0])
partition4 = Partition(model4[0])
partition5 = Partition(model5[0])
partition6 = Partition(model6[0])

prob1 = Probability(partition1)
prob2 = Probability(partition2)
prob3 = Probability(partition3)
prob4 = Probability(partition4)
prob5 = Probability(partition5)
prob6 = Probability(partition6)
probs = [prob1,prob2,prob3,prob4,prob5]

print prob6
print ""

quadrant = [0,1,2,3,4,5,6,7]
to_predict = -6.1

finalA = []
for i in quadrant:
	finalA.append(SVM(probs,data,i,to_predict))

print finalA
print ""

#Second iteration on quadrant 5
partition1B = Partition(partition1[4])
partition2B = Partition(partition2[4])
partition3B = Partition(partition3[4])
partition4B = Partition(partition4[4])
partition5B = Partition(partition5[4])
partition6B = Partition(partition6[4])

prob1B = Probability(partition1B)
prob2B = Probability(partition2B)
prob3B = Probability(partition3B)
prob4B = Probability(partition4B)
prob5B = Probability(partition5B)
prob6B = Probability(partition6B)
probsB = [prob1B,prob2B,prob3B,prob4B,prob5B]

print prob6B
print ""

finalB = []
for i in quadrant:
	finalB.append(SVM(probsB,data,i,to_predict))

print finalB
print ""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(model1[1], model1[2], model1[3])
plt.show()
