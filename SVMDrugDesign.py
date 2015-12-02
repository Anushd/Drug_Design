from __future__ import division
from sklearn import datasets, svm, metrics 
import numpy as np
import math

np.set_printoptions(threshold='nan')

# NOTE: Drug Target is Hedgehog Interacting Protein (HHIP)

#####Array_Dimensions#####

dimension = 100
d = dimension/2
	
#####LIGAND_Miglustat#####

def Miglustat1():
	
# Cartesian Atom Coordinates
	a = [26,23,23,26,26,25,24,24,25,28,27,27,27,28,29]
	acar_a = [x+d-1 for x in a]
	b = [17,17,20,21,19,18,18,20,21,19,19,22,24,25,24]
	acar_b = [d-x-1 for x in b]
	c = [14,16,17,15,14,14,15,16,16,13,12,15,15,15,14]
	acar_c = [x+d-1 for x in c]
	atomnum = 15
	masses = [16,16,16,14.01,12.01,12.01,12.01,12.01,12.01,12.01,16,12.01,12.01,12.01,12.01]
	data = [5.8,219.278,-1.2,5,4] # (Binding Affinity, Molecular Weight, logP, H Acceptor Count, H Donor Count)
	return acar_a, acar_b, acar_c, atomnum, masses, data	
	
#####LIGAND_Acarbose#####

def Acarbose1():
	
# Cartesian Atom Coordinates
	a = [28,27,27,25,26,28,25,26,29,30,30,31,28,32,31,33,32,34,33,34,34,35,37,37,38,36,38,36,35,34,27,28,24,23,23,21,21,21,19,22,20,22,24,24]
	acar_a = [x+d-1 for x in a]
	b = [21,22,20,21,19,19,20,18,22,23,24,25,25,25,26,24,23,23,26,27,27,28,28,28,29,27,29,27,28,28,23,24,19,18,17,16,16,18,16,19,18,20,16,15]
	acar_b = [d-x-1 for x in b]
	c = [14,15,13,15,14,12,15,13,13,13,12,11,11,11,10,12,13,13,11,10,9,8,9,11,8,11,11,13,7,6,15,16,15,16,15,16,17,18,18,17,19,18,14,14]
	acar_c = [x+d-1 for x in c]
	atomnum = 44
	masses = [12.01,12.01,12.01,16,12.01,16,12.01,16,16,12.01,12.01,12.01,16,12.01,16,12.01,16,12.01,14.01,12.01,12.01,12.01,12.01,12.01,16,12.01,16,16,12.01,16,12.01,16,16,12.01,12.01,16,12.01,12.01,16,12.01,16,16,12.01,16]
	data = [9.3,645.6048,-7.6,19,14] # (Binding Affinity, Molecular Weight, logP, H Acceptor Count, H Donor Count)
	return acar_a, acar_b, acar_c, atomnum, masses, data

#####LIGAND_Acetic_Acid#####

def Acetic1():
	
# Cartesian Atom Coordinates
	a = [38,38,37,39]
	acar_a = [x+d-1 for x in a]
	b = [28,29,30,30]
	acar_b = [d-x-1 for x in b]
	c = [26,27,27,28]
	acar_c = [x+d-1 for x in c]
	atomnum = 4
	masses = [12.01,12.01,16,16]
	data = [3.2,59.044,-0.22,2,0] # (Binding Affinity, Molecular Weight, logP, H Acceptor Count, H Donor Count)
	return acar_a, acar_b, acar_c, atomnum, masses, data


def Spatial_Dist(points_x, points_y, points_z, atomnum):
# Atom Array
	array_a = np.zeros((dimension, dimension, dimension))
	count = 0
	for i in points_x:
		array_a[i, points_y[count], points_z[count]] = 1
		count += 1

# Atom Densitity Quadrants_1
	count = 0
	q1 = 0
	q2 = 0
	q3 = 0
	q4 = 0
	q5 = 0
	q6 = 0
	q7 = 0
	q8 = 0
	
	r1a = range(0,50)
	r2a = range(50,101)

	r1b = range(0,26)
	r2b = range(26,50)
	r3b = range(50,76)
	r4b = range(76,101)

	for i in points_x:		
# (-,-,-)
		if i in r1a and points_y[count] in r1a and points_z[count] in r1a:
			q1 += 1	
# (+,-,-)		
		if i in r2a and points_y[count] in r1a and points_z[count] in r1a:
			q2 += 1	
# (+,+,-)			
		if i in r2a and points_y[count] in r2a and points_z[count] in r1a:
			q3 += 1				
# (-,+,-)			
		if i in r1a and points_y[count] in r2a and points_z[count] in r1a:
			q4 += 1				
# (-,-,+)			
		if i in r1a and points_y[count] in r1a and points_z[count] in r2a:
			q5 += 1					
# (+,-,+)			
		elif i in r2a and points_y[count] in r1a and points_z[count] in r2a:
			q6 += 1				
# (+,+,+)			
		if i in r2a and points_y[count] in r2a and points_z[count] in r2a:
			q7 += 1			
# (-,+,+)			
		if i in r1a and points_y[count] in r2a and points_z[count] in r2a:
			q8 += 1				
		count += 1
		
	a_densities1 = [q1/atomnum,q2/atomnum,q3/atomnum,q4/atomnum,q5/atomnum,q6/atomnum,q7/atomnum,q8/atomnum]

# Atom Densitity Quadrants_2
	qnum = 0
	a_densities2 = []
	for i in a_densities1:
		q1 = 0
		q2 = 0
		q3 = 0
		q4 = 0
		q5 = 0
		q6 = 0
		q7 = 0
		q8 = 0
		if i > 0 and qnum == 0:	
			count = 0
			q1 = 0
			q2 = 0
			q3 = 0
			q4 = 0
			q5 = 0
			q6 = 0
			q7 = 0
			q8 = 0
			for i in points_x:
	# (-,-,-)	
				if i in r1b and points_y[count] in r1b and points_z[count] in r1b:
					q1 += 1	
	# (+,-,-)		
				if i in r2b and points_y[count] in r1b and points_z[count] in r1b:
					q2 += 1	
	# (+,+,-)			
				if i in r2b and points_y[count] in r2b and points_z[count] in r1b:
					q3 += 1	
	# (-,+,-)			
				if i in r1b and points_y[count] in r2b and points_z[count] in r1b:
					q4 += 1		
	# (-,-,+)			
				if i in r1b and points_y[count] in r1b and points_z[count] in r2b:
					q5 += 1	
	# (+,-,+)			
				if i in r2b and points_y[count] in r1b and points_z[count] in r2b:
					q6 += 1				
	# (+,+,+)			
				if i in r2b and points_y[count] in r2b and points_z[count] in r2b:
					q7 += 1			
	# (-,+,+)			
				if i in r1b and points_y[count] in r2b and points_z[count] in r2b:
					q8 += 1
			count +=1

		
		if i > 0 and qnum == 1:
			count = 0
			q1 = 0
			q2 = 0
			q3 = 0
			q4 = 0
			q5 = 0
			q6 = 0
			q7 = 0
			q8 = 0
			for i in points_x:
	# (-,-,-)	
				if i in r3b and points_y[count] in r1b and points_z[count] in r1b:
					q1 += 1	
	# (+,-,-)		
				if i in r4b and points_y[count] in r1b and points_z[count] in r1b:
					q2 += 1	
	# (+,+,-)			
				if i in r4b and points_y[count] in r2b and points_z[count] in r1b:
					q3 += 1	
	# (-,+,-)			
				if i in r3b and points_y[count] in r2b and points_z[count] in r1b:
					q4 += 1		
	# (-,-,+)			
				if i in r3b and points_y[count] in r1b and points_z[count] in r2b:
					q5 += 1	
	# (+,-,+)			
				if i in r4b and points_y[count] in r1b and points_z[count] in r2b:
					q6 += 1				
	# (+,+,+)			
				if i in r4b and points_y[count] in r2b and points_z[count] in r2b:
					q7 += 1			
	# (-,+,+)			
				if i in r3b and points_y[count] in r2b and points_z[count] in r2b:
					q8 += 1
			count +=1
				
		if i > 0 and qnum == 2:
			count = 0
			q1 = 0
			q2 = 0
			q3 = 0
			q4 = 0
			q5 = 0
			q6 = 0
			q7 = 0
			q8 = 0
			for i in points_x:
	# (-,-,-)	
				if i in r3b and points_y[count] in r3b and points_z[count] in r1b:
					q1 += 1	
	# (+,-,-)		
				if i in r4b and points_y[count] in r3b and points_z[count] in r1b:
					q2 += 1	
	# (+,+,-)			
				if i in r4b and points_y[count] in r4b and points_z[count] in r1b:
					q3 += 1	
	# (-,+,-)			
				if i in r3b and points_y[count] in r4b and points_z[count] in r1b:
					q4 += 1		
	# (-,-,+)			
				if i in r3b and points_y[count] in r3b and points_z[count] in r2b:
					q5 += 1	
	# (+,-,+)			
				if i in r4b and points_y[count] in r3b and points_z[count] in r2b:
					q6 += 1				
	# (+,+,+)			
				if i in r4b and points_y[count] in r4b and points_z[count] in r2b:
					q7 += 1			
	# (-,+,+)			
				if i in r3b and points_y[count] in r4b and points_z[count] in r2b:
					q8 += 1
				count +=1
	
		if i > 0 and qnum == 3:		
			count = 0
			q1 = 0
			q2 = 0
			q3 = 0
			q4 = 0
			q5 = 0
			q6 = 0
			q7 = 0
			q8 = 0
			for i in points_x:
	# (-,-,-)	
				if i in r1b and points_y[count] in r3b and points_z[count] in r1b:
					q1 += 1	
	# (+,-,-)		
				if i in r2b and points_y[count] in r3b and points_z[count] in r1b:
					q2 += 1	
	# (+,+,-)			
				if i in r2b and points_y[count] in r4b and points_z[count] in r1b:
					q3 += 1	
	# (-,+,-)			
				if i in r1b and points_y[count] in r4b and points_z[count] in r1b:
					q4 += 1		
	# (-,-,+)			
				if i in r1b and points_y[count] in r3b and points_z[count] in r2b:
					q5 += 1	
	# (+,-,+)			
				if i in r2b and points_y[count] in r3b and points_z[count] in r2b:
					q6 += 1				
	# (+,+,+)			
				if i in r2b and points_y[count] in r4b and points_z[count] in r2b:
					q7 += 1			
	# (-,+,+)			
				if i in r1b and points_y[count] in r4b and points_z[count] in r2b:
					q8 += 1
			count +=1
				
		if i > 0 and qnum == 4:
			count = 0
			q1 = 0
			q2 = 0
			q3 = 0
			q4 = 0
			q5 = 0
			q6 = 0
			q7 = 0
			q8 = 0
			for i in points_x:
	# (-,-,-)	
				if i in r1b and points_y[count] in r1b and points_z[count] in r3b:
					q1 += 1	
	# (+,-,-)		
				if i in r2b and points_y[count] in r1b and points_z[count] in r3b:
					q2 += 1	
	# (+,+,-)			
				if i in r2b and points_y[count] in r2b and points_z[count] in r3b:
					q3 += 1	
	# (-,+,-)			
				if i in r1b and points_y[count] in r2b and points_z[count] in r3b:
					q4 += 1		
	# (-,-,+)			
				if i in r1b and points_y[count] in r1b and points_z[count] in r4b:
					q5 += 1	
	# (+,-,+)			
				if i in r2b and points_y[count] in r1b and points_z[count] in r4b:
					q6 += 1				
	# (+,+,+)			
				if i in r2b and points_y[count] in r2b and points_z[count] in r4b:
					q7 += 1			
	# (-,+,+)			
				if i in r1b and points_y[count] in r2b and points_z[count] in r4b:
					q8 += 1
				count +=1
		
		if i > 0 and qnum == 5:
			count = 0
			q1 = 0
			q2 = 0
			q3 = 0
			q4 = 0
			q5 = 0
			q6 = 0
			q7 = 0
			q8 = 0
			for i in points_x:
	# (-,-,-)	
				if i in r3b and points_y[count] in r1b and points_z[count] in r3b:
					q1 += 1
	# (+,-,-)		
				if i in r4b and points_y[count] in r1b and points_z[count] in r3b:
					q2 += 1	
	# (+,+,-)			
				if i in r4b and points_y[count] in r2b and points_z[count] in r3b:
					q3 += 1	
	# (-,+,-)			
				if i in r3b and points_y[count] in r2b and points_z[count] in r3b:
					q4 += 1		
	# (-,-,+)			
				if i in r3b and points_y[count] in r1b and points_z[count] in r4b:
					q5 += 1	
	# (+,-,+)			
				if i in r4b and points_y[count] in r1b and points_z[count] in r4b:
					q6 += 1				
	# (+,+,+)			
				if i in r4b and points_y[count] in r2b and points_z[count] in r4b:
					q7 += 1			
	# (-,+,+)			
				if i in r3b and points_y[count] in r2b and points_z[count] in r4b:
					q8 += 1
				count +=1
				
		if i > 0 and qnum == 6:	
			count = 0
			q1 = 0
			q2 = 0
			q3 = 0
			q4 = 0
			q5 = 0
			q6 = 0
			q7 = 0
			q8 = 0
			for i in points_x:
	# (-,-,-)	
				if i in r3b and points_y[count] in r3b and points_z[count] in r3b:
					q1 += 1	
	# (+,-,-)		
				if i in r4b and points_y[count] in r3b and points_z[count] in r3b:
					q2 += 1	
	# (+,+,-)			
				if i in r4b and points_y[count] in r4b and points_z[count] in r3b:
					q3 += 1	
	# (-,+,-)			
				if i in r3b and points_y[count] in r4b and points_z[count] in r3b:
					q4 += 1		
	# (-,-,+)			
				if i in r3b and points_y[count] in r3b and points_z[count] in r4b:
					q5 += 1	
	# (+,-,+)			
				if i in r4b and points_y[count] in r3b and points_z[count] in r4b:
					q6 += 1				
	# (+,+,+)			
				if i in r4b and points_y[count] in r3b and points_z[count] in r4b:
					q7 += 1			
	# (-,+,+)			
				if i in r3b and points_y[count] in r4b and points_z[count] in r4b:
					q8 += 1
				count +=1

		if i > 0 and qnum == 7:	
			count = 0
			q1 = 0
			q2 = 0
			q3 = 0
			q4 = 0
			q5 = 0
			q6 = 0
			q7 = 0
			q8 = 0
			for i in points_x:
	# (-,-,-)	
				if i in r1b and points_y[count] in r3b and points_z[count] in r3b:
					q1 += 1	
	# (+,-,-)		
				if i in r2b and points_y[count] in r3b and points_z[count] in r3b:
					q2 += 1	
	# (+,+,-)			
				if i in r2b and points_y[count] in r4b and points_z[count] in r3b:
					q3 += 1	
	# (-,+,-)			
				if i in r1b and points_y[count] in r4b and points_z[count] in r3b:
					q4 += 1		
	# (-,-,+)			
				if i in r1b and points_y[count] in r3b and points_z[count] in r4b:
					q5 += 1	
	# (+,-,+)			
				if i in r2b and points_y[count] in r3b and points_z[count] in r4b:
					q6 += 1				
	# (+,+,+)			
				if i in r2b and points_y[count] in r3b and points_z[count] in r4b:
					q7 += 1			
	# (-,+,+)			
				if i in r1b and points_y[count] in r4b and points_z[count] in r4b:
					q8 += 1
				count +=1		
		a_densities2.append([q1/atomnum,q2/atomnum,q3/atomnum,q4/atomnum,q5/atomnum,q6/atomnum,q7/atomnum,q8/atomnum])
		qnum +=1	
	a_densities1 = np.array(a_densities1)
	a_densities2 = np.array(a_densities2)
	return a_densities1.flatten(order='C'), a_densities2.flatten(order='C')

# Molecular Weight Array
def Mass_Dist(points_x,points_y,points_z,masses):
	array_m = np.zeros((dimension, dimension, dimension))
	count = 0
	for i in points_x:
			array_m[i, points_y[count], points_z[count]] = masses[count]
			count += 1
	array_m = np.array(array_m)
	return array_m.flatten(order='C')		 

#####SVM_Training#####

def SVM_a():
	
	level1 = []
	count = 0
	while count<8:
		labels = np.array([label1a[count],label1b[count],label1c[count]])
		data = np.array([f1,f2,f3])
		clf = svm.SVR(kernel='rbf', gamma=1e-3,C=1e7)
		clf.fit(data, labels)  
		level1.append(clf.predict(predict))
		count +=1
	
	level2 = []
	count = 0
	while count<64:
		labels = np.array([label2a[count],label2b[count],label2c[count]])
		data = np.array([f1,f2,f3])
		clf = svm.SVR(kernel='rbf', gamma=1e-3,C=1e7)
		clf.fit(data, labels)  
		level2.append(clf.predict(predict))
		count +=1
	return level1, level2
	
def SVM_m():	
	masses = []
	count = 0
	while count<1000000:
		labels = np.array([label3a[count],label3b[count],label3c[count]])
		data = np.array([f1,f2,f3])
		clf = svm.SVR(kernel='rbf', gamma=1e-3,C=1e7)
		clf.fit(data, labels)  
		masses.append(clf.predict(predict))
		count +=1
	return masses

#####Calling_Functions#####

a1,b1,c1,d1,e1,f1 = Acetic1()
a2,b2,c2,d2,e2,f2 = Acarbose1()
a3,b3,c3,d3,e3,f3 = Miglustat1()
label1a,label2a = Spatial_Dist(a1,b1,c1,d1)
label1b,label2b = Spatial_Dist(a2,b2,c2,d2)
label1c,label2c = Spatial_Dist(a3,b3,c3,d3)
label3a = Mass_Dist(a1,b1,c1,e1)
label3b = Mass_Dist(a2,b2,c2,e2)
label3c = Mass_Dist(a3,b3,c3,e3)

# Prediction Parameters
predict = [3.2,59.044,-0.22,2,0]
level1,level2 = SVM_a()	

from visual import *
dim= [0,0,0,0,0,0,0,0]
l = [0,0,0,0,0,0,0,0]
dim1= [0,0,0,0,0,0,0,0]

def Model():
	count = 0
	for i in level1:
		
		if i>0:
			dim[count] = i*13
			l[count] = i*20
		elif i==0:
			dim[count] = 0.25
			l[count] = 10
		count +=1
		
	ell= ellipsoid( pos=(2.5,-10,2.5), length=l[1],width=dim[1],height=dim[1],color=(0.5,0.2,0),opacity=0.6)
	ell.rotate(angle=3.14/4,axis=(0,0,-1))
	
	ell= ellipsoid( pos=(-12,4,2.5), length=l[7],width=dim[7],height=dim[7],color=(0.5,0.2,0),opacity=0.6)
	ell.rotate(angle=3.14/4,axis=(0,0,-1))

	ell= ellipsoid( pos=(2.5,4,2.5), length=l[5],width=dim[5],height=dim[5],color=(0,0.3,0.5),opacity=0.6)
	ell.rotate(angle=3.14/4,axis=(0,0,1))

	ell= ellipsoid( pos=(-12,-10,2.5), length=l[3],width=dim[3],height=dim[3],color=(0,0.3,0.5),opacity=0.6)
	ell.rotate(angle=radians(135),axis=(0,0,-1))

	ell= ellipsoid( pos=(-5,-10,-5), length=l[3],width=dim[2],height=dim[2],color=(0,0.5,0.5),opacity=0.6)
	ell.rotate(angle=radians(90),axis=(0,0,-1))
	ell.rotate(angle=radians(45),axis=(1,0,0))

	ell= ellipsoid( pos=(-5,-10,10), length=l[0],width=dim[0],height=dim[0],color=(0.5,0.5,0),opacity=0.6)
	ell.rotate(angle=radians(90),axis=(0,0,-1))
	ell.rotate(angle=radians(45),axis=(-1,0,0))

	ell= ellipsoid( pos=(-4.5,4,-5), length=l[6],width=dim[6],height=dim[6],color=(0.5,0.5,0),opacity=0.6)
	ell.rotate(angle=radians(90),axis=(0,0,-1))
	ell.rotate(angle=radians(45),axis=(-1,0,0))

	ell= ellipsoid( pos=(-5,3.5,9), length=l[4],width=dim[4],height=dim[4],color=(0,0.5,0.5),opacity=0.6)
	ell.rotate(angle=radians(90),axis=(0,0,-1))
	ell.rotate(angle=radians(45),axis=(1,0,0))	
	
def Model1():	
	count=0
	for i in level2:
		if count<8:
			if i>0:
				dim1[count] = i*13
			elif i==0:
				dim1[count] = 1
			count +=1
	
Model()