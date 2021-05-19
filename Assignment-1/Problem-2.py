#import all necessary library.
from gurobipy import *

#Given a time requirements (in hours), for doctors to attend patients at ER at the hospital.
#Dictionary U indicates the value of each box in table. 
U = [
	[69,86,55,98,76,115],
	[99,89,59,108,132,78],
	[116,69,79,140,82,93],
	[130,95,75,96,120,122],
	[118,83,76,97,130,132],
	[67,137,89,96,50,63]
]

#Here we will start by taking one 6x6 dictionary with inital values 0 for each box.
V = [
	[0,0,0,0,0,0],
	[0,0,0,0,0,0],
	[0,0,0,0,0,0],
	[0,0,0,0,0,0],
	[0,0,0,0,0,0],
	[0,0,0,0,0,0]
]

#create a model
m = Model("Optimal assignment of patients to doctors in the entire Table")

#Assign a binary variable to selected value.
#For the each box of table, I am assigning the new variable and if it is minimum then assign that doctor to patient.
#Print 1 if doctor is assigned to that patient or print 0.
for i in range (1,len(U)+1):
	for j in range(1,len(U)+1):
		V[i-1][j-1] = m.addVar(lb=0, ub= GRB.INFINITY, vtype = GRB.BINARY)
		
#Adding the Objective function.
#Objective: selected grids should give the minimized summation of the values.
object=0
for i in range(1,len(U)+1):
	for j in range(1,len(U)+1):
		object += U[i-1][j-1] * V[i-1][j-1]
m.setObjective(object, GRB.MINIMIZE)

#Adding the Constraints:each patient should be visited by each doctor and each doctor will assigned to one particular patient. 
#no more than one patient should be selected.
#which means each row should have only 1 selected value. 
for i in range (1,len(U)+1):
	constraint = 0
	for j in range(1,len(U)+1):
		constraint += V[i-1][j-1]
	m.addConstr(constraint == 1)
#no more than one doctor should be selected.
#which means each column should have only 1 selected value.
for i in range (1,len(U)+1):
	constraint = 0
	for j in range(1,len(U)+1):
		constraint += V[j-1][i-1]
	m.addConstr(constraint == 1)

#optimize model	
m.optimize()

#Print the whole matrix by denoting which value of respective row or column has been selected.	
for i in range (1,len(U)+1):
	l = []
	for j in range(1,len(U)+1):
		l.append(V[i-1][j-1].x*U[i-1][j-1])
	print(l)
	
# print final minimum sum of selected doctors for each patients.
print("sum of the selected doctors for each patients is:",m.objVal)