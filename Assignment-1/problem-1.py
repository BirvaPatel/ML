#import all necessary libraries.
from gurobipy import *

#create a model
m = Model("Assignment of Patient 1 and Patient 2 to Doctor 4 and Doctor 5")

#Add the binary variables
x1 = m.addVar(lb=0, ub= GRB.INFINITY, vtype = GRB.BINARY, name="patient-1 assigned to doctor-4 is -X1")
x2 = m.addVar(lb=0, ub= GRB.INFINITY, vtype = GRB.BINARY, name="patient-2 assigned to doctor-4 is -X2")
x3 = m.addVar(lb=0, ub= GRB.INFINITY, vtype = GRB.BINARY, name="patient-1 assigned to doctor-5 is -X3")
x4 = m.addVar(lb=0, ub= GRB.INFINITY, vtype = GRB.BINARY, name="patient-2 assigned to doctor-5 is -X4")

#Objective function: In this case, hours required by doctor-4 and doctor-5 will sumed up for both patients.
m.setObjective((130*x1 + 95*x2)+(118*x3 + 83*x4), GRB.MINIMIZE)

#Adding the constraints.
#1. from x1 and x2, only one value should be choosen for doctor-4 as each doctor should assigned to only one patient.
#2. from x3 and x4, only one value should be choosen for doctor-5 as each doctor should assigned to only one patient.
#3. from x1 and x3, only one value should be choosen as each patient should assigned to only one doctor.
#4. from x2 and x4, only one value should be choosen as each patient should assigned to only one doctor.
#5. sum of x1 and x2 value will be less than 225.
#6. sum of x3 and x4 value will be less than 201.
m.addConstr(x1+x2>=1, "constr1")
m.addConstr(x3+x4>=1, "constr2")
m.addConstr(x1+x3>=1, "constr3")
m.addConstr(x2+x4>=1, "constr4")
m.addConstr(130*x1 + 95*x2 <= 225, "constr5")
m.addConstr(118*x3 + 83*x4 <= 201, "constr6")

#optimize model
m.optimize()

#Print 1 if doctor is assigned to that patient or print 0
for var in m.getVars():
	print(var.varName,":",var.x)
	
#Print total minimum hours required by doctors
print("Total minimum value:",m.objVal)