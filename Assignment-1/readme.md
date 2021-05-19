## Problem:
There are 6 new patients at an Emergency Room (ER) at a busy hospital. Each patient needs to be addressed by any one of its six available doctors. 
Because of the differences in symptoms of the patients as well as the expertise and experience of the doctors, the doctors require spending varying amounts of 
time on attending each patient. The senior doctor of the ER has estimated the time requirements as shown in Table 1.
![alt text](https://github.com/BirvaPatel/ML/blob/main/Assignment-1/Table.PNG)
1. Formulate an optimal assignment of Patient 1 and Patient 2 to Doctor 4 and Doctor 5 in such a way that each doctor receives a different patient and the total hours spent 
by the doctors is minimized.
2. Fill the empty cells in Table 1 with random integer values with your assumptions (e.g., between 50 and 150). Formulate an optimal assignment of patients to 
doctors in the entire Table 1 such that each doctor receives a different patient and the total time expended by the ER is minimized.
And finally Write a Gurobi script to model the problem in part 1 and part 2

## Solution Part-1:
#### Variables:
Take each box-value as a variable so there will be 4 variables x1,x2,x3 and x4.</br>
Doctor-4 - 130*x1 + 95*x2</br>
Doctor-5 - 118*x3 + 83*x4
#### Objective Function:
The final objective function would be summation of patient selected by doctor-4 and patient selected by doctor-5.</br>
Objective: ((130*x1 + 95*x2)+(118*x3 + 83*x4))
#### Constraints:
Here we have some conditions to satisfy:
1. No more than one patient should be selected. which means each row should have only 1 selected value.
2. No more than one doctor should be selected. which means each column should have only 1 selected value.
## Solution Part-2:
table with randomly selected values,</br>
![alt text](https://github.com/BirvaPatel/ML/blob/main/Assignment-1/Selectedtable.PNG)
According to my algorithm, it checks all possibilities to minimize the sum and no doctor is assigned to two or more patients or no patient is assigned to more than two doctors.</br>
Same logic as first question, but here I can not define all the variables independently, so I used for loops to satisfy all the conditions.</br>
for example:</br>
Case-1:</br>
Row-1 select 55 (column-3 selected)</br>
Row 2 select 78 (59 is not possible-it has already selected by first doctor) (column-6 selected)</br>
Row-3 select 69(column-2 selected)</br>
Row-4 select 96(column-4 selected)</br>
Row-5 select 118(column-1 selected)</br>
Row-6 select 50(column-5 selected)</br>
Sum : 55+78+69+96+118+50 = 466</br>
Case-2 :</br>
Row-1 select 69 (column-1 selected)</br>
Row 2 select 78 (column-6 selected)</br>
Row-3 select 69(column-2 selected)</br>
Row-4 select 96(column-4 selected)</br>
Row-5 select 76(column-3 selected)</br>
Row-6 select 50(column-5 selected)</br>
Sum : 69+78+69+96+76+50 = 438</br>

## Output:
![alt text](https://github.com/BirvaPatel/ML/blob/main/Assignment-1/Output-problem-1.png)

![alt text](https://github.com/BirvaPatel/ML/blob/main/Assignment-1/Output-problem-2.png)
