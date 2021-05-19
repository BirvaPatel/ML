## Problem:
Write a cuda code to multipy the metrix of different dimention.</br>
A 1* B = C1</br>
A 2* B = C2</br>
.</br>
.</br>
An * B = Cn</br>
Where A range from 1 to n and B is static.</br>
#### CONDITION -1 :</br>
Dim for A = 500 * 500</br>
Dim for B = 500 * 400</br>
So Dim of C will be = 500* 400</br>
And number of iterations are 100.</br>
#### CONDITION -2 :</br>
Dim for A = 50 * 20</br>
Dim for B = 20 * 50</br>
So Dim of C will be = 50* 50</br>
And number of iterations are 5000.</br>
#### CONDITION -3 :</br>
Dim for A = 6 * 4000</br>
Dim for B = 4000 * 9</br>
So Dim of C will be = 6* 9</br>
And number of iterations are 1000.</br>

## Results:
#### Runtime for each condition is as follows:
Condition-1 : 26144.99 ms</br>
Condition-2 : 3110.16 ms</br>
Condition-3 : 1481.38 ms</br>

