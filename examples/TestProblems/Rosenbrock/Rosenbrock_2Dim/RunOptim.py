import csv
import numpy as np
from FADO import *
import matplotlib.pyplot as plt
import pandas as pd

# Define a callback function
def store_data(xk):

    # Initialize a list to store the iteration values
    #iteration_values = []
    #with open('iteration_results.csv', 'a', newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    #    writer.writerow(xk[0], xk[1])
    # Get the gradient norm
    Gnorm = np.linalg.norm(driver.grad(xk))
    with open('optimality.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([Gnorm])

#def postProcess():
    
   
# Design variables of the problem
# this defines initial value and how they are written to an arbitrary file


var1 = InputVariable(1.0,LabelReplacer("__X__"))
var2 = InputVariable(2.0,LabelReplacer("__Y__"))

# Parameters
# these parameters tailor the template config to each function
parData1 = Parameter(["data1.txt"],LabelReplacer("__DATA_FILE__"))
parData2 = Parameter(["data2.txt"],LabelReplacer("__DATA_FILE__"))
parFunc1 = Parameter(["rosenbrock"],LabelReplacer("__FUNCTION__"))
parFunc2 = Parameter(["constraint"],LabelReplacer("__FUNCTION__"))

# Evaluations
# "runs" that are needed to compute functions and their gradients
evalFun1 = ExternalRun("Direct","python ../../direct.py config_tmpl.txt")
evalFun1.addConfig("config_tmpl.txt")
evalFun1.addData("data1.txt")
evalFun1.addParameter(parData1)

evalJac1 = ExternalRun("Adjoint","python ../../adjoint.py config_tmpl.txt")
evalJac1.addConfig("config_tmpl.txt")
evalJac1.addData("data1.txt")
evalJac1.addData("Direct/results.txt") # simulate we need data from the direct run
evalJac1.addParameter(parData1)
evalJac1.addParameter(parFunc1)

#----Define constraints
evalJacCon1 = ExternalRun("Geometry","python ../../adjoint.py config_tmpl.txt")
evalJacCon1.addConfig("config_tmpl.txt")
evalJacCon1.addData("data1.txt")
evalJacCon1.addParameter(parData1)
evalJacCon1.addParameter(parFunc2)



# Functions
# now variables, parameters, and evaluations are combined
fun1 = Function("Rosenbrock1","Direct/results.txt",TableReader(0,0))
fun1.addInputVariable(var1,"Adjoint/gradient.txt",TableReader(0,0))
fun1.addInputVariable(var2,"Adjoint/gradient.txt",TableReader(1,0))
fun1.addValueEvalStep(evalFun1)
fun1.addGradientEvalStep(evalJac1)


Con1 = Function("Constraint","Direct/results.txt",TableReader(1,0))  # Scalar value of the contraint
Con1.addInputVariable(var1,"Geometry/gradient.txt",TableReader(0,0)) # First partial for constraint
Con1.addInputVariable(var2,"Geometry/gradient.txt",TableReader(1,0)) # Second partial for contraint
Con1.addValueEvalStep(evalFun1)
Con1.addGradientEvalStep(evalJacCon1)

# Driver settingsx
driver = ScipyDriver()
driver.addObjective("min", fun1, 1e-04, 1)

#driver.addUpperBound(Con1,0.001)

driver.setWorkingDirectory("OPTIM")
driver.setEvaluationMode(False,2.0)
driver.setStorageMode(True,"DSN_")
driver.setFailureMode("HARD")


his = open("history.csv","w",1)
driver.setHistorian(his)

class exampleAction:
    def __init__(self, message):
        self._message = message

    def __call__(self):
        print(self._message)
#end

# Optimization, Scipy
import scipy.optimize

driver.preprocess()
x = driver.getInitial()


options = {'disp': True, 'ftol': 1e-12, 'maxiter':100}

optimum = scipy.optimize.minimize(driver.fun, x, method="SLSQP", jac=driver.grad,\
          constraints=driver.getConstraints(), bounds=driver.getBounds(), options=options, callback=store_data)


his.close()



