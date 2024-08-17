import os
import time
import numpy as np
import cyipopt

from drivers.constrained_optim_driver import ConstrainedOptimizationDriver

class CyipoptDriver(ConstrainedOptimizationDriver):
    """
    Driver to use with the Ipopt optimizer via cyipopt.
    """

    def __init__(self):
        ConstrainedOptimizationDriver.__init__(self)

    # the optimization problem
        self._nlp = None

    def objective(self, x):
        self._evaluateFunctions(x)
        return self._ofval.sum()
    
    def gradient(self, x):
        """Method passed to SciPy to get the objective function gradient."""
        # Evaluates gradients and functions if necessary, otherwise it
        # simply combines and scales the results.
        self._jacTime -= time.time()
        try:
            self._evaluateGradients(x)

            os.chdir(self._workDir)

            self._grad_f[()] = 0.0
            for obj in self._objectives:
                self._grad_f += obj.function.getGradient(self._variableStartMask) * obj.scale
            self._grad_f /= self._varScales

            # keep copy of result to use as fallback on next iteration if needed
            self._old_grad_f[()] = self._grad_f
        except:
            if self._failureMode == "HARD": raise
            self._grad_f[()] = self._old_grad_f
        #end

        if not self._parallelEval:
            self._runAction(self._userPostProcessGrad)

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return self._grad_f
    #end


    # Method passed to SciPy to expose the constraint vector.
    
    def constraints(self, x, idx):
        self._evaluateFunctions(x)

        if idx < len(self._constraintsEQ):
            out = self._eqval[idx]
        else:
            out = self._gtval[idx-len(self._constraintsEQ)]

        # Scale the out with local scaling (local to each constraint)
        # Incase of equality constraint:
        if idx < len(self._constraintsEQ):
            con = self._constraintsEQ[idx]
            f = -1.0 
            out = out * con.localscale

        # Incase of inequality constraint 
        else:   
            con = self._constraintsGT[idx-len(self._constraintsEQ)]
            f = self._gtval[idx-len(self._constraintsEQ)]
            out = out * con.localscale
        #end
        return out 
    #end

        # Method passed to SciPy to expose the constraint Jacobian.
    def jacobain(self, x, idx):
        self._jacTime -= time.time()
        try:
            self._evaluateGradients(x)

            os.chdir(self._workDir)

            mask = self._variableStartMask

            if idx < len(self._constraintsEQ):
                con = self._constraintsEQ[idx]
                f = -1.0 # for purposes of lazy evaluation equality is always active
            else:
                con = self._constraintsGT[idx-len(self._constraintsEQ)]
                f = self._gtval[idx-len(self._constraintsEQ)]
            #end

            if f < 0.0 or not self._asNeeded:
                self._jac_g[:,idx] = con.function.getGradient(mask) * (con.scale) / self._varScales
            else:
                self._jac_g[:,idx] = 0.0
            #end

            # keep reference to result to use as fallback on next iteration if needed
            self._old_jac_g[:,idx] = self._jac_g[:,idx]
        except:
            if self._failureMode == "HARD": raise
            self._jac_g[:,idx] = self._old_jac_g[:,idx]
        #end

        if not self._parallelEval:
            self._runAction(self._userPostProcessGrad)

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return self._jac_g[:,idx]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size,
        alpha_du,alpha_pr,ls_trials):

        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


    def getConUpperBound(self):
        conUpperBound = np.zeros([self._nCon,])
        i = len(self._constraintsEQ)
        conUpperBound[i:(i+len(self._constraintsGT))] = 1e20
        return conUpperBound

    def getConLowerBound(self):
        conLowerBound = np.zeros([self._nCon,])
        return conLowerBound

    def getnDV(self):
        """ Get the number of design variables """
        return self._nVar
    
    def getnCon(self):
        return self._nCon