import scipy
from scipy.integrate import quad
import numpy as np
from numpy.linalg import inv, eig, qr, det
import math
import random

class TauSampler: 
    def __init__(self, nVal, pVal, XMat, LMat, yVec, integration_config):
        '''
        Initialize parameters and store reused values
        '''
        n = nVal
        p = pVal
        X = XMat
        L = LMat
        y = yVec
        
        Q, R = qr(X)
        D_arr, V = eig(R @ L @ L @ R.T)
        D = np.diag(D_arr) # Converting eigenvalues into eigenvalue matrix
        d = 1 # Product of eigenvalues
        for ele in D_arr:
            d = d*ele
            
        yTy = y.T @ y
        yTQ = y.T @ Q
        yTQQTy = yTQ @ yTQ.T
        yTQV = y.T @ Q @ V
        VTQTy = yTQV.T
        
        cdfMap = {}
        
        config = integration_config
        
    def probability(tau):
        '''
        Calculate probability density for tau
        '''
        det = 1
        for d_i in D_arr:
            det *= (1+(tau**2)*d_i)

        ##Difficult multiplication term
        mult = yTy - yTQQTy + (yTQV @ np.diag(1/(np.diag(D)+tau**(-2))) @ VTQTy)/(tau**2)

        return det**(-0.5) * ((w/2 + mult/2)**(-0.5*(n+w)))*(1/((tau**(-1))*(1+tau**(-2))))
    
    def g():
        '''
        Calculate normalizing constant
        '''
        NORM_STEP = config["NORM_STEP"]
        NORM_THRESHOLD = config["NORM_THRESHOLD"]
        initial = config["initial"]

        ## Initialize lower and upper vals
        lower = initial
        upper = lower + NORM_STEP
        prob = lambda x: probability(x)
        val = (prob(lower) + prob(upper))/2 * NORM_STEP

        cdfMap[(lower+upper)/2] = val

        ## Res is ultimately returned
        res = 0

        ## While there's a lot of change to res
        while res == 0 or (abs((val-res)/res) > NORM_THRESHOLD):
            res = val
            lower = lower + NORM_STEP
            upper = upper + NORM_STEP
            val += (prob(lower) + prob(upper))/2 * NORM_STEP
            cdfMap[val] = (lower+upper)/2
            
        return res, upper
    

    def sample(num_points):
        '''
        Sample tau
        -- PARAM num_points: number of points to sample
        -- RETURN ret: array of samples
        '''
        u_vals = np.random.rand(num_points)
        u_vals = np.sort(u_vals)
        ret = []
        u_counter = 0
        cdf_keys = sorted(list(cdfMap.keys()))
        for count in range(len(cdf_keys)):
            
            ## If we've collected enough samples, break
            if u_counter == len(u_vals):
                break
            
            ## If the desired U is in between two stored CDF's...
            if u_vals[u_counter] >= cdf_keys[count] \
            and u_vals[u_counter] < cdf_keys[count+1]:
                
                ## Take the difference between U and the lower CDF
                diff = u_vals[u_counter] - cdf_keys[count]
                lower = cdfMap[cdf_keys[count]]
                
                ## Approximate the value between the lower and upper tau's for which CDF=U
                nstep = config["NORM_STEP"]/50
                i = 1
                while (prob(lower)+prob(lower+i*nstep))/2 * nstep*i < diff:
                    i += 1
                ret.append(lower+nstep*(i-1))
                
                ## Increment u_counter, decrement count
                u_counter += 1
                count -= 1
                
            ## If the desired U is above our highest    
            elif u_vals[u_counter] >= cdf_keys[count] and count = len(cdf_keys)-1:
                ret.append(cdfMap[cdf_keys[count]])
                u_counter += 1
                count -= 1
                
        return ret
