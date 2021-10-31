# Approximate solution of F(t)=0 using Newton's method.

import numpy as np

########################################################################################################

def my_newton2(k,tau_r, I_1, tau_d1, I_2, tau_d2, t0, t_in, TOL, max_iter):
    tn = t_in
    for n in range(0,max_iter):
        #print('k,tau_r,I_1,tau_d1,I_2,tau_d2:  ' , k,tau_r,I_1,tau_d1,I_2,tau_d2)
        Ftn=fun2(tn,k,tau_r,I_1,tau_d1,I_2,tau_d2,0)
        DFtn = der_fun2(tn,k,tau_r,I_1,tau_d1,I_2,tau_d2,0)
        d = abs(Ftn/DFtn)
        #print('error: ',d, '; iteration:', n)
        if  d <  TOL:
            print('Found solution after',n,'iterations.')
            return tn
        
        if DFtn == 0:
            print('Zero derivative. No solution found.')
            return None

        tn = tn - d
    print('Exceeded maximum iterations. No solution found.')
    return None

### detailed functions
def fun2(t, k, tau_r, I_1, tau_d1, I_2, tau_d2, t0):
    A = t - (tau_r*tau_d1)/(tau_r-tau_d1) *np.log((I_1/k * tau_r/tau_d1) + (I_2/k*tau_r/tau_d2)*np.exp(((t-t0)*(tau_d2-tau_d1))/(tau_d1*tau_d2))) -t0
    return A

def der_fun2(t, k, tau_r, I_1, tau_d1, I_2, tau_d2, t0):
    A = 1- (I_2*tau_r*tau_d1*(tau_d2-tau_d1))/(tau_d2*(tau_r-tau_d1))*(I_1*tau_d2+I_2*tau_d1*np.exp(((t-t0)*(tau_d2-tau_d1))/(tau_d1*tau_d2)))**(-1)*np.exp(((t-t0)*(tau_d2-tau_d1))/(tau_d1*tau_d2))
    return A


