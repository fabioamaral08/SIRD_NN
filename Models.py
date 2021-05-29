import abc
import numpy as np
from scipy.integrate import solve_ivp
class Model(metaclass=abc.ABCMeta):
    @classmethod
    def __init__(self, val_0, cols=[]):
        self.val_0 = val_0
        self.cols = cols

    @abc.abstractmethod
    def get_infected(self, y):
        pass
    @abc.abstractmethod
    def get_death(self, y):
        pass
    @abc.abstractmethod
    def get_rec(self, y):
        pass
    @abc.abstractmethod
    def calc_rt(self, sol, params_calibration, other_param, net):
        pass
    @abc.abstractmethod
    def get_vac1(self,y):
        pass
    @abc.abstractmethod
    def get_vac2(self,y):
        pass
    @abc.abstractmethod
    def get_params(self, paramns, other_params):
        pass

    @abc.abstractmethod
    def model(self, t,y, param_calibration, net, other_param):
        pass

    
    def loss(self, point, data, net, params):
        """
        Parameters:
            point: array, list,iterable
                parameters for calibration
            data: array, list,iterable
                Training data
            net: Rede()
                Object to run the ANN for evaluate beta(t)
        """
        
        size = len(data[0])
        solution = solve_ivp(self.model, [0, size], self.val_0,
                            t_eval=np.arange(0, size, 1), vectorized=True, method='LSODA', args=(point, net, params))

        
        try:
            t = [int(t) for t in solution.t]
            y = solution.y
            I = self.get_infected(y)
            R = self.get_rec(y)
            D = self.get_death(y)
            V1 = self.get_vac1(y)
            V2 = self.get_vac2(y)
            
            model_vars = [I,V1, V2, R, D]
            l = 0
            for i,d in enumerate(model_vars):
                if d is not None:
                    # l += (np.mean(((np.log(d*nf +1) - np.log(data[i][t].astype('float32') * nf + 1 ) )) ** 2))
                    l += (np.mean(((np.log(d) - np.log(data[i][t].astype('float32')) )) ** 2))


        except Exception as e: 
            print(str(e))
        # print(l)
        return l

"""
###########################################################################
###########                   IMPLEMENTATIONS                   ###########
###########################################################################
"""

class SIRD(Model):
    def __init__(self, val_0):
        cols = ['Susceptible', 'Infected', 'Recovered', 'Death']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[1]
    
    def get_death(self, y):
        return y[3]
    
    def get_rec(self, y):
        return y[2]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = sol.y[0]
        rt = betas/(gamma + gammaD) * S
        return rt
    
    def get_vac1(self,y):
        return None
    
    def get_vac2(self,y):
        return None

    def get_params(self, paramns, other_params):
        d = {}
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        return d
    
    def model(self, t,y, param_calibration, net, other_param=None):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        # theta = 0.52
        # theta2 = 0.98
        S,I,_,_ = y

        dS = -beta * S * I
        dI =  beta * S * I - (gamma_r + gamma_d) * I
        dR = gamma_r * I
        dD = gamma_d * I
        return [dS, dI, dR, dD]


class SIR(Model):
    def __init__(self, val_0):
        cols = ['Susceptible', 'Infected', 'Recovered']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[1]
    
    def get_death(self, y):
        return None
    
    def get_rec(self, y):
        return y[2]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma = params_calibration[-1:]
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = 1 -  np.sum(sol.y[1:],axis=0).flatten()
        rt = betas/(gamma) * S
        return rt
    
    def get_vac1(self,y):
        return None
    
    def get_vac2(self,y):
        return None
    def get_params(self, paramns, other_params):
        d = {}
        d['Gamma'] = paramns[-1]
        return d

    
    def model(self, t,y, param_calibration, net, other_param=None):
        #Parameters
        gamma_r = param_calibration[-1]
        
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        # theta = 0.52
        # theta2 = 0.98
        S,I,_ = y
 
        dS = -beta * S * I
        dI =  beta * S * I - (gamma_r) * I
        dR = gamma_r * I
        return [dS, dI, dR]


############################### 
####                       ####
####   VACINAÇÃO - TAXA    ####
####                       ####
###############################     
class SVIRD_1D(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated' ,
                'Infected (non Vacc.)',
                'Infected (Vacc.)',
                'Recovered (non Vacc.)',
                'Recovered (Vacc)', 
                'Death']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[2] + y[3]
    
    def get_death(self, y):
        return y[6]
    
    def get_rec(self, y):
        return y[4] + y[5]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        _, _, theta1 = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = 1 -  np.sum(sol.y[1:],axis=0).flatten()
        V1 = ((1-theta1) * (sol.y[1]))
        rt = betas / (gamma + gammaD) * (S + V1)
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[3] + y[5]
    
    def get_vac2(self,y):
        return None
    def get_params(self, paramns, other_param):
        d = {}
        vac, theta = other_param 
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        d['theta(t)'] = theta
        d['Vacciantion Rate'] = vac
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        theta, vac = other_param
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        S,V,Is,Iv, Rs, _, _ = y

        I = Is + Iv
        #Function
        dS = -vac * S - beta * I * S
        dV =  vac * S - beta * (1-theta) * I * V
        dIs = beta * I * S  - (gamma_r + gamma_d) * Is
        dIv = beta * (1-theta)  * I * V  - gamma_r * Iv
        dD = gamma_d * Is 
        dRs = gamma_r * Is - Rs * vac
        dRv = gamma_r * Iv + Rs * vac
        return [dS, dV, dIs,dIv, dRs, dRv, dD]


class SVIRD_1Di(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated (Intermed)' ,
                'Vaccinated' ,
                'Infected (non Vacc.)',
                'Infected (Vacc.)',
                'Recovered (non Vacc.)',
                'Recovered (Vacc)', 
                'Death']
        super().__init__( val_0, cols)

    
    def get_infected(self, y):
        return y[4] + y[3]
    
    def get_death(self, y):
        return y[7]
    
    def get_rec(self, y):
        return y[6] + y[5]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        _, _, theta1 = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = 1 -  np.sum(sol.y[2:],axis=0).flatten()
        V1 = ((1-theta1) * (sol.y[3]))
        rt = betas / (gamma + gammaD) * (S + V1)
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[2] + y[4] + y[5] 
    
    def get_vac2(self,y):
        return None
    def get_params(self, paramns, other_param):
        d = {}
        vac, theta, _ = other_param 
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        d['theta(t)'] = theta
        d['Vacciantion Rate'] = vac
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        vac, theta, alpha = other_param 
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        
        S,Vi,V,Is,Iv, Rs, _, _ = y

        I = Is + Iv
        #Function
        dS = -vac * S - beta * I * S
        dVi =  vac * S - beta *  I * Vi - alpha *Vi
        dV =  alpha * Vi - beta * (1-theta) * I * V
        dIs = beta * I  * S  - (gamma_r + gamma_d) * Is
        dIv = beta * I * ((1-theta) * V + Vi ) - gamma_r * Iv
        dD = gamma_d * Is 
        dRs = gamma_r * Is - Rs * vac
        dRv = gamma_r * Iv + Rs * vac
        return [dS, dVi, dV, dIs,dIv, dRs, dRv, dD]



class SVIRD_2D(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated 1D' ,
                'Vaccinated 2D' ,
                'Infected (non Vacc)',
                'Infected (Vacc 1)',
                'Infected (Vacc 2)',
                'Recovered (non Vacc)',
                'Recovered (Vacc 1)', 
                'Recovered (Vacc 2)', 
                'Death']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[5] + y[4] + y[3]
    
    def get_death(self, y):
        return y[9]
    
    def get_rec(self, y):
        return y[6] + y[7] + y[8]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        _, _, theta1, theta2 = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = 1 -  np.sum(sol.y[1:],axis=0).flatten()
        V1 = (1-theta1) * (sol.y[1])
        V2 = (1-theta2) * sol.y[2]
        rt = betas / (gamma + gammaD) * (S + V1 + V2)
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[4] + y[7] + self.get_vac2(y)
    
    def get_vac2(self,y):
        return y[2] + y[5] + y[8] 
    def get_params(self, paramns, other_param):
        d = {}
        vac1, vac2 , theta1, theta2 = other_param
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        d['theta1(t)'] = theta1
        d['theta2(t)'] = theta2
        d['1D Vacciantion Rate'] = vac1
        d['2D Vacciantion Rate'] = vac2
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        vac1, vac2, theta, theta2 = other_param
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)

        S,V1,V2,Is,Iv1, Iv2, Rs, Rv1, _,_ = y

        I = Is + Iv1 + Iv2
        #Function

        dS = -vac1 * S - beta * I * S                                                               # Susceptible
        dV1 = vac1 * S  - beta * (1-theta) * I * V1 - vac2 * V1                                   # effective 1 Dose
        sV2 = vac2 * V1  - beta * (1-theta2) * I * V2                                               # effective 2 Dose
        dIs = beta * I * S  - (gamma_r + gamma_d) * Is                                              # Not vaccinated Infected
        dIv1 = beta * (1-theta) * I * V1 - (gamma_r) * Iv1                        # Infected 1 Dose
        dIv2 = beta * (1-theta2) * I * V2 - gamma_r * Iv2              # Infected 2 Dose
        dRs = gamma_r * Is - Rs * vac1                                                              # Recovered S
        dRv1 = gamma_r * Iv1 + Rs * vac1 - Rv1 * vac2                                               # Recovered Iv1
        dRv2 = gamma_r * Iv2 + Rv1 * vac2                                                           # Recovered iv2
        dD = gamma_d * (Is)                                                                         # Deceased
        return [dS,dV1,sV2,dIs,dIv1,dIv2,dRs,dRv1,dRv2,dD]


class SVIRD_2Di(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated 1Di' ,
                'Vaccinated 1D' ,
                'Vaccinated 2Di' ,
                'Vaccinated 2D' ,
                'Infected (non Vacc)',
                'Infected (Vacc 1)',
                'Infected (Vacc 2)',
                'Recovered (non Vacc)',
                'Recovered (Vacc 1)', 
                'Recovered (Vacc 2)', 
                'Death']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[5] + y[6] + y[7]
    
    def get_death(self, y):
        return y[11]
    
    def get_rec(self, y):
        return y[8] + y[9] + y[10]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        _, _, theta1, theta2, _ = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S =  np.sum(sol.y[:2],axis=0).flatten()
        V1 = (1-theta1) * (sol.y[2] + sol.y[3])
        V2 = (1-theta2) * sol.y[4]
        rt = betas * ( S/(gamma + gammaD) + (V1 + V2) / (gamma) ) 
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[2] + y[6] + y[9] 
    
    def get_vac2(self,y):
        return y[3] + y[4] + y[7] + y[10]
    def get_params(self, paramns, other_param):
        d = {}
        vac1, vac2 , theta1, theta2, alpha = other_param
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        d['theta1(t)'] = theta1
        d['theta2(t)'] = theta2
        d['1D Vacciantion Rate'] = vac1
        d['2D Vacciantion Rate'] = vac2
        d['Effectiveness Delay'] = alpha
        
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        vac1, vac2, theta, theta2, alpha = other_param
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        # theta = 0.52
        # theta2 = 0.98
        S,V1i,V1,V2i,V2,Is,Iv1, Iv2, Rs, Rv1, _,_ = y
        
        I = Is + Iv1 + Iv2
        #Function

        dS = -vac1 * S - beta * I * S                                                               # Susceptible
        dV1i = vac1 * S - beta * I * V1i - alpha * V1i  - vac2 * V1i                                           # 1 Dose 
        dV1 = alpha * V1i - beta * (1-theta) * I * V1 - vac2 * V1                                   # effective 1 Dose
        dV2i = vac2 * V1 + vac2 * V1i - alpha * V2i  - beta * (1-theta) * I * V2i                                # 2 Dose
        sV2 = alpha * V2i - beta * (1-theta2) * I * V2                                              # effective 2 Dose
        dIs = beta * I * S  - (gamma_r + gamma_d) * Is                                              # Not vaccinated Infected
        dIv1 = beta * (1-theta) * I * V1 + beta * I * V1i  - (gamma_r) * Iv1                        # Infected 1 Dose
        dIv2 = beta * (1-theta) * I * V2i + beta * (1-theta2) * I * V2 - gamma_r * Iv2              # Infected 2 Dose
        dRs = gamma_r * Is - Rs * vac1                                                              # Recovered S
        dRv1 = gamma_r * Iv1 + Rs * vac1 - Rv1 * vac2                                               # Recovered Iv1
        dRv2 = gamma_r * Iv2 + Rv1 * vac2                                                           # Recovered Iv2
        dD = gamma_d * (Is)                                                                         # Deceased
        return [dS,dV1i,dV1,dV2i,sV2,dIs,dIv1,dIv2,dRs,dRv1,dRv2,dD]



class SVIRD_2Di_vacrate(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated 1Di' ,
                'Vaccinated 1D' ,
                'Vaccinated 2Di' ,
                'Vaccinated 2D' ,
                'Infected (non Vacc)',
                'Infected (Vacc 1)',
                'Infected (Vacc 2)',
                'Recovered (non Vacc)',
                'Recovered (Vacc 1)', 
                'Recovered (Vacc 2)', 
                'Death']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[5] + y[6] + y[7]
    
    def get_death(self, y):
        return y[11]
    
    def get_rec(self, y):
        return y[8] + y[9] + y[10]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD,_,_ = params_calibration[-4:]
        theta1, theta2, _ = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S =  np.sum(sol.y[:2],axis=0).flatten()
        V1 = (1-theta1) * (sol.y[2] + sol.y[3])
        V2 = (1-theta2) * sol.y[4]
        rt = betas * ( S/(gamma + gammaD) + (V1 + V2) / (gamma) ) 
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[2] + y[6] + y[9] 
    
    def get_vac2(self,y):
        return y[3] + y[4] + y[7] + y[10]
    def get_params(self, paramns, other_param):
        d = {}
        theta1, theta2, alpha = other_param
        d['Gamma_Rec'] = paramns[-4]
        d['Gamma_Death'] = paramns[-3]
        d['theta1(t)'] = theta1
        d['theta2(t)'] = theta2
        d['1D Vacciantion Rate'] = paramns[-2]
        d['2D Vacciantion Rate'] = paramns[-1]
        d['Effectiveness Delay'] = alpha
        
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d, vac1, vac2, = param_calibration[-4:]
        theta, theta2, alpha = other_param
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        # theta = 0.52
        # theta2 = 0.98
        S,V1i,V1,V2i,V2,Is,Iv1, Iv2, Rs, Rv1, _,_ = y
        
        I = Is + Iv1 + Iv2
        #Function

        dS = -vac1 * S - beta * I * S                                                               # Susceptible
        dV1i = vac1 * S - beta * I * V1i - alpha * V1i                                              # 1 Dose 
        dV1 = alpha * V1i - beta * (1-theta) * I * V1 - vac2 * V1                                   # effective 1 Dose
        dV2i = vac2 * V1  - alpha * V2i  - beta * (1-theta) * I * V2i                               # 2 Dose
        sV2 = alpha * V2i - beta * (1-theta2) * I * V2                                              # effective 2 Dose
        dIs = beta * I * S  - (gamma_r + gamma_d) * Is                                              # Not vaccinated Infected
        dIv1 = beta * (1-theta) * I * V1 + beta * I * V1i  - (gamma_r) * Iv1                        # Infected 1 Dose
        dIv2 = beta * (1-theta) * I * V2i + beta * (1-theta2) * I * V2 - gamma_r * Iv2              # Infected 2 Dose
        dRs = gamma_r * Is - Rs * vac1                                                              # Recovered S
        dRv1 = gamma_r * Iv1 + Rs * vac1 - Rv1 * vac2                                               # Recovered Iv1
        dRv2 = gamma_r * Iv2 + Rv1 * vac2                                                           # Recovered Iv2
        dD = gamma_d * (Is)                                                                         # Deceased
        return [dS,dV1i,dV1,dV2i,sV2,dIs,dIv1,dIv2,dRs,dRv1,dRv2,dD]
##############################
####                      ####
####   VACINAÇÃO - REC    ####
####                      ####
##############################

class SVIRD_1D_Rec(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated' ,
                'Infected (non Vacc.)',
                'Infected (Vacc.)',
                'Recovered (non Vacc.)',
                'Recovered (Vacc)', 
                'Death']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[2] + y[3]
    
    def get_death(self, y):
        return y[6]
    
    def get_rec(self, y):
        return y[4] + y[5]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        _, _, theta1 = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = 1 -  np.sum(sol.y[2:],axis=0).flatten()
        rt = betas / (gamma + gammaD) * S
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[3] + y[5]
    
    def get_vac2(self,y):
        return None
    def get_params(self, paramns, other_param):
        d = {}
        vac, theta = other_param 
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        d['theta(t)'] = theta
        d['Vacciantion Rate'] = vac
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        theta, vac = other_param
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        S,V,Is,Iv, Rs, _, _ = y

        I = Is + Iv
        #Function
        dS = -vac * S - beta * I * S
        dV =  vac * S * (1-theta) - beta * I * V
        dIs = beta * I * S  - (gamma_r + gamma_d) * Is
        dIv = beta * I * V  - gamma_r * Iv
        dD = gamma_d * Is 
        dRs = gamma_r * Is - Rs * vac
        dRv = gamma_r * Iv + Rs * vac + vac * S * theta
        return [dS, dV, dIs,dIv, dRs, dRv, dD]


class SVIRD_1Di_Rec(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated (Intermed)' ,
                'Vaccinated' ,
                'Infected (non Vacc.)',
                'Infected (Vacc.)',
                'Recovered (non Vacc.)',
                'Recovered (Vacc)', 
                'Death']
        super().__init__( val_0, cols)

    
    def get_infected(self, y):
        return y[4] + y[3]
    
    def get_death(self, y):
        return y[7]
    
    def get_rec(self, y):
        return y[6] + y[5]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        _, _, theta1 = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = 1 -  np.sum(sol.y[3:],axis=0).flatten()
        rt = betas / (gamma + gammaD) * S
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[2] + y[4] + y[5] 
    
    def get_vac2(self,y):
        return None
    def get_params(self, paramns, other_param):
        d = {}
        vac, theta, _ = other_param 
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        d['theta(t)'] = theta
        d['Vacciantion Rate'] = vac
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        vac, theta, alpha = other_param 
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        
        S,Vi,V,Is,Iv, Rs, _, _ = y

        I = Is + Iv
        #Function
        dS = -vac * S - beta * I * S
        dVi = vac * S - beta * I * Vi - alpha *Vi
        dV =  alpha * Vi * (1-theta) - beta * I * V
        dIs = beta * I  * S  - (gamma_r + gamma_d) * Is
        dIv = beta * I * (V + Vi) - gamma_r * Iv
        dD = gamma_d * Is 
        dRs = gamma_r * Is - Rs * vac
        dRv = gamma_r * Iv + Rs * vac + alpha * Vi * theta
        return [dS, dVi, dV, dIs,dIv, dRs, dRv, dD]



class SVIRD_2D_Rec(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated 1D' ,
                'Vaccinated 2D' ,
                'Infected (non Vacc)',
                'Infected (Vacc 1)',
                'Infected (Vacc 2)',
                'Recovered (non Vacc)',
                'Recovered (Vacc 1)', 
                'Recovered (Vacc 2)', 
                'Death',
                'Imunized 1D',
                'Imunized 2D']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[5] + y[4] + y[3]
    
    def get_death(self, y):
        return y[9]
    
    def get_rec(self, y):
        return y[6] + y[7] + y[8]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        _, _, theta1, theta2 = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = 1 -  np.sum(sol.y[2:],axis=0).flatten()
        rt = betas / (gamma + gammaD) * S
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[4] + y[7] + y[10] + self.get_vac2(y)
    
    def get_vac2(self,y):
        return y[2] + y[5] + y[8] + y[11]
    def get_params(self, paramns, other_param):
        d = {}
        vac1, vac2 , theta1, theta2 = other_param
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        d['theta1(t)'] = theta1
        d['theta2(t)'] = theta2
        d['1D Vacciantion Rate'] = vac1
        d['2D Vacciantion Rate'] = vac2
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        vac1, vac2, theta, theta2 = other_param
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)

        S,V1,V2,Is,Iv1, Iv2, Rs, Rv1, _,_,Im1,_ = y

        I = Is + Iv1 + Iv2
        #Function

        dS = -vac1 * S - beta * I * S                                                               # Susceptible
        dV1 = vac1 * S * (1-theta) - beta * I * V1 - vac2 * V1                                      # ineffective 1 Dose
        sV2 = vac2 * V1 * (1-theta2) - beta * I * V2                                                # ineffective 2 Dose
        dIs = beta * I * S  - (gamma_r + gamma_d) * Is                                              # Not vaccinated Infected
        dIv1 = beta * I * V1 - gamma_r * Iv1                                                        # Infected 1 Dose
        dIv2 = beta * I * V2 - gamma_r * Iv2                                                        # Infected 2 Dose
        dRs = gamma_r * Is - Rs * vac1                                                              # Recovered S
        dRv1 = gamma_r * Iv1 + Rs * vac1 - Rv1 * vac2                                               # Recovered Iv1
        dRv2 = gamma_r * Iv2 + Rv1 * vac2                                                           # Recovered iv2
        dD = gamma_d * Is                                                                           # Deceased
        dIm1 = vac1 * S * theta - Im1 * vac2
        dIm2 = vac2 * V1 * theta2 + Im1 * vac2
        return [dS,dV1,sV2,dIs,dIv1,dIv2,dRs,dRv1,dRv2,dD,dIm1, dIm2]


class SVIRD_2Di_Rec(Model):
    def __init__(self, val_0):
        cols = ['Susceptible',
                'Vaccinated 1Di' ,
                'Vaccinated 1D' ,
                'Vaccinated 2Di' ,
                'Vaccinated 2D' ,
                'Infected (non Vacc)',
                'Infected (Vacc 1)',
                'Infected (Vacc 2)',
                'Recovered (non Vacc)',
                'Recovered (Vacc 1)', 
                'Recovered (Vacc 2)', 
                'Death',
                'Imunized 1D',
                'Imunized 2D']
        super().__init__(val_0, cols)

    
    def get_infected(self, y):
        return y[5] + y[6] + y[7]
    
    def get_death(self, y):
        return y[11]
    
    def get_rec(self, y):
        return y[8] + y[9] + y[10]
    
    def calc_rt(self, sol, params_calibration, other_param, net):
        gamma, gammaD = params_calibration[-2:]
        _, _, theta1, theta2 = other_param
        t = sol.t
        betas = net.run(t)
        betas = np.array(betas).flatten()
        S = 1 -  np.sum(sol.y[4:],axis=0).flatten()
        rt = betas / (gamma + gammaD) * S
        return rt
    
    def get_vac1(self,y):
        return y[1] + y[2] + y[6] + y[9] + y[12] + self.get_vac2(y)
    
    def get_vac2(self,y):
        return y[3] + y[4] + y[7] + y[10]  + y[13]
    def get_params(self, paramns, other_param):
        d = {}
        vac1, vac2 , theta1, theta2 = other_param
        d['Gamma_Rec'] = paramns[-2]
        d['Gamma_Death'] = paramns[-1]
        d['theta1(t)'] = theta1
        d['theta2(t)'] = theta2
        d['1D Vacciantion Rate'] = vac1
        d['2D Vacciantion Rate'] = vac2
        return d

    
    def model(self, t,y, param_calibration, net, other_param):
        #Parameters
        gamma_r, gamma_d = param_calibration[-2:]
        vac1, vac2, theta, theta2 = other_param
        n = net.get_num_param()
        net.set_weights(param_calibration[:n])
        beta = net.run(t)
        # theta = 0.52
        # theta2 = 0.98
        S,V1i,V1,V2i,V2,Is,Iv1, Iv2, Rs, Rv1, _,_,Im1,_ = y

        I = Is + Iv1 + Iv2
        #Function

        dS =  -vac1 * S - beta * I * S                                                              # Susceptible
        dV1i = vac1 * S - beta * I * V1i - 1/20 * V1i                                               # 1 Dose 
        dV1 = 1/20 * V1i * (1-theta) - beta * I * V1 - vac2 * V1                                    # effective 1 Dose
        dV2i = vac2 * V1 - 1/20 * V2i - beta * I * V2i                                              # 2 Dose
        sV2 = 1/20 * V2i * (1-theta2) - beta * I * V2                                               # effective 2 Dose
        dIs = beta * I * S  - (gamma_r + gamma_d) * Is                                              # Not vaccinated Infected
        dIv1 = beta * I * (V1 + V1i) - gamma_r * Iv1                                                # Infected 1 Dose
        dIv2 = beta * I * (V2i + V2) - gamma_r * Iv2                                                # Infected 2 Dose
        dRs = gamma_r * Is - Rs * vac1                                                              # Recovered S
        dRv1 = gamma_r * Iv1 + Rs * vac1 - Rv1 * vac2                                               # Recovered Iv1
        dRv2 = gamma_r * Iv2 + Rv1 * vac2                                                           # Recovered iv2
        dD = gamma_d * (Is)                                                                         # Deceased
        dIm1 = 1/20 * V1i * theta - vac2 * Im1
        dIm2 = 1/20 * V2i * theta2 + vac2 * Im1
        return [dS,dV1i,dV1,dV2i,sV2,dIs,dIv1,dIv2,dRs,dRv1,dRv2,dD, dIm1, dIm2]