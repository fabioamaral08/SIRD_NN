# Imports
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import timedelta, datetime


class Learner(object):
    def __init__(self, country, loss, predict_range, s_0, i_0, rC_0, rD_0):
        self.country = country
        self.loss = loss
        self.predict_range = predict_range

        self.norm_fat = (s_0 + i_0 + rC_0 + rD_0)

        self.s_0 = s_0 / self.norm_fat
        self.i_0 = i_0 / self.norm_fat
        self.rC_0 = rC_0 / self.norm_fat
        self.rD_0 = rD_0 / self.norm_fat

        # Parameters creation
        b1, b2 = createLay([1, 10, 1])
        params = np.empty((0,))
        for i in range(len(b1)):
            params = np.concatenate([params, b1[i].reshape(-1, )])
        for i in range(len(b2)):
            params = np.concatenate([params, b2[i].reshape(-1, )])

        self.params = np.concatenate([params, np.array([1 / 14, .001]).reshape(-1, )])
        self.worked = False
    def extend_index(self, index, new_size):
        values = index.values
        current = 0
        try:
            current = datetime.strptime(index[-1], '%m/%d/%Y')
            while len(values) < (new_size):
                current = current + timedelta(days=1)
                values = np.append(values, datetime.strftime(current, '%m/%d/%Y'))
            return values

        except:
            for i in range(len(index)):
                current = datetime.strptime(index[i], '%d/%m/%Y')
                values[i] = datetime.strftime(current, '%m/%d/%Y')
            while len(values) < new_size:
                current = current + timedelta(days=1)
                values = np.append(values, datetime.strftime(current, '%m/%d/%Y'))
            return values

    def predict(self, b1, b2, gamma, gammaD, data, s_0, i_0, r_0, rD_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        
        def SIRD(t, y):
            beta = run_beta(b1, b2, t)
            S = y[0]
            I = y[1]
            return [-beta * S * I, beta * S * I - (gamma + gammaD) * I, gamma * I, gammaD * I]

        IVP = solve_ivp(SIRD, [0, size], [s_0, i_0, r_0, rD_0],
                        t_eval=np.arange(0, size, 1), vectorized=True, method='LSODA')
        return new_index, IVP

    def train(self, recovered, death, data, ini, fim):
        recovered = (recovered) / self.norm_fat
        death = death / self.norm_fat
        data = data / self.norm_fat
        worked = False
        eps = 1e-9
        while not worked:
            options_lbfgsb={'disp': None, 
            'maxcor': 10, 
            'ftol': 2.220446049250313e-09, 
            'gtol': 1e-05, 
            'eps': eps, 
            'maxfun': 15000, 
            'maxiter': 15000, 
            'iprint': - 1, 
            'maxls': 20, 
            'finite_diff_rel_step': None}

            optimalGamma = minimize(self.loss, self.params,
                                    args=(data, recovered, death, self.s_0, self.i_0, self.rC_0, self.rD_0),
                                    method='L-BFGS-B', tol=1e-13, options = options_lbfgsb)
            eps /= 10
            worked = optimalGamma.success
            if eps < 10e-15:
                break
        self.worked = optimalGamma.success
        self.params = optimalGamma.x

    def save_results(self, data):
        b1, b2 = unpack_lay(self.params[:-2])
        gamma, gammaD = [self.params[-2], self.params[-1]]

        pred_res = self.predict(b1, b2, gamma, gammaD, data,
                                 self.s_0, self.i_0, self.rC_0, self.rD_0)
        new_index, prediction = pred_res
        pred_inf = prediction.y[1] * self.norm_fat
        pred_rec = prediction.y[2] * self.norm_fat
        pred_death = prediction.y[3] * self.norm_fat

        is_train = np.arange(len(new_index)) < len(data)
        betas = []
        for t in range(len(new_index)):
            betas.append(run_beta(b1, b2, t)[0][0])
        # Save CSV:
        betas = np.array(betas)
        lethality = pred_death / (pred_inf + pred_rec + pred_death)
        Rt = betas / (gamma + gammaD) * (1 - prediction.y[1] - prediction.y[2] - prediction.y[3])
        df_save = pd.DataFrame({'Data': new_index,
                                'SP-Subregião': self.country,
                                'Infected': np.rint(pred_inf),
                                'Recovered': np.rint(pred_rec),
                                'Death': np.rint(pred_death),
                                'Used in Train': is_train,
                                'beta(t)': betas,
                                'gamma_Rec': gamma,
                                'gamma_Death': gammaD,
                                'Lethality': lethality,
                                'Rt': Rt,
                                'Optim Results': self.worked})
        return df_save


class Learner_SIR(object):
    def __init__(self, country, loss, predict_range, s_0, i_0, rC_0):
        self.country = country
        self.loss = loss
        self.predict_range = predict_range

        self.norm_fat = (s_0 + i_0 + rC_0)

        self.s_0 = s_0 / self.norm_fat
        self.i_0 = i_0 / self.norm_fat
        self.rC_0 = rC_0 / self.norm_fat

        # Parameters creation
        b1, b2 = createLay([1, 10, 1])
        params = np.empty((0,))
        for i in range(len(b1)):
            params = np.concatenate([params, b1[i].reshape(-1, )])
        for i in range(len(b2)):
            params = np.concatenate([params, b2[i].reshape(-1, )])

        self.params = np.concatenate([params, np.array([1 / 14]).reshape(-1, )])

    def extend_index(self, index, new_size):
        values = index.values
        current = 0
        try:
            current = datetime.strptime(index[-1], '%m/%d/%Y')
            while len(values) < (new_size):
                current = current + timedelta(days=1)
                values = np.append(values, datetime.strftime(current, '%m/%d/%Y'))
            return values

        except:
            for i in range(len(index)):
                current = datetime.strptime(index[i], '%d/%m/%Y')
                values[i] = datetime.strftime(current, '%m/%d/%Y')
            while len(values) < new_size:
                current = current + timedelta(days=1)
                values = np.append(values, datetime.strftime(current, '%m/%d/%Y'))
            return values

    def predict(self, b1, b2, gamma, data, s_0, i_0, r_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        def SIR(t, y):
            beta = run_beta(b1, b2, t)
            S = y[0]
            I = y[1]
            return [-beta * S * I, beta * S * I - (gamma) * I, gamma * I]

        IVP = solve_ivp(SIR, [0, size], [s_0, i_0, r_0],
                        t_eval=np.arange(0, size, 1), vectorized=True, method='LSODA')
        return new_index, IVP

    def train(self, recovered, death, data, ini, fim):
        recovered = (recovered + death) / self.norm_fat
        data = data / self.norm_fat

        optimalGamma = minimize(self.loss, self.params,
                                args=(data, recovered, death, self.s_0, self.i_0, self.rC_0),
                                method='L-BFGS-B', tol=1e-13)

        self.params = optimalGamma.x

    def save_results(self, data):
        b1, b2 = unpack_lay(self.params[:-1])
        gamma = self.params[-1]

        pred_res = self.predict(b1, b2, gamma, data,
                                 self.s_0, self.i_0, self.rC_0)
        new_index, prediction = pred_res
        pred_inf = prediction.y[1] * self.norm_fat
        pred_rec = prediction.y[2] * self.norm_fat

        is_train = np.arange(len(new_index)) < len(data)
        betas = []
        for t in range(len(new_index)):
            betas.append(run_beta(b1, b2, t)[0][0])
        # Save CSV:
        betas = np.array(betas)
        Rt = betas / (gamma) * (1 - prediction.y[1] - prediction.y[2])
        df_save = pd.DataFrame({'Data': new_index,
                                'SP-Subregião': self.country,
                                'Infected': np.rint(pred_inf),
                                'Recovered': np.rint(pred_rec),
                                'Used in Train': is_train,
                                'beta(t)': betas,
                                'gamma_Rec': gamma,
                                'Rt': Rt})
        return df_save


# BETA REDE - GAMMA E GAMMA_D
def lossGamma(point, data, recovered, death, s_0, i_0, r_0, rD_0):
    size = len(data)
    gamma, gammaD = [point[-2], point[-1]]
    b1, b2 = unpack_lay(point[:-2])

    def SIR(t, y):
        beta = run_beta(b1, b2, t)
        S = y[0]
        I = y[1]
        return [-beta * S * I, beta * S * I - (gamma + gammaD) * I, gamma * I, gammaD * I]

    solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0, rD_0],
                         t_eval=np.arange(0, size, 1), vectorized=True, method='LSODA')

    try:
        l1 = (np.mean(((np.log(solution.y[1]) - np.log(data.astype('float32')))) ** 2))
        l2 = (np.mean(((np.log(solution.y[2]) - np.log(recovered.astype('float32')))) ** 2))
        l3 = (np.mean(((np.log(solution.y[3]) - np.log(death.astype('float32')))) ** 2))
    except:

        l1 = 0
        l2 = 0
        l3 = 0
        pass

    l = l1 + l2 + l3
    
    return l

    # BETA REDE - GAMMA
def lossSIR(point, data, recovered, death, s_0, i_0, r_0):
    size = len(data)
    gamma = point[-1]
    b1, b2 = unpack_lay(point[:-1])

    def SIR(t, y):
        beta = run_beta(b1, b2, t)
        S = y[0]
        I = y[1]
        return [-beta * S * I, beta * S * I - (gamma) * I, gamma * I]

    solution = solve_ivp(SIR, [0, size], [s_0, i_0, r_0],
                         t_eval=np.arange(0, size, 1), vectorized=True, method='LSODA')

    try:
        l1 = (np.mean(((np.log(solution.y[1]) - np.log(data))) ** 2))
        l2 = (np.mean(((np.log(solution.y[2]) - np.log(recovered))) ** 2))
    except:

        l1 = 0
        l2 = 0
        pass

    l = l1 + l2

    return l


def createLay(lay):
    beta = []
    bias = []
    for i in range(len(lay) - 1):
        n = np.zeros((lay[i], lay[i + 1]))
        beta.append(n)
        bias.append(np.ones((1, lay[i + 1])))
    return beta, bias


def run_beta(beta, bias, t):
    x = np.array([t])
    x = x.reshape((x.flatten().shape[0], 1))
    xmin = 1e-3

    def sig(x):
        return 1.0 / (1 + np.exp(-x))

    def relu(x):
        return np.maximum(x, 0) + xmin

    for i in range(len(beta) - 1):
        x = sig(np.dot(x, beta[i]) + bias[i])
    x = relu(np.dot(x, beta[-1]))
    return x


def unpack_lay(param, nn=10):
    b = param[:-1].reshape((3, nn))[0:2, :]
    bias = param[:-1].reshape((3, nn))[-1, :]

    beta = [b[0].reshape((1, -1)), b[1].reshape((1, -1)).T]
    bias = [bias, param[-2]]
    return beta, bias


class Learner_Geral(object):
    def __init__(self, country, loss, model, predict_range, p, *val_0, **kargs):
        self.country = country
        self.loss = loss
        self.predict_range = predict_range

        self.norm_fat = np.sum(val_0)
        self.val_0 = []
        self.model = model
        p[0] = p[0]/self.norm_fat
        self.params_fixed = p

        if 'cols' not in kargs.keys():
            kargs['cols'] = ['Susceptible', 'Infected', 'Recovered', 'Deaths']
        for i,v in enumerate(val_0):
            v = v/self.norm_fat
            self.val_0.append(v)

        # Parameters creation
        b1, b2 = createLay([1, 10, 1])
        params = np.empty((0,))
        for i in range(len(b1)):
            params = np.concatenate([params, b1[i].reshape(-1, )])
        for i in range(len(b2)):
            params = np.concatenate([params, b2[i].reshape(-1, )])
        self.params_calibration = params
        if len(val_0) == 6:
            self.SIRD = Learner(country, loss, predict_range, val_0[0], val_0[2] + val_0[3], val_0[4], val_0[5])
        else:
            self.SIRD = Learner(country, loss, predict_range, val_0[0], val_0[2], val_0[3], val_0[4])
        self.pred_ini = np.zeros((len(val_0),))
        

    def extend_index(self, index, new_size):
        values = index.values
        current = 0
        try:
            current = datetime.strptime(index[-1], '%m/%d/%Y')
            while len(values) < (new_size):
                current = current + timedelta(days=1)
                values = np.append(values, datetime.strftime(current, '%m/%d/%Y'))
            return values

        except:
            for i in range(len(index)):
                current = datetime.strptime(index[i], '%d/%m/%Y')
                values[i] = datetime.strftime(current, '%m/%d/%Y')
            while len(values) < new_size:
                current = current + timedelta(days=1)
                values = np.append(values, datetime.strftime(current, '%m/%d/%Y'))
            return values

    def predict(self, data, val_0):
        old_s =len(data.index)
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        IVP = solve_ivp(self.model, [0, size], val_0,
                        t_eval=np.arange(old_s-1, size, 1), vectorized=True, method='LSODA', args = (self.params_calibration, self.params_fixed))
        return new_index[old_s-1:], IVP

    def train(self, recovered, death, inf, ini, fim):

        self.SIRD.train(recovered, death, inf, ini, fim)
        recovered = recovered / self.norm_fat
        death     = death / self.norm_fat
        inf     = inf / self.norm_fat

        s0  = 1 - (inf[-1] + recovered[-1] + death[-1])
        self.pred_ini = [s0,0, inf[-1], 0, recovered[-1], death[-1]]

        self.params_calibration = self.SIRD.params

    def save_results(self, data):
        b1, b2 = unpack_lay(self.params_calibration[:-2])
        gamma, gammaD = [self.params_calibration[-2], self.params_calibration[-1]]
        
        pred_res = self.predict(data, self.pred_ini)
        new_index, prediction = pred_res

        betas = []
        offset = len(data)-1
        for t in range(len(new_index)):
            betas.append(run_beta(b1, b2, t + offset)[0][0])
        betas = np.array(betas)
        
        df_SIRD = self.SIRD.save_results(data)
        df_SIRD = df_SIRD.rename(columns={'Infected': f'Infected (No Vaccine)',
                                'Recovered': 'Recovered (No Vaccine)',
                                'Death': 'Death (No Vaccine)',
                                'Rt': 'Rt (No Vaccine)'})
        df_SIRD.set_index(df_SIRD['Data'],inplace=True)
        # Save CSV:
        eta = self.params_fixed[1]
        if len( prediction.y) == 6:
            pred_inf_s = prediction.y[2] * self.norm_fat
            pred_inf_v = prediction.y[3] * self.norm_fat
            pred_vac = prediction.y[1] * self.norm_fat
            pred_rec = prediction.y[4] * self.norm_fat
            pred_death = prediction.y[5] * self.norm_fat
            Rt = betas / (gamma + gammaD) * ((1 - prediction.y[2] - prediction.y[3] - prediction.y[4] - prediction.y[5] -  prediction.y[1] ) + (eta * prediction.y[1]))
            df_save = pd.DataFrame({
                                'Infected (Vaccinated)': np.rint(pred_inf_v),
                                'Infected (Non Vaccinated)': np.rint(pred_inf_s),
                                'Recovered': np.rint(pred_rec),
                                'Death': np.rint(pred_death),
                                'Vaccinated': np.rint(pred_vac),
                                'Rt': Rt}, index = new_index)
        else:
            pred_inf = prediction.y[2] * self.norm_fat
            pred_vac = prediction.y[1] * self.norm_fat
            pred_rec = prediction.y[3] * self.norm_fat
            pred_death = prediction.y[4] * self.norm_fat
            
            Rt = betas / (gamma + gammaD) * (1 - prediction.y[2] - prediction.y[3] - prediction.y[4] )
            df_save = pd.DataFrame({
                                'SP-Subregião': self.country,
                                'Infected': np.rint(pred_inf),
                                'Recovered': np.rint(pred_rec),
                                'Death': np.rint(pred_death),
                                'Vaccinated': np.rint(pred_vac),
                                'Rt': Rt}, index = new_index)
    


        df_save = pd.concat([df_SIRD, df_save], axis=1)
        df_save = Utils.sort_data(df_save)
        return df_save

def SVIIRD(t,y, params_calibration, params_fixed):
    """
    SIRD model: vaccination adpt.

    Pramaters:
        fixed:
            vac: vaccination rate
            theta_v: immutity rate by vaccination
        calibration:
            beta: transmission rate
            gamma_r: recovery rate
            gamma_d: death rate
        

    Model variables:
        S: Susceptible
        V: Vaccinated but not immunized
        Is: Infection in non-vacinated group
        Iv: Infection in vacinated group
        R: Immunized
        D: Deceased
    """
    #Parameters
    gamma_r, gamma_d = params_calibration[-2:]
    vac, theta_v= params_fixed
    net, bias = unpack_lay(params_calibration[:-2])
    beta = run_beta(net, bias, t)
    S,V,Is, Iv, _, _ = y
    I = Is + Iv

    #Function
    dS = -vac * S - beta * I * S
    dV = vac * (1-theta_v) * S - beta * I * V
    dIs = beta * I * S  - (gamma_r + gamma_d) * Is 
    dIv = beta * I * V  - (gamma_r ) * Iv
    dD = gamma_d * Is 
    dR = gamma_r * (Is + Iv) + vac * S * theta_v
    return [dS, dV, dIs, dIv, dR,dD]

def lossSVIIRD(point, data, val_0, param_fixed, model = SVIIRD, var_idx = (2,3,4,5)):
    """
    Parameters:
        point: array, list,iterable
            parameters for calibration
        data: array, list,iterable
            Training data
        val_0: array, list,iterable
            initial values for IVP
        param_fixed: array, list,iterable
            non-trainable parameters (pre determined)
        model: callable (fun(t,y,param_calibration, param_fixed))
            Function of IVP model
        var_idx: interable of ints (index)
            index of data used on training
    """
    size = len(data)
    solution = solve_ivp(model, [0, size], val_0,
                         t_eval=np.arange(0, size, 1), vectorized=True, method='LSODA', args=(point,param_fixed))
    try:
        l = 0
        for i in var_idx:
            l = l + (np.mean(((np.log(solution.y[i]) - np.log(data[i]))) ** 2))
        
    except:
        l = 0
    return l


def SVIRD(t,y, params_calibration, params_fixed):
    """
    SIRD model: vaccination adpt.

    Pramaters:
        fixed:
            vac: vaccination rate
            theta_v: immutity rate by vaccination
        calibration:
            beta: transmission rate
            gamma_r: recovery rate
            gamma_d: death rate
        

    Model variables:
        S: Susceptible
        V: Vaccinated but not immunized
        I: infected
        R: Immunized
        D: Deceased
    """
    #Parameters
    gamma_r, gamma_d = params_calibration[-2:]
    vac, theta_v = params_fixed
    net, bias = unpack_lay(params_calibration[:-2])
    beta = run_beta(net, bias, t)
    S,V,I, _, _ = y

    
    #Function
    dS = -vac * S - beta * I * S
    dV = vac * (1-theta_v) * S - beta * I * V
    dI = beta * I * (S+V)  - (gamma_r + gamma_d) * I
    dD = gamma_d * I
    dR = gamma_r * I + vac * theta_v * S
    return [dS, dV, dI, dR,dD]


def SVIIRD_tetteh(t,y, params_calibration, params_fixed):
    """
    SIRD model: vaccination adpt.

    Pramaters:
        fixed:
            vac: vaccination rate
            theta_v: immutity rate by vaccination
        calibration:
            beta: transmission rate
            gamma_r: recovery rate
            gamma_d: death rate
        

    Model variables:
        S: Susceptible
        V: Vaccinated but not immunized
        Is: Infection in non-vacinated group
        Iv: Infection in vacinated group
        R: Immunized
        D: Deceased
    """
    #Parameters
    gamma_r, gamma_d = params_calibration[-2:]
    vac, theta_v= params_fixed
    net, bias = unpack_lay(params_calibration[:-2])
    beta = run_beta(net, bias, t)
    S,V,Is, Iv, _, _ = y
    I = Is + Iv
    # medRxvi doi: https://doi.org/10.1101/2020.12.22.20248693; (Adp.)
    #Function
    dS = -vac * S - beta * I * S
    dV =  vac * S - beta * (1-theta_v) * I * V
    dIs = beta * I * S  - (gamma_r + gamma_d) * Is 
    dIv = beta * I * V * (1-theta_v)  - (gamma_r ) * Iv
    dD = gamma_d * Is 
    dR = gamma_r * (Is + Iv)
    return [dS, dV, dIs, dIv, dR,dD]