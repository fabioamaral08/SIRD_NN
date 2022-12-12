# Imports
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import timedelta, datetime
import SIRD_NN.Rede as Rede
import SIRD_NN.Models as Mod

class Learner_Geral(object):
    def __init__(self, country, model, predict_range,is_semanal, *val_0, **kargs):
        self.country = country
        self.loss = model.loss
        self.predict_range = predict_range

        self.norm_fat = np.sum(val_0)
        self.val_0 = []
        if is_semanal:
            self.time_step = 7
        else:
            self.time_step = 1
        for v in val_0:
            v = v/self.norm_fat
            self.val_0.append(v)
        
        self.model = model(self.val_0)
        # Parameters creation
        #Beta
        self.net_beta = Rede.Rede()
        self.net_beta.add_lay(num_in = 1, num_out = 10,bias = True, activation='Relu_min')
        # self.net_beta.add_lay(num_in = 10, num_out = 10,bias = True, activation='Sigmoid')
        self.net_beta.add_lay(num_out = 1,bias = False, activation='Relu_min')

        self.params_calibration = np.concatenate([self.net_beta.get_weights(), np.array([1 / 17, .0005]).reshape(-1, )])
        self.other_param = kargs['params']
        if 'param_calibration' in kargs.keys():
            pc = kargs['param_calibration']
            self.params_calibration = np.concatenate([self.params_calibration, pc])
        

    def extend_index(self, index, new_size):
        values = index.values
        current = 0
        td = timedelta(days=self.time_step)
        try:
            current = datetime.strptime(index[-1], '%m/%d/%Y')
            while len(values) < (new_size):
                current = current + td
                values = np.append(values, datetime.strftime(current, '%m/%d/%Y'))
            return values

        except:
            for i in range(len(index)):
                current = datetime.strptime(index[i], '%d/%m/%Y')
                values[i] = datetime.strftime(current, '%m/%d/%Y')
            while len(values) < new_size:
                current = current + td
                values = np.append(values, datetime.strftime(current, '%m/%d/%Y'))
            return values

    def predict(self, data, val_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        IVP = solve_ivp(self.model.model, [0, size], val_0,
                        t_eval=np.arange(0, size, 1), vectorized=True, method='LSODA', args = (self.params_calibration, self.net_beta, self.other_param))

        return new_index, IVP

    def train(self, recovered, death, inf,vac1, vac2):
        recovered = recovered / self.norm_fat
        death     = death / self.norm_fat
        inf     = inf / self.norm_fat
        vac1     = vac1 / self.norm_fat
        vac2     = vac2/ self.norm_fat
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

            optimalGamma = minimize(self.model.loss, self.params_calibration,
                                    args=([inf, vac1, vac2, recovered, death], self.net_beta, self.other_param),
                                    method='L-BFGS-B', tol=1e-13, options = options_lbfgsb)

            
            eps /= 10
            worked = optimalGamma.success
            if eps < 10e-15:
                break
        self.worked = optimalGamma.success
        self.params_calibration = optimalGamma.x
        
        n = self.net_beta.get_num_param()
        self.net_beta.set_weights(self.params_calibration[:n])
        

    def save_results(self, data):
        
        pred_res = self.predict(data, self.val_0)
        new_index, prediction = pred_res
        y = prediction.y
        is_train = np.arange(len(new_index)) < len(data)
        t = prediction.t
        betas = self.net_beta.run(t)
        betas = np.array(betas).flatten()

        df_save = pd.DataFrame()
        df_save['Data'] = new_index
        df_save.set_index(new_index,inplace=True)
        df_save['SP-Subregião'] = self.country
        d = self.model.get_params(self.params_calibration, self.other_param)
        

        df_save['Infected'] = self.model.get_infected(y) * self.norm_fat
        df_save['Recovered'] = self.model.get_rec(y) * self.norm_fat
        df_save['Death'] = self.model.get_death(y) * self.norm_fat
        v1 = self.model.get_vac1(y)
        v2 = self.model.get_vac2(y)
        if v1 is not None:
            df_save['Vaccinated (1 Dose)'] = v1 * self.norm_fat
        if v2 is not None:
            df_save['Vaccinated (2 Dose)'] = v2 * self.norm_fat

        df_save['Used in Train'] = is_train
        df_save['beta(t)'] = betas
        df_save = pd.concat((df_save,pd.DataFrame(d,index=new_index)),1)
        df_save['Rt'] = self.model.calc_rt(prediction, self.params_calibration, self.other_param, self.net_beta)
        for i, v in enumerate(y):
            df_save[self.model.cols[i]] = v * self.norm_fat
        df_save['OPTM_Result'] = self.worked

        df_save = Utils.sort_data(df_save)
        return df_save
