import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import SIRD_NN
import SIRD_NN.Models as Mod
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import numpy as np
import pandas as pd
from datetime import timedelta, datetime


def run_region(region, sl1, sl, dc, dt, step=14, mAvg=False,
                min_casos=0, is_SIR=False, model=Mod.SIRD):
    """
        region: str
            Name of the region/counry/city
        sl1: int
            position in the dataframe of the first training data
        sl: int
            position in the dataframe of the last training data (sl > sl1)
        dc: pandas.DataFrame
            metadata of region (to gat population)
        dt: pandas.Dataframe
            source of traing data. the len is assumed to be >= sl. The required columns are:
                - 'At': active cases
                - 'Rt': cummulative cases
                - 'Óbitos': cummulative deceases
                - 'Confirmados': cummulative confirmed cases
                - 'Data': Date of the data
            the rows should be sorted by date (column 'Data')
        step: int
            prediction range (in days) after the last training day data.
            Default: 14
        mAvg: boolean
            Use the moving average to smooth the data
            default: False
        min_casos: int
            discards data od dt where the cummulative confirmed is less then min_casos.
            default: 0
        is_SIR: boolean
            Use SIR model instead of SIRD model
            default: False
    """
    pop = get_pop(region, dc)

    ini = get_data(dt, region, is_pred=False)
    ini = ini[["Data", "At", "Rt", "Óbitos", 'Confirmados']]
    ini = sort_data(ini)
    ini = ini.set_index(ini['Data'])

    countday = np.arange(0, len(ini))
    ini['Count Day'] = countday

    if mAvg:
        ini = movingAvg(ini, 7, ["Data", "At", "Rt", "Óbitos", 'Confirmados'])

    recovered_pp = ini["Rt"]
    death_pp = ini["Óbitos"]
    data_pp = ini["At"]
    conf = ini['Confirmados'].iloc[sl1]
    d = sl - sl1 + step
    i_0 = data_pp.iloc[sl1]
    rC_0 = recovered_pp.iloc[sl1]
    rD_0 = death_pp.iloc[sl1]
    s_0 = pop - conf
    if is_SIR:
        val_0 = [s_0, i_0, rC_0 + rD_0]
    else:
        val_0 = [s_0, i_0, rC_0, rD_0]
    learner = SIRD_NN.Learner_Geral(region, model, d, *val_0, params=[])
    # recovered, death, inf,vac1, vac2,
    learner.train(recovered_pp[sl1:sl], death_pp[sl1:sl],
                  data_pp[sl1:sl], 0, 0, sl1, sl)
    df_save = learner.save_results(data_pp[sl1:sl])
    return learner, df_save


def run_vac(region, sl1, sl, dc, dt, step=14, mAvg=False,
                min_casos=0, model=Mod.SVIRD_1D, dose=False, intermed=False, rec=False, **kargs):
    """
        region: str
            Name of the region/counry/city
        sl1: int
            position in the dataframe of the first training data
        sl: int
            position in the dataframe of the last training data (sl > sl1)
        dc: pandas.DataFrame
            metadata of region (to gat population)
        dt: pandas.Dataframe
            source of traing data. the len is assumed to be >= sl. The required columns are:
                - 'At': active cases
                - 'Rt': cummulative cases
                - 'Óbitos': cummulative deceases
                - 'Confirmados': cummulative confirmed cases
                - 'Data': Date of the data
            the rows should be sorted by date (column 'Data')
        step: int
            prediction range (in days) after the last training day data.
            Default: 14
        mAvg: boolean
            Use the moving average to smooth the data
            default: False
        min_casos: int
            discards data od dt where the cummulative confirmed is less then min_casos.
            default: 0

    """
    pop = get_pop(region, dc)

    ini = get_data(dt, region, is_pred=False)
    ini = ini[["Data", "At", "Rt", "Óbitos", 'Confirmados', 'Vac 1', 'Vac 2']]
    ini = sort_data(ini)
    ini = ini.set_index(ini['Data'])
    countday = np.arange(0, len(ini))
    ini['Count Day'] = countday

    if mAvg:
        ini = movingAvg(ini, 7, ["Data", "At", "Rt",
                        "Óbitos", 'Confirmados', 'Vac 1', 'Vac 2'])

    recovered_pp = ini["Rt"]
    death_pp = ini["Óbitos"]
    data_pp = ini["At"]
    vac1 = ini['Vac 1']
    vac2 = ini['Vac 2']
    conf = ini['Confirmados'].iloc[sl1]
    d = sl - sl1 + step
    i_0 = data_pp.iloc[sl1]
    rC_0 = recovered_pp.iloc[sl1]
    rD_0 = death_pp.iloc[sl1]

    s_0 = pop - conf
    v2_0 = vac2.iloc[sl1]
    if dose:
        if intermed:
            vi2 = vac2.iloc[sl1-20]
            vi1 = vac1.iloc[sl1-20] - vi2 
            v2 = v2_0 - vi2
            v1 = vac1.iloc[sl1] - vi1 - vi2 - v2
            if rec:
                _,_, theta1,theta2 = kargs['params']
                v1,im1 = [v1*(1-theta1), v1*theta1]
                v2,im2 = [v2*(1-theta2), v2*theta2]
                val_0 = [s_0,vi1,v1,vi2,v2,i_0,0,0,rC_0,0,0,rD_0,im1,im2]
            else:
                val_0 = [s_0,vi1,v1,vi2,v2,i_0,0,0,rC_0,0,0,rD_0]
        else:
            v1_0 = vac1.iloc[sl1] - v2_0
            if rec:
                _,_, theta1,theta2 = kargs['params']
                v1_0,im1 = [v1_0*(1-theta1), v1_0*theta1]
                v2_0,im2 = [v2_0*(1-theta2), v2_0*theta2]
                val_0 = [s_0,v1_0,v2_0,i_0,0,0,rC_0,0,0,rD_0,im1,im2]
            else:
                val_0 = [s_0,v1_0,v2_0,i_0,0,0,rC_0,0,0,rD_0]
    else:
        v1_0 = vac1.iloc[sl1]
        if intermed:
            vi1 = vac1.iloc[sl1-20]
            v1 = v1_0 - vi1
            if rec:
                _,theta1,_ = kargs['params']
                v1,im1 = [v1*(1-theta1), v1*theta1]
                val_0 = [s_0,vi1,v1,i_0,0,rC_0,0,rD_0,im1]
            else:
                val_0 = [s_0,vi1,v1,i_0,0,rC_0,0,rD_0]
        else:
            if rec:
                _,theta1 = kargs['params']
                v1_0,im1 = [v1_0*(1-theta1), v1_0*theta1]
                val_0 = [s_0,v1_0,i_0,0,rC_0,0,rD_0,im1]
            else:
                val_0 = [s_0,v1_0,i_0,0,rC_0,0,rD_0]





    # if dose:
    #     if intermed:
    #         if rec:
                
    #         else:
    #             val_0 = [s_0,vi1,v1,vi2,v2,i_0,0,0,rC_0,0,0,rD_0]
    #     else:
    #         if rec:
    #             val_0 = [s_0,v1_0,v2_0,i_0,0,0,rC_0,0,0,rD_0,0,0]
    #         else:
    #             val_0 = [s_0,v1_0,v2_0,i_0,0,0,rC_0,0,0,rD_0]
    # else:
    #     if intermed:
    #         if rec:
    #             val_0 = [s_0,vi1,v1,i_0,0,rC_0,0,rD_0,0]
    #         else:
    #             val_0 = [s_0,vi1,v1,i_0,0,rC_0,0,rD_0]
    #     else:
    #         if rec:
    #             val_0 = [s_0,v1_0,i_0,0,rC_0,0,rD_0,0]
    #         else:
    #             val_0 = [s_0,v1_0,i_0,0,rC_0,0,rD_0]
    
    learner = SIRD_NN.Learner_Geral(region, model, d, *val_0, **kargs)
    learner.train(recovered_pp[sl1:sl], death_pp[sl1:sl], data_pp[sl1:sl], vac1[sl1:sl], vac2[sl1:sl], sl1, sl)
    
    df_save = learner.save_results(data_pp[sl1:sl])
    
    return learner, df_save, data_pp[sl1:sl]

def get_pop(region, dc):
    if region == 'São Paulo (Estado)' or region == 'Brasil':
        pop = dc['Habitantes (2019)'].sum()
    elif region in dc['Região'].unique():
        pop = dc[dc['Região'] == region]['Habitantes (2019)'].sum()
    else:
        c = dc[dc["SP-Subregião"] == region]
        pop = c["Habitantes (2019)"].values[0]
    return pop


def calc_vacRate(vac):
    vac = vac[1:].values - vac[:-1].values
    vac = vac[vac>0]
    vacR = np.mean(vac)
    return vacR


def atualiza_dados(sheet_page = 'Data_subregions'):
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('data/cred-sir.json',scope)
    client = gspread.authorize(creds)


    sheet =  client.open('dados')
    d = sheet.worksheet(sheet_page)
    df = pd.DataFrame(d.get_all_records())
    if 'Data' in df.columns:
        df.loc[:,'Data'] = pd.to_datetime(df.Data)
        df.loc[:,'Data'] = df['Data'].dt.strftime('%m/%d/%Y')
    df.to_csv(f"data\\dados - {sheet_page}.csv")

def translate(r):
        translator = {
        'Grande SP Leste': 'Greater SP East',
        'Grande SP Norte': 'Greater SP North',
        'Grande SP Oeste': 'Greater SP West',
        'Grande SP Sudeste': 'Greater SP Southeast',
        'Grande SP Sudoeste': 'Greater SP Southwest',
        'Sul':'South',
        'Norte':'North',
        'Nordeste':'Northeast',
        'Sudeste': 'Southeast',
        'Centro-Oeste': 'Midwest',
        'Brasil':'Brazil',
        'Metropolitana': 'Greater São Paulo',
        'Litorânea':'Coastal',
        'Interior Leste': 'Interior (East)',
        'Interior Oeste': 'Interior (West)',
        'Estado de SP': 'State of São Paulo',
        'São Paulo (Estado)': "São Paulo (State)"
        }
        if r in translator.keys():
            return translator[r]
        return r

def read_global(region):
    # read files
    df_c = pd.read_csv('JHU/time_series_covid19_confirmed_global.csv')
    df_d = pd.read_csv('JHU/time_series_covid19_deaths_global.csv')
    df_r = pd.read_csv('JHU/time_series_covid19_recovered_global.csv')
    
    # get data from country
    df_c = df_c[df_c['Country/Region'] == region ]
    df_d = df_d[df_d['Country/Region'] == region ]
    df_r = df_r[df_r['Country/Region'] == region ]
    
#     #unify regions and transpose dataframe
    df_c = df_c.groupby('Country/Region', as_index = False).sum().transpose()
    df_d = df_d.groupby('Country/Region', as_index = False).sum().transpose()
    df_r = df_r.groupby('Country/Region', as_index = False).sum().transpose()
    
    # format date and set as index
    data_c = pd.to_datetime(df_c.index[4:])
    data_d = pd.to_datetime(df_d.index[4:])
    data_r = pd.to_datetime(df_r.index[4:])
    
    idx_c = df_c.index.values
    idx_d = df_d.index.values
    idx_r = df_r.index.values


    idx_c[4:] = data_c.strftime('%m/%d/%Y')
    idx_d[4:] = data_d.strftime('%m/%d/%Y')
    idx_r[4:] = data_r.strftime('%m/%d/%Y')
    
    df_c = df_c.set_index(idx_c)
    df_d = df_d.set_index(idx_d)
    df_r = df_r.set_index(idx_r)
    
    # get array of dates
    data = data_r.strftime('%m/%d/%Y')
    
    # get datas
    conf =  df_c.iloc[4:].values.flatten()
    death =  df_d.iloc[4:].values.flatten()
    dailyDeath = np.zeros((len(death),))
    dailyDeath[1:] = death[1:] - death[:-1]
    rec =  df_r.iloc[4:].values.flatten()
    infec = conf - death - rec

    # create dataframe
    df = pd.DataFrame(data = {
        'SP-Subregião':region,
        'Data':data,
        'Rt':rec,
        'Óbitos':death,
        'Confirmados' :conf,
        'At' : infec
    },index = data)
    return df

def get_data(df_data,r, is_pred = True):
    cols = ['beta(t)','gamma_Rec','gamma_Death','Lethality','Rt']
    dc = pd.read_csv(f'data/dados - Agrupamento.csv')
    if r in df_data["SP-Subregião"].unique():
        df_d = df_data[df_data["SP-Subregião"] == r]
    elif r in dc['Região'].unique():
        est = dc[dc['Região'] == r]['SP-Subregião'].unique()
        df_d = pd.DataFrame()
        for e in est:
            df_d =df_d.append(df_data[df_data['SP-Subregião'] == e])
        df_d = df_d.groupby(['Data'], as_index=False).sum()
        df_d['SP-Subregião'] = r
        if is_pred:
            l = len(est)
            for c in cols:
                if c in df_d.columns:
                    df_d.loc[:,c] = df_d[c]/l
    else:
        return pd.DataFrame(columns = df_data.columns)
    if 'Used in Train' in df_data.columns:
        df_d =  df_d.astype({'Used in Train': 'bool'})
    return df_d

def sort_data(df, col = 'Data'):
    df.loc[:,col] = pd.to_datetime(df[col])
    df.sort_values(by=col, inplace=True, ascending=True)
    df.loc[:,col] = df[col].dt.strftime('%m/%d/%Y')
    return df



def diff_mean(df_vals):
    df_mean = df_vals.mean(1)
    idx = df_mean.index
    er_total = np.zeros((len(idx),))
    for pos, i in enumerate(idx):
        p = df_mean.loc[i]
        q = df_vals.loc[i]
        er = 0
        n = (~np.isnan(q)).sum()
        if n > 1:
            for v in q:
                if np.isnan(v):
                    continue
                er += (p - v) ** 2
            er = er / (n - 1)
        er_total[pos] = (er)
    return np.linalg.norm(er_total)

def movingAvg(df, n, cols = None):
    if cols is None:
        return df.rolling(window=n).mean()
    else:
        df.loc[:, 1:] = df.loc[:, cols].rolling(window=n).mean().copy()
        return df

def run_mape(df_d, df_p):
    return df_d.sub(df_p, axis  = 0).div(df_d, axis = 0).abs().mean() * 100

def run_rmse(df_d, df_p):
    return np.sqrt(np.mean(np.square(df_d.sub(df_p, axis  = 0))))

def run_nrmse(df_d, df_p,**kwarg):
    rmse = run_rmse(df_d, df_p)
    med = np.std(df_d)
    return rmse/med

def run_mae(df_d, df_p):
    return df_d.sub(df_p, axis  = 0).abs().mean()


def ERROR_DF(df_data, df_p, r, cols_p = ['Infected', 'Recovered', 'Death'],
             cols_d=["At", 'Rt', "Óbitos", 'Data'],prev = True, error = run_mape): 
    dc = pd.read_csv(f'data/dados - Agrupamento.csv')
    df_d = get_data(df_data,r,is_pred = False)

    df_p = get_data(df_p,r)
    if prev:
        df_p = df_p[~df_p['Used in Train']]
    else:
        df_p = df_p[df_p['Used in Train']]
        
    

    idx = df_p['Data']
    df_p = df_p.set_index(df_p['Data'])
    df_d = df_d.set_index(df_d['Data'])
    idx_d = df_d.index
    idx_d = idx_d.intersection(idx)
    df_d = df_d.loc[idx_d]
    df_p = df_p.loc[idx_d]
    

    
    df_d['Total'] = df_d[cols_d[0:3]].sum(axis=1)
    
    df_p.loc[:,'Total'] = df_p.sum(axis=1)

    l = len(df_d)
    df_d = df_d.set_index(np.arange(l))
    df_p = df_p.set_index(np.arange(l))
    
    er_I = error(df_d[cols_d[0]], df_p[cols_p[0]])
    er_R = error(df_d[cols_d[1]], df_p[cols_p[1]])
    er_D = error(df_d[cols_d[2]], df_p[cols_p[2]])
    er_T = error(df_d['Total'], df_p['Total'])
    return er_I, er_R, er_D,er_T

def MAPE(arq_data, arq_prev, file = 'MAPE.csv', total = False, save = False, MM = False):
    if arq_data != 'JHU':
        df_data = pd.read_csv(arq_data, index_col = False)
    df_prev = pd.read_csv(arq_prev)
    
    df_save = pd.DataFrame(columns = ["SP-Subregião", 'MAPE Infectados', 'MAPE Óbitos', 'MAPE  Recuperados'])
    for r in df_prev["SP-Subregião"].unique():
        df_p = df_prev[df_prev["SP-Subregião"] == r]
        if arq_data != 'JHU':
            df_d = get_data(df_data,r,is_pred = False)
        else:
            df_d = read_global(r)
        if total:
            df_p = df_p[df_p['Used in Train']]
        else:
            df_p = df_p[~df_p['Used in Train']]
        if MM:
#             df_d = movingAvg(df_d, 7, ["Data", "At", "Rt", "Óbitos", 'Confirmados'])
            pass
        df_d = df_d[df_d["Data"].isin(df_p["Data"])][["At", "Óbitos", 'Rt','Data']]
        df_p = df_p[df_p["Data"].isin(df_d["Data"])]
        l = len(df_d)
        df_d = df_d.set_index(np.arange(l))
        df_p = df_p[['Infected', 'Death', 'Recovered']].set_index(np.arange(l))
        er_I = run_mape(df_d['At'], df_p['Infected'])
        er_R = run_mape(df_d['Rt'], df_p['Recovered'])
        er_D = run_mape(df_d['Óbitos'], df_p['Death'])
        df_save = df_save.append({"SP-Subregião":r, 'MAPE Infectados':er_I, 'MAPE Óbitos': er_D, 'MAPE  Recuperados':er_R}, ignore_index=True)
        
    if save:
        df_save.to_csv(file)
    return df_save
def cluster(region,prev,pasta, lim = None,coef_I = 1, coef_D = 0, coef_R = 0, percent = 0.05):
    if os.path.isfile(f'{pasta}/{region}/MAPE_Total-{region}-Prev{prev}.csv'):
        file_mape = f'{pasta}/{region}/MAPE_Total-{region}-Prev{prev}.csv'
    else:
        file_mape = f'{pasta}/{region}/MAPE_Real-{region}.csv'
    MAPES = pd.read_csv(file_mape,index_col = 0)
    
    grupos = {}
    if lim is None:
        lim_I = (np.max(MAPES['MAPE Infectados']) - np.min(MAPES['MAPE Infectados'])) * percent + np.min(MAPES['MAPE Infectados'])
        lim_R = (np.max(MAPES['MAPE  Recuperados']) - np.min(MAPES['MAPE  Recuperados'])) * percent + np.min(MAPES['MAPE  Recuperados'])
        lim_D = (np.max(MAPES['MAPE Óbitos']) - np.min(MAPES['MAPE Óbitos'])) * percent + np.min(MAPES['MAPE Óbitos'])
        
        lim = (lim_I * coef_I + lim_R * coef_R + lim_D * coef_D)/(coef_I + coef_R + coef_D)
    for i in range(len(MAPES)):
        Inf = MAPES['MAPE Infectados'].iloc[i] * coef_I
        Rec = MAPES['MAPE  Recuperados'].iloc[i] * coef_R
        Dea = MAPES['MAPE Óbitos'].iloc[i] * coef_D
        Dia = MAPES.index[i]
        
#         metrica = (Inf + Rec + Dea)/(coef_I + coef_R + coef_D)
        if Inf < lim and Rec < lim and Dea < lim:
#         if metrica < lim:
            g = 0
        else:
            g = 1
        grupos[f'{Dia}'] = g
    return grupos

def filter_results(region, dia_ini, dia_fim, prev, pasta, inner_dir, coef_I = 1, coef_D = 1, coef_R = 1, lim = None,return_total = False, dir_sufix=None):
    g = cluster(region,prev, coef_I = coef_I, coef_D = coef_D, coef_R = coef_R, lim = lim, pasta = pasta)
    df_g1 = pd.DataFrame()   
    df_data = pd.read_csv("data\\dados - Data_subregions.csv")
    for dLen in range(dia_ini,dia_fim+1):
        if inner_dir:
            if dir_sufix is None:
                file = f'{pasta}/{region}/prev-{prev}/Subregions_Pred_{dLen}D_prev-{prev}-{region}.csv'
            else:
                file = f'{pasta}/{region}/{dir_sufix}/Subregions_Pred_{dLen}D_prev-{prev}-{region}.csv'
        else:        
            file = f'{pasta}/{region}/Subregions_Pred_{dLen}D_prev-{prev}-{region}.csv'
        if not os.path.exists(file):
            continue
        df = pd.read_csv(file,index_col = 0)
        if g[f'{dLen}'] == 0:# and max(df['Rt']) < 2.5:
            if return_total:
                df = df.set_index(df['Data'])
                df_g1 = pd.concat([df_g1,df], axis =1)
            else:
                df_g1 = df_g1.append(df)
    
    if return_total or len(df_g1) == 0:
        return df_g1,df_g1,df_g1
    
    df_mean = df_g1.groupby('Data', as_index=False).mean()
    df_min = df_g1.groupby('Data', as_index=False).min()
    df_max = df_g1.groupby('Data', as_index=False).max()
    
    df_mean.insert(1,'SP-Subregião', region)

    return df_mean,df_min,df_max

def unifica(dia_ini, dia_fim, prev, region, pasta, inner_dir = False, df_geral = None,
            crop = 10,file1 = f'data/dados - Data_subregions.csv', MM = False, dir_sufix=None):
    
    df_MAPE = pd.DataFrame()
    
    for dLen in range(dia_ini,dia_fim+1):
        if inner_dir:
            if dir_sufix is None:
                file2 = f'{pasta}/{region}/prev-{prev}/Subregions_Pred_{dLen}D_prev-{prev}-{region}.csv'
            else:
                file2 = f'{pasta}/{region}/{dir_sufix}/Subregions_Pred_{dLen}D_prev-{prev}-{region}.csv'
        else:        
            file2 = f'{pasta}/{region}/Subregions_Pred_{dLen}D_prev-{prev}-{region}.csv'
        if not os.path.exists(file2):
            df_MAPE1 = pd.DataFrame({'SP-Subregião':f'{region}',
                                     'MAPE Infectados':100.0,
                                     'MAPE Óbitos':100.0,
                                     'MAPE Recuperados':100.0}, index = [dLen])
            print(f'{file2} - does not exist')
        else:       
            df_MAPE1 = MAPE(file1, file2,total = True,MM=MM)
            df_MAPE1 = df_MAPE1.set_index(pd.Index([dLen]))
    
        df_MAPE = df_MAPE.append(df_MAPE1)
    print(file2)
    df_MAPE.to_csv(f'{pasta}/{region}/MAPE_Total-{region}-Prev{prev}.csv')
    df_pred,df_pred_min, df_pred_max  = filter_results(region,dia_ini,dia_fim, prev, pasta, inner_dir, lim = 50, dir_sufix=dir_sufix)
    
    df_pred = df_pred.drop('Unnamed: 0', axis=1,errors ='ignore')
    df_pred = df_pred.round({'Infected':0, 'Recovered':0, 'Death':0})
    
    if len(df_pred) == 0:
        if df_geral is None:
            return df_pred, df_pred_min, df_pred_max
        else: 
            return df_geral
    
    df_pred = sort_data(df_pred)
    df_pred_min = sort_data(df_pred_min)
    df_pred_max = sort_data(df_pred_max)
    count_idx = np.arange(0, len(df_pred))
    df_pred.set_index(count_idx,inplace=True)
    df_pred_min.set_index(count_idx,inplace=True)
    df_pred_max.set_index(count_idx,inplace=True)
    
    df_pred = df_pred.astype({'Used in Train': 'bool'})

    esp = len(df_pred[~df_pred['Used in Train']])
    
    if crop is not None:
        if esp > crop:
            cut = esp-crop
            df_pred = df_pred.iloc[:-cut]
            df_pred_min = df_pred_min.iloc[:-cut]
            df_pred_max = df_pred_max.iloc[:-cut]


    if df_geral is not None:
        df_geral[0] = df_geral[0].append(df_pred)
        df_geral[1] = df_geral[1].append(df_pred_min)
        df_geral[2] = df_geral[2].append(df_pred_max)
        
        return df_geral
    return df_pred, df_pred_min, df_pred_max

def MAPE_DF(df_data, df_p, r, cols_p = ['Infected', 'Recovered', 'Death'], cols_d=["At", 'Rt', "Óbitos", 'Data'],prev = True): 
    dc = pd.read_csv(f'data/dados - Agrupamento.csv')
    df_d = get_data(df_data,r,is_pred = False)

    df_p = get_data(df_p,r)

    if prev:
        df_p = df_p[~df_p['Used in Train']]
    else:
        df_p = df_p[df_p['Used in Train']]
        
    

    df_d = df_d[df_d["Data"].isin(df_p["Data"])][cols_d]
    df_p = df_p[df_p["Data"].isin(df_d["Data"])]
    df_d['Total'] = df_d[cols_d[0:3]].sum(axis=1)
    
    df_p.loc[:,'Total'] = df_p.sum(axis=1)
    l = len(df_d)
    df_d = df_d.set_index(np.arange(l))
    df_p = df_p.set_index(np.arange(l))

    er_I = run_mape(df_d[cols_d[0]], df_p[cols_p[0]])
    er_R = run_mape(df_d[cols_d[1]], df_p[cols_p[1]])
    er_D = run_mape(df_d[cols_d[2]], df_p[cols_p[2]])
    er_T = run_mape(df_d['Total'], df_p['Total'])
    return er_I, er_R, er_D,er_T

def calc_rec(T, conf, death):
    recT = np.zeros((len(conf),))
    recT[T:] = conf[:-T] - death[:-T]
    infecT =  conf - death - recT
    return recT, infecT

def plot_unique(df_avg, df_d,col_d,title,fs,savefile,idx,esp=None,is_rt = False, **kwargs):
    fig, ax = plt.subplots(figsize=(15,10))
    if is_rt:
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(x, ',.2f')))
    else:
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(x, ',.2f').rstrip('0').rstrip('.')))
    
    matplotlib.rcParams.update({'font.size': fs})
    stp = 5
    lx = 0
    if esp == None:
        esp = len(df_avg) - 30
    if isinstance(df_avg, list): 
        train_lim = len(df_avg[0])-esp
        leg = kwargs['leg']
        for i,df in enumerate(df_avg):
            df = pd.DataFrame({leg[i]:df[:]})
            idx_a = idx.intersection(df_d.index)
            df.set_index(idx_a)
            df.plot(ax=ax,lw=2)
            
            if len(df) > lx:
                lx = len(df)-1
    else:
        train_lim = len(df_avg)-esp
        df_avg = pd.DataFrame({'Mean':df_avg[:]})
        df_avg.set_index(idx)
        
        df_avg.plot( ax=ax, c='r',lw=2)
        lx = len(df_avg)-1
        nidx = len(df_avg)
        if 'plot_all_days' in kwargs.keys() and kwargs['plot_all_days'] == True:
            total = kwargs['Total'][kwargs['col']]
            if 'faixa' in kwargs.keys() and kwargs['faixa'] == True:
                lim_min = total.min(1)[:nidx]
                lim_max = total.max(1)[:nidx]
                plt.fill_between(idx, lim_min, 
                     lim_max, alpha = 0.25, label = 'Infection estimative range')
            else:
                colors = pl.cm.cool(np.linspace(0,1,len(total.columns)))
                total.plot(ax=ax, color=colors, alpha=0.6,legend=None,linewidth=3)
    
    if df_d is not None:
        df_d[[col_d]].plot(ax=ax, c='k', linestyle='-.',lw=2,marker='o')
    
    ax.axvline(train_lim-1, ymin = 0, linestyle = ':', c = 'k',lw=2)
    ax.yaxis.grid(lw=1)
    plt.title(title)
    ax.set_xlabel('Date')
    tks = np.linspace(idx.index[0], lx, num=stp, endpoint=True)
    tks = tks.astype(int)
    
    plt.xticks(tks,idx.iloc[tks])
    plt.xlim(idx.index[0], lx)
    
    bottom, top = plt.ylim()
    left, right = plt.xlim()
    
    dist = 3*(right - left)/20
    off = (top - bottom)/20
    plt.text(train_lim - dist, bottom + off, "Training")
    plt.text(train_lim + dist/6, bottom + off, "Test")
    dirname = os.path.dirname(savefile)
    fig.tight_layout()
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok = True)
    plt.savefig(savefile)
    
    plt.close()
def plot(df_data, df_pred,r,pasta,pasta_graph,fs = 24,T = None, **kwargs):
    if isinstance(df_pred, list):
        plot_mult(df_data, df_pred,r,pasta,pasta_graph,fs)
        return
    pasta_save = f'{pasta}/{pasta_graph}'
    dc = pd.read_csv(f'data/dados - Agrupamento.csv')
    if not os.path.isdir(pasta_save):
        os.makedirs(pasta_save, exist_ok = True)
    df_d = get_data(df_data, r,is_pred = False)
    df_p = get_data(df_pred, r)
    df_d = sort_data(df_d)
    
    if T is not None:
        rec, inf = calc_rec(T,df_d['Confirmados'], df_d['Óbitos'] )
        df_d.loc[:,'Rt'] = rec
        df_d.loc[:,'At'] = inf
    
    df_conf = df_p[['Infected', 'Recovered', 'Death']].sum(axis=1)
    perday = np.zeros((len(df_conf),))
    perday[0] = np.nan
    perday[1:] = df_conf.iloc[1:].values - df_conf.iloc[:-1].values
    df_p.loc[:,'Daily new cases'] = perday
    df_pA = df_p[~df_p['Used in Train']]
    esp = len(df_pA)
    title =  translate(r)
    idx = df_p['Data']
    df_d.set_index('Data',inplace = True)
    idx_d = df_d.index
    idx_d = idx_d.intersection(idx)
    
    perday = np.zeros((len(df_d),))
    perday[0] = np.nan
    perday[1:] = df_d['Confirmados'].iloc[1:].values - df_d['Confirmados'].iloc[:-1].values
    df_d.loc[:,'Daily new cases (Real data)'] = perday

    df_d = df_d.loc[idx_d].rename(columns = {'At':f'Active Cases (Real data)',
                                           'Rt':'Recovered Cases (Real data)',
                                           'Óbitos':'Deceased (Real data)',
                                           'Confirmados': 'Confirmed (Real data)'})
    
    
    ####################               PLOT INFECTADOS                #######################
    plot_unique(df_p['Infected'], df_d,'Active Cases (Real data)',f'Infected - {title}',fs,f'{pasta_save}/Infected\\{r}_Inf.png',idx, esp = esp, **kwargs, col = 'Infected')
    ####################               PLOT Recovered                 #######################
    plot_unique(df_p['Recovered'], df_d,'Recovered Cases (Real data)',f'Cumulative Recovered - {title}',fs,f'{pasta_save}/recovered\\{r}_Rec.png',idx, esp = esp, **kwargs, col = 'Recovered')
    ####################               PLOT ÓBITOS                    #######################
    plot_unique(df_p['Death'], df_d,'Deceased (Real data)',f'Cumulative Deceased - {title}',fs,f'{pasta_save}/Death\\{r}_Death.png',idx, esp = esp, **kwargs, col = 'Death')
    ###################                PLOT R(t)                      #######################
    plot_unique(df_p['Rt'], None,None,r'$R_0(t)$ - '+f'{title}',fs,f'{pasta_save}/Rt\\{r}_Rt.png',idx, esp = esp,is_rt = True, **kwargs, col = 'Rt')
    ###################                PLOT R(t)                      #######################
    plot_unique(df_p[['Infected', 'Recovered', 'Death']].sum(axis=1), df_d,'Confirmed (Real data)',f'Cumulative Confirmed - {title}',fs,f'{pasta_save}/Confirmed\\{r}_Conf.png',idx, esp = esp)


    
    df_aux = movingAvg(df_d['Daily new cases (Real data)'],7)
    df_d.loc[:,'Daily new cases (Real data - MM)'] = df_aux
    df_pA.set_index('Data',inplace=True)
    idx_a = df_aux.index
    idx_a = idx_a.intersection(df_pA.index)
    df_aux1 = df_aux.loc[idx_a]
    df_pA = df_pA.loc[idx_a]

    ###################                PLOT New Cases                      #######################
    plot_unique(df_p['Daily new cases'], df_d, 'Daily new cases (Real data)',
                f'Daily New Cases - {title}', fs, f'{pasta_save}/Newcases/{r}_newCases.png', idx, esp=esp, leg=['Daily new cases', 'Moving Average of real data'])
    ##################                PLOT New Cases MM                   #######################
    plot_unique(df_p['Daily new cases'], df_d, 'Daily new cases (Real data - MM)',
                f'Daily New Cases - {title}', fs, f'{pasta_save}/Newcases/MM/{r}_newCasesMM.png', idx, esp=esp, leg=['Daily new cases', 'Moving Average of real data'])


def plot_mult(df_data, df_pred,r,pasta,pasta_graph,fs = 24,**kwargs):
    pasta_save = f'{pasta}/{pasta_graph}'
    dc = pd.read_csv(f'data/dados - Agrupamento.csv')
    if not os.path.isdir(pasta_save):
        os.makedirs(pasta_save, exist_ok = True)
    if r in df_data["SP-Subregião"].unique():
        df_d = df_data[df_data["SP-Subregião"] == r]
        df_p = []
        for df in df_pred:
            df_p.append(df[df["SP-Subregião"] == r])
    elif r in dc['Região'].unique():
        est = dc[dc['Região'] == r]['SP-Subregião'].unique()
        df_d = pd.DataFrame()
        
        df_p = pd.DataFrame()
        for e in est:
            df_d =df_d.append(df_data[df_data['SP-Subregião'] == e])
            df_p =df_p.append(df_pred[df_pred['SP-Subregião'] == e])
        df_d = df_d.groupby(['Data'], as_index=False).sum()
        df_p = df_p.groupby(['Data'], as_index=False).sum()
        df_p.loc[:,'Rt'] = df_p['Rt']/len(df_p['Rt'])
    idx = df_p[0].set_index(df_p[0]['Data']).index
    print(df_p)
    title =  translate(r)
    
    df_d.set_index('Data',inplace = True)
    idx_d = df_d.index
    idx_d = idx_d.intersection(idx)
    df_d = df_d.loc[idx_d].rename(columns = {'At':f'Active Cases (Real data)',
                                           'Rt':'Recovered Cases (Real data)',
                                           'Óbitos':'Deceased (Real data)',
                                           'Confirmados': 'Confirmed (Real data)'})
    leg = [r'Transient $\beta(t)$', r'Constant $\beta$']
    ####################               PLOT INFECTADOS                #######################
    plt_df = []
    for df in df_p:
        plt_df.append(df['Infected'])
    plot_unique(plt_df, df_d,'Active Cases (Real data)',f'Infected - {title}',fs,f'{pasta_save}/Infected\\{r}_Inf.png',idx, leg = leg, **kwargs)
    ####################               PLOT Recovered                 #######################
    plt_df = []
    for df in df_p:
        plt_df.append(df['Recovered'])
    plot_unique(plt_df, df_d,'Recovered Cases (Real data)',f'Recovered - {title}',fs,f'{pasta_save}/Recovered\\{r}_Rec.png',idx, leg = leg, **kwargs)
    ####################               PLOT ÓBITOS                    #######################
    plt_df = []
    for df in df_p:
        plt_df.append(df['Death'])
    plot_unique(plt_df, df_d,'Deceased (Real data)',f'Deceased - {title}',fs,f'{pasta_save}/Death\\{r}_Death.png',idx, leg = leg, **kwargs)
    ###################                PLOT R(t)                      #######################
    plt_df = []
    for df in df_p:
        plt_df.append(df['Rt'])
    plot_unique(plt_df, None,None,f'R(t) - {title}',fs,f'{pasta_save}\\{r}_Rt.png',idx, leg = leg,is_rt = True, **kwargs)
    ###################                PLOT R(t)                      #######################
    plt_df = []
    for df in df_p:
        plt_df.append(df[['Infected', 'Recovered', 'Death']].sum(axis=1))
    plot_unique(plt_df, df_d,'Confirmed (Real data)',f'Accumulated Confirmed - {title}',fs,f'{pasta_save}/Confirmed\\{r}_Conf.png',idx, leg = leg, **kwargs)

def eval_ivp(df,interval,y0, method = 'LSODA'):
    def sird(t,y,df):
        t = int(np.rint(t))
        beta = df['beta(t)'].iloc[t]
        gamma_d = df['gamma_Death'].iloc[t]
        gamma_r = df['gamma_Rec'].iloc[t]
        S,I,R,D = y
        dS = -I * beta * S
        dI =  I * beta * S - I*(gamma_d + gamma_r)
        dR =  I * gamma_r
        dD =  I * gamma_d
        return [dS, dI, dR,dD]
    IVP = solve_ivp(sird,interval, y0,
                        t_eval=np.arange(interval[0], interval[1]+1, 1), vectorized=False, method=method, args = ([df]))
    return IVP

def run_ivp(df,reg,dc):
    df = df.drop('Unnamed: 0', axis=1, errors='ignore')
    df = df.round({'Infected': 0, 'Recovered': 0, 'Death': 0})
    df['Data'] = pd.to_datetime(df.Data)
    df.sort_values(by='Data', inplace=True, ascending=True)
    d0 = df.Data.iloc[0]
    idx = (df.Data - d0).dt.days
    df.set_index(idx.values, inplace=True)
    df['Data'] = df['Data'].dt.strftime('%m/%d/%Y')

    tlim = len(df) -1
    t = [0,tlim]
    pop = get_pop(reg, dc)
    cols = ['Infected', 'Recovered', 'Death']

    y0 = df[cols].iloc[0].to_list()
#     y0[0] = y0[0] + 300
    y0 = y0/pop
    s0 = 1 - sum(y0)
    y0 = [s0] + y0.tolist()
    
    IVP = eval_ivp(df,t,y0, method='RK45')
    
    for i,c in enumerate(cols,1):
        df[c] = IVP.y[i] * pop
    df = df.round({'Infected':0, 'Recovered':0, 'Death':0})
    S = 1 - (IVP.y[1] +IVP.y[2]+IVP.y[3])
    new_rt = df['beta(t)'] / (df['gamma_Rec'] + df['gamma_Death']) * S
    df['Rt'] = new_rt
    return df



def run_unifica(dtime,case, prev=0,regs = None, unify = True, crop = 10, MM = False,
                dia_ini = 10, dia_fim = 30, rerun = False,save_sufix='',
                pasta=None, inner_dir=False, is_SIR=False, dir_sufix=None):
    if case == 'state':
        if pasta is None:
            pasta = f'Run_States/{dtime}'
        dc = pd.read_csv('data/dados - states.csv')
        file_d = f'data/dados - Data_states.csv'
        df_data = pd.read_csv(file_d, index_col = False)
        if regs == None:
            regs = ['Brasil', 'Norte', 'Nordeste', 'Sul', 'Sudeste', 'Centro-Oeste']
    elif case == 'subregion':

        dc = pd.read_csv('data/dados - subregions.csv')
        if pasta is None:
            pasta = f'Run_Semanal/{dtime}'
        file_d = f'data/dados - Data_subregions.csv'
        df_data = pd.read_csv(file_d, index_col = False)
        if regs ==  None:
            regs = df_data['SP-Subregião'].unique().tolist() + ['São Paulo (Estado)']
    elif case == 'city':
        dc = pd.read_csv('data/dados - Cidades.csv')
        if pasta is None:
            pasta = f'Run_City/{dtime}'
        file_d = f'data/dados - Data_cidades.csv'
        df_data = pd.read_csv(file_d, index_col = False)
        if regs ==  None:
            regs = df_data['SP-Subregião'].unique().tolist() 
    elif case == 'JHU':
        if pasta is None:
            pasta = f'Run_JHU/{dtime}'
        file_d  ='JHU'
        if regs ==  None:
            regs = ['Canada', 'Germany']
        df_data = pd.DataFrame()
        for r in regs:
            df_aux = read_global(r)
            df_data = df_data.append(df_aux)
            

    df_geral = pd.DataFrame()
    df_geral_min = pd.DataFrame()
    df_geral_max = pd.DataFrame()
    if unify:
        for r in regs:
            [df_,df_min,df_max] = unifica(dia_ini, dia_fim,prev, r, pasta = pasta,df_geral = None, crop = crop,
                               inner_dir = inner_dir, file1=file_d, MM = MM, dir_sufix=dir_sufix)
            
            if rerun:
                df_ = run_ivp(df_,r,dc)
                df_min = run_ivp(df_min,r,dc)
                df_max = run_ivp(df_max,r,dc)

            df_geral = df_geral.append(df_)
            df_geral_min = df_geral_min.append(df_min)
            df_geral_max = df_geral_max.append(df_max)
            
            
        
        date_str = df_geral[df_geral['Used in Train']]['Data'].iloc[-1]
        dtime = datetime.strptime(date_str,'%m/%d/%Y')
        pred_day = dtime.strftime('%Y-%b-%d')+save_sufix
        print(pred_day)
        if case == 'state':
            dir_res = f'Val-Results-states/{pred_day}/{dia_ini}-{dia_fim}'
        elif case == 'subregion':
            dir_res = f'Val-Results/{pred_day}/{dia_ini}-{dia_fim}' 
        elif case == 'city':
            dir_res = f'Val-Results-City/{pred_day}/{dia_ini}-{dia_fim}' 
        elif case == 'JHU':
            dir_res = f'Val-Results-JHU/{pred_day}/{dia_ini}-{dia_fim}' 
        if MM:
            dir_res = dir_res + '/MM'
        if not os.path.isdir(dir_res):
                os.makedirs(dir_res, exist_ok = True)
        df_geral.to_csv(f'{dir_res}/pred_all.csv')
        df_geral_min.to_csv(f'{dir_res}/pred_all_min.csv')
        df_geral_max.to_csv(f'{dir_res}/pred_all_max.csv')
        
    else:
        df_geral = unifica(dia_ini, dia_fim,prev, regs[0], pasta = pasta,df_geral = df_geral, crop = 10, inner_dir = False, file1=file_d)
        date_str = df_geral[df_geral['Used in Train']]['Data'].iloc[-1]
        dtime = datetime.strptime(date_str,'%m/%d/%Y')
        pred_day = dtime.strftime('%Y-%b-%d')
        if case == 'state':
            dir_res = f'Val-Results-states/{pred_day}/{dia_ini}-{dia_fim}'
        elif case == 'subregion':
            dir_res = f'Val-Results/{pred_day}/{dia_ini}-{dia_fim}'
        elif case == 'city':
            dir_res = f'Val-Results-City/{pred_day}/{dia_ini}-{dia_fim}' 
        elif case == 'JHU':
            dir_res = f'Val-Results-JHU/{pred_day}/{dia_ini}-{dia_fim}' 
        
        if MM:
            dir_res = dir_res + '/MM'
    df_att = pd.DataFrame()
    for r in regs:
        print(r)
        df_geral_t = pd.read_csv(f'{dir_res}/pred_all.csv',index_col = 0)
        df_geral_min = pd.read_csv(f'{dir_res}/pred_all_min.csv',index_col = 0)
        df_geral_max = pd.read_csv(f'{dir_res}/pred_all_max.csv',index_col = 0)
        

            
        df_plt = get_data(df_geral_t,r)
        df_plt_min = get_data(df_geral_min,r)
        df_plt_max = get_data(df_geral_max,r)
        
        if len(df_plt) == 0:
            continue
        if rerun:
            plot(df_data, df_plt,r,dir_res,'Graficos-att_rerun',fs = 24)
            plot(df_data, df_plt_min,r,dir_res,'Graficos-min_rerun',fs = 24)
            plot(df_data, df_plt_max,r,dir_res,'Graficos-max_rerun',fs = 24)
        else:   
            plot(df_data, df_plt,r,dir_res,'Graficos-att',fs = 24)
            plot(df_data, df_plt_min,r,dir_res,'Graficos-min',fs = 24)
            plot(df_data, df_plt_max,r,dir_res,'Graficos-max',fs = 24)
        df_att = df_att.append(df_geral_t[df_geral_t['SP-Subregião'] == r].iloc[:40])
    df_att.to_csv(f'{dir_res}/pred_att.csv')
