import pandas as pd
import numpy as np
import SIRD_NN
import SIRD_NN.Utils as Utils
import SIRD_NN.Models as Mod
from datetime import datetime
import os

if __name__ == '__main__':
    Utils.atualiza_dados('Data_subregions')
    dt = pd.read_csv("data\\dados - Data_subregions.csv")   
    dc = pd.read_csv("data\\dados - subregions.csv")

    dia_ini = 10
    dia_fim = 30
    reg = dc['SP-Subregião'].to_list() + ['São Paulo (Estado)']
    prev = 0
    dtime = datetime.today().strftime("%y-%m-%d")
    pasta = f'Run_Semanal/{dtime}'
    for r in reg:
        df = Utils.get_data(dt,r, False)
        df = Utils.sort_data(df)
        os.makedirs(f'{pasta}/{r}', exist_ok = True)
        for dLen in range(dia_ini, dia_fim + 1):
            ds = df[["At"]]

            ini = ds[ds["At"] > 0]
            di = ini.index[0]
            nData = len(ini)
            isPrev = not prev == 0
            if isPrev:
                prevStep = prev
            else:
                prevStep = 60
            sl1 = nData - prev - dLen
            sl = nData - prev
            
            net, df_save = Utils.run_region(r, sl1, sl, dc, dt, step=14, mAvg=False,
                            is_SIR = True, model = Mod.SIR)
            
            df_save.to_csv(f'{pasta}/{r}/Subregions_Pred_{dLen}D_prev-{prev}-{r}.csv') 

    case = 'subregion'
    Utils.run_unifica(dtime,case, unify = True, crop = 60, MM = False, dia_ini = 10, dia_fim = 30, rerun = False) 