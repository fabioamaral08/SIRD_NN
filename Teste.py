from SIRD_NN import Models as Mod
from datetime import datetime

class Teste(object):
    def __init__(self, regions, is_vac, is_mm, date, effs=[0,0], annotation='', is_dose = True, is_intermed = True, is_rec=False, train_rate=False):
        self.is_vac = is_vac
        self.is_mm = is_mm
        self.regions = regions
        self.date = date
        self.effs = effs
        self.train_rate = train_rate

        if is_vac:
            self.sufixo = '-Vac2Di-'
        else:
            self.sufixo = '-SIRD-'
        if annotation != '':
            self.sufixo += annotation+'-'

        if len(regions) == 1:
            self.dtime = regions[0] + self.sufixo
        else:
            self.dtime = self.sufixo
        d = datetime.strptime(date, '%m/%d/%Y')
        d_str = d.strftime('%Y-%b-%d')

        self.dtime += d_str
        if is_mm:
            self.sufixo += '-MM'
            self.dtime += '-MM'
        self.pasta = pasta = f'Run_Teste/{self.dtime}'

        self.is_dose = is_dose
        self.is_intermed = is_intermed
        self.is_rec = is_rec
        if is_vac:
            if is_dose:
                if is_intermed:
                    if is_rec:
                        self.model = Mod.SVIRD_2Di_Rec
                    elif train_rate:
                        self.model = Mod.SVIRD_2Di_vacrate
                    else:
                        self.model = Mod.SVIRD_2Di
                else:
                    if is_rec:
                        self.model = Mod.SVIRD_2D_Rec
                    else:
                        self.model = Mod.SVIRD_2D
            else:
                if is_intermed:
                    if is_rec:
                        self.model = Mod.SVIRD_1Di_rec
                    else:
                        self.model = Mod.SVIRD_1Di
                else:
                    if is_rec:
                        self.model = Mod.SVIRD_1D_rec
                    else:
                        self.model = Mod.SVIRD_1D
        else:
            self.model = Mod.SIRD


    def __str__(self):
        msg =  f'Titulo: {self.dtime}\n'
        msg += f'Pasta: {self.pasta}\n'
        msg += f'is_vac: {self.is_vac}\n'
        msg += f'is_mm: {self.is_mm}\n'
        msg += f'Date: {self.date}\n'
        msg += f'effs: [{self.effs[0]},{self.effs[1]}]\n'
        msg += f'Regs:\n'
        for r in self.regions:
            msg += f'\t{r}'
        return msg

    def save_desc(self):
        f = open(f'{self.pasta}/desc_test.txt','w+')
        f.write(self.__str__())
        f.close()

    def configs(self):

        return self.dtime, self.pasta, self.is_vac, self.is_mm, self.date, self.effs[0], self.effs[1], self.regions,\
                self.sufixo, self.model, self.is_dose, self.is_intermed, self.is_rec, self.train_rate
