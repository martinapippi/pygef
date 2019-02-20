import numpy as np
import pandas as pd

# INPUT: insert manually the water level
water_level_NAP = -1.56  # (m) respect to the NAP


class BeenJeffrey:
    def __init__(self, gef):
        self.gef = None
        self.df_complete = None
        self.depth = None
        self.fs = None
        self.pre_excavated_depth = None
        self.water_level = None
        self.p_a = None
        self.qt = None
        self.df_complete = None

        self.gef = gef
        qc = gef.df['qc']
        self.depth = gef.df['depth']
        self.fs = gef.df['fs']
        self.pre_excavated_depth = gef.pre_excavated_depth
        zid = gef.zid
        try:
            self.u2 = gef.df['u2']
        except KeyError:
            raise SystemExit("ERROR: u2 not defined in .gef file, change classifier")

        self.water_level = zid - water_level_NAP

        # qt
        if self.gef.net_surface_area_quotient_of_the_cone_tip is not None and 'qc' \
                in self.gef.df.columns and 'u2' in self.gef.df.columns:
            self.qt = qc + gef.df['u2'] * (1 - gef.net_surface_area_quotient_of_the_cone_tip)
        else:
            self.qt = qc

    def classify(self):
        # calculation of sigma_v and u
        u = self.hydrostatic_water_pressure(self.water_level, self.gef.df['depth'])
        soil_type_been = []
        Ic = []
        sig0 = []
        series_Qt = []
        series_Fr = []
        for i, depth in enumerate(self.depth):
            qti = self.qt[i]
            fsi = self.fs[i]
            u2i = self.u2[i]
            ui = u[i]
            if i == 0:  # add the check for the pre-excavation
                if self.pre_excavated_depth is not None:
                    sigma_v0i = 18 * self.pre_excavated_depth  # use of a standard gamma=18 for the excavated soil
                else:
                    sigma_v0i = 0
                ic = self.type_index(fsi, qti, sigma_v0i, ui, u2i)
            else:
                depth1 = self.depth.iloc[i - 1]
                depth2 = depth
                sig0i = sig0[i - 1]
                # iteration: it starts assuming gamma of the sand and iterate until the real gamma is found.
                gamma1 = 20
                delta_sigma_v0i = self.delta_vertical_stress(depth1, depth2, gamma1)
                sigma_v0i = self.vertical_stress(sig0i, delta_sigma_v0i)
                ic = self.type_index(fsi, qti, sigma_v0i, ui, u2i)

                gamma2 = self.get_gamma(ic, depth)
                ii = 0
                max_it = 5
                while gamma2 != gamma1 and ii < max_it:
                    gamma1 = gamma2
                    delta_sigma_v0i = self.delta_vertical_stress(depth1, depth2, gamma1)
                    sigma_v0i = self.vertical_stress(sig0i, delta_sigma_v0i)
                    ic = self.type_index(fsi, qti, sigma_v0i, ui, u2i)
                    gamma2 = self.get_gamma(ic, depth)
                    ii += 1

            Qti = self.normalized_cone_resistance(qti, sigma_v0i, ui)
            Fri = self.normalized_friction_ratio(fsi, qti, sigma_v0i)

            series_Qt.append(Qti)
            series_Fr.append(Fri)
            sig0.append(sigma_v0i)
            Ic.append(ic)
            soil_type = self.type_index_to_soil_type(ic)
            soil_type_been.append(soil_type)
        df_Ic = pd.DataFrame(Ic, columns=['Ic'])
        df_soil_type = pd.DataFrame(soil_type_been, columns=['soil_type_been_jeffrey'])
        df_been = pd.concat([df_Ic, df_soil_type], axis=1, sort=False)
        df_u = pd.DataFrame(u, columns=['hydrostatic_pore_pressure'])
        df_Qt = pd.DataFrame(series_Qt, columns=['normalized_Qt'])
        df_Fr = pd.DataFrame(series_Fr, columns=['normalized_Fr'])
        self.df_complete = pd.concat([self.gef.df, df_u, df_been, df_Qt, df_Fr], axis=1, sort=False)
        return self.df_complete

    @staticmethod
    def hydrostatic_water_pressure(water_level, depth):
        hydrostatic_water_pressure = []
        for z in depth:
            if z <= water_level:
                hydrostatic_water_pressure.append(0)
            else:
                hydrostatic_water_pressure.append((z - water_level) * 9.81)  # kN/m3 = kPa
        return hydrostatic_water_pressure

    @staticmethod
    def type_index_to_gamma(ic):
        gamma = None
        if ic > 3.22:
            gamma = 11
        elif 2.76 < ic <= 3.22:
            gamma = 16
        elif 2.40 < ic <= 2.76:
            gamma = 18
        elif 1.80 < ic <= 2.40:
            gamma = 18
        elif 1.25 < ic <= 1.80:
            gamma = 18
        elif ic <= 1.25:
            gamma = 18
        return gamma

    @staticmethod
    def type_index_to_gamma_sat(ic):
        gamma_sat = None
        if ic > 3.22:
            gamma_sat = 11
        elif 2.76 < ic <= 3.22:
            gamma_sat = 16
        elif 2.40 < ic <= 2.76:
            gamma_sat = 18
        elif 1.80 < ic <= 2.40:
            gamma_sat = 19
        elif 1.25 < ic <= 1.80:
            gamma_sat = 20
        elif ic <= 1.25:
            gamma_sat = 20
        return gamma_sat

    def get_gamma(self, ic, depth):
        if depth <= self.water_level:
            gamma = self.type_index_to_gamma(ic)
        else:
            gamma = self.type_index_to_gamma_sat(ic)
        return gamma

    @staticmethod
    def type_index_to_soil_type(ic):
        soil_type = None
        if ic > 3.22:
            soil_type = 'Peat'
        elif 2.76 < ic <= 3.22:
            soil_type = 'Clays'
        elif 2.40 < ic <= 2.76:
            soil_type = 'Clayey silt to silty clay'
        elif 1.80 < ic <= 2.40:
            soil_type = 'Silty sand to sandy silt'
        elif 1.25 < ic <= 1.80:
            soil_type = 'Sands: clean sand to silty'
        elif ic <= 1.25:
            soil_type = 'Gravelly sands'
        return soil_type

    @staticmethod
    def effective_stress(sigma_v0, u):
        sigma_v0_eff = sigma_v0 - u
        return sigma_v0_eff

    def normalized_cone_resistance(self, qt, sigma_v0, u):
        sigma_v0_eff = self.effective_stress(sigma_v0, u)
        if sigma_v0_eff > 0 and (qt - sigma_v0 * (10 ** -3)) > 0:
            Qt = (qt - sigma_v0 * (10 ** -3)) / (sigma_v0_eff * (10 ** -3))
        else:
            Qt = 1
        return Qt

    @staticmethod
    def normalized_friction_ratio(fs, qt, sigma_v0):
        if (qt - sigma_v0 * (10 ** -3)) > 0 and fs > 0:
            fr = fs * 100 / (qt - sigma_v0 * (10 ** -3))
        else:
            fr = 0.1
        return fr

    @staticmethod
    def delta_vertical_stress(depth1, depth2, gamma):
        delta_sigma_v0 = (depth2 - depth1) * gamma
        return delta_sigma_v0

    @staticmethod
    def vertical_stress(sig0, delta_sigma_v0):
        sigma_v0 = sig0 + delta_sigma_v0
        return sigma_v0

    @staticmethod
    def excess_pore_pressure_ratio(qt, sigma_v0, u, u2):
        return (u2 - u * (10 ** -3)) / (qt - sigma_v0 * (10 ** -3))

    def type_index(self, fs, qt, sigma_v0, u, u2):
        Qt = self.normalized_cone_resistance(qt, sigma_v0, u)
        Fr = self.normalized_friction_ratio(fs, qt, sigma_v0)
        Bq = self.excess_pore_pressure_ratio(qt, sigma_v0, u, u2)
        I_c = ((3 - np.log10(Qt*(1-Bq)+1)) ** 2 + (1.5 + 1.3*np.log10(Fr)) ** 2) ** 0.5
        return I_c

