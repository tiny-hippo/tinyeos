import os
import numpy as np


class TinyPTEoS():
    # to-do: outdated -- fix!

    def __init__(self, heavy_element='water'):
        # build interpolants from scratch
        if not os.path.exists('interpolants'):
            exit('Could not find interpolants')
        # load interpolants from disk
        self.cache_path = 'interpolants'

        self.interpPT_logRho_x = self.load_interp('interpPT_logRho_x.npy')
        self.interpPT_logRho_y = self.load_interp('interpPT_logRho_y.npy')
        self.interpPT_logS_x = self.load_interp('interpPT_logS_x.npy')
        self.interpPT_logS_y = self.load_interp('interpPT_logS_y.npy')
        self.interpPT_logU_x = self.load_interp('interpPT_logU_x.npy')
        self.interpPT_logU_y = self.load_interp('interpPT_logU_y.npy')

        if heavy_element not in ['water', 'rock']:
            exit('Invalid option for heavy element')

        self.interpPT_logRho_z = self.load_interp(
            'interpPT_logRho_z_' + heavy_element + '.npy')
        self.interpPT_logS_z = self.load_interp(
            'interpPT_logS_z_' + heavy_element + '.npy')
        self.interpPT_logU_z = self.load_interp(
            'interpPT_logU_z_' + heavy_element + '.npy')


        self.logP_max = 15.00
        self.logP_min = 1.00
        self.logT_max = 6.00
        self.logT_min = 2.00
        self.logRho_max = 99
        self.logRho_min = -99

    @staticmethod
    def check_composition(X, Z):
        ''' Checks that the mass fractions sum up to less than one
        and sets them to zero or one if they are sufficiently close.
        '''
        assert (X + Z <= 1), 'X + Z must be smaller than 1'
        eps = 1e-3
        Y = 1 - X - Z
        composition = np.array([X, Y, Z])
        check_zero = np.isclose(composition, 0, atol=eps)
        if np.any(check_zero):
            X, Y, Z = (~check_zero) * composition

        check_one = np.isclose(composition, 1, atol=eps)
        if np.any(check_one):
            X, Y, Z = check_one * np.ones(3)
        return (X, Y, Z)

    @staticmethod
    def ideal_mixing_law(rho_x, rho_y, rho_z, X, Y, Z):
        if X == 0:
            x_rho_x = 0
        else:
            x_rho_x = X / rho_x
        if Y == 0:
            y_rho_y = 0
        else:
            y_rho_y = Y / rho_y
        if Z == 0:
            z_rho_z = 0
        else:
            z_rho_z = Z / rho_z

        return(1 / (x_rho_x + y_rho_y + z_rho_z))


    def load_interp(self, filename):
        ''' Loads the interpolant from the disk.'''
        src = os.path.join(self.cache_path, filename)
        if not os.path.isfile(src):
            print('Missing interpolant cache', src)
            exit()
        return(np.load(src, allow_pickle=True).item())

    def check_PT(self, logT, logP):
        ''' Makes sure that logT and logP are within the
        equation of state boundaries.
        '''
        assert(self.logT_min <= logT <= self.logT_max), \
            'logT out of bounds'
        assert(self.logP_min <= logP <= self.logP_max), \
            'logP out of bounds'

    def ideal_mixture(self, logT, logP, X, Z, debug=False):
        self.check_PT(logT, logP)
        X, Y, Z = self.check_composition(X, Z)

        if Z == 1:
            logRho_x = self.logRho_min
            logRho_y = self.logRho_min
            logRho_z = self.interpPT_logRho_z(logT, logP, grid=False)

        elif X == 1:
            logRho_x = self.interpPT_logRho_x(logT, logP, grid=False)
            logRho_y = self.logRho_min
            logRho_z = self.logRho_min

        elif Y == 1:
            logRho_x = self.logRho_min
            logRho_y = self.interpPT_logRho_y(logT, logP, grid=False)
            logRho_z = self.logRho_min

        elif Z > 0:
            logRho_z = self.interpPT_logRho_z(logT, logP, grid=False)
            if X > 0:
                logRho_x = self.interpPT_logRho_x(logT, logP, grid=False)
            else:
                logRho_x = self.logRho_min
            if Y > 0:
                logRho_y = self.interpPT_logRho_y(logT, logP, grid=False)
            else:
                logRho_y = self.logRho_min

        elif X > 0:
            logRho_x = self.interpPT_logRho_x(logT, logP, grid=False)
            logRho_y = self.interpPT_logRho_y(logT, logP, grid=False)
            logRho_z = self.logRho_min

        rho_x = 10**logRho_x
        rho_y = 10**logRho_y
        rho_z = 10**logRho_z
        rho = self.ideal_mixing_law(rho_x, rho_y, rho_z, X, Y, Z)
        logRho = np.log10(rho)

        return logRho

    def evaluate(self, logT, logP, X, Z):
        self.check_PT(logT, logP)
        X, Y, Z = self.check_composition(X, Z)

        logRho = self.ideal_mixture(logT, logP, X, Z)

        # to-do: add mixing entropy
        logS_x = self.interpPT_logS_x(logT, logP, grid=False)
        logS_y = self.interpPT_logS_y(logT, logP, grid=False)
        logS_z = self.interpPT_logS_z(logT, logP, grid=False)
        S = X * (10**logS_x) + Y * (10**logS_y) + Z * (10**logS_z)
        logS = np.log10(S)

        logU_x = self.interpPT_logU_x(logT, logP, grid=False)
        logU_y = self.interpPT_logU_y(logT, logP, grid=False)
        logU_z = self.interpPT_logU_z(logT, logP, grid=False)
        U = X * (10**logU_x) + Y * (10**logU_y) + Z * (10**logU_z)
        logU = np.log10(U)

        return (logRho, logS, logU)

if __name__ == '__main__':
    import argparse

    desc = 'Calculates logRho, logS and logU of an ideal mixture'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('logT', type=float, help='logT [K]')
    parser.add_argument('logP', type=float, help='logRho [Ba]')
    parser.add_argument('X', type=float, help='Hydrogen fraction')
    parser.add_argument('Z', type=float, help='Heavy-element fraction')
    parser.add_argument('heavy_element', type=str, help='water or rock')

    args = parser.parse_args()
    logP = args.logP
    logT = args.logT
    X = args.X
    Z = args.Z
    heavy_element = args.heavy_element

    T = TinyPTEoS(heavy_element=heavy_element)
    res_eval = T.evaluate(logT, logP, X, Z)
    res = np.array([logT, logP, X, Z, res_eval[0],
                    res_eval[1], res_eval[2]])
    np.set_printoptions(precision=2)
    print('logT    logP    X    Z     logRho     logS    logE')
    print(res)
