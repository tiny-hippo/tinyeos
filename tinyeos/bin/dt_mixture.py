import os
import numpy as np
from scipy.optimize import root_scalar


class TinyDTEos():
    # to-do: outdated -- fix!

    def __init__(self, heavy_element='water'):
        self.logRho_max = 2.00
        self.logRho_min = -8.00
        self.logT_max = 6.00
        self.logT_min = 2.00

        self.cache_path = 'interpolants'
        self.interpDT_logP_x = self.load_interp('interpDT_logP_x.npy')
        self.interpDT_logP_y = self.load_interp('interpDT_logP_y.npy')
        self.interpPT_logRho_x = self.load_interp('interpPT_logRho_x.npy')
        self.interpPT_logRho_y = self.load_interp('interpPT_logRho_y.npy')
        self.interpDT_logS_x = self.load_interp('interpDT_logS_x.npy')
        self.interpDT_logS_y = self.load_interp('interpDT_logS_y.npy')
        self.interpDT_logU_x = self.load_interp('interpDT_logU_x.npy')
        self.interpDT_logU_y = self.load_interp('interpDT_logU_y.npy')

        if not heavy_element in ['water', 'rock']:
            exit('Invalid option for heavy element')

        self.interpDT_logP_z = self.load_interp(
            'interpDT_logP_z_' + heavy_element + '.npy')
        self.interpDT_logS_z = self.load_interp(
            'interpDT_logS_z_' + heavy_element + '.npy')
        self.interpDT_logU_z = self.load_interp(
            'interpDT_logU_z_' + heavy_element + '.npy')

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

    def load_interp(self, filename):
        ''' Loads the interpolant from the disk.'''
        src = os.path.join(self.cache_path, filename)
        if not os.path.isfile(src):
            print('Missing interpolant cache', src)
            exit()
        return(np.load(src, allow_pickle=True).item())


    def check_DT(self, logT, logRho):
        ''' Makes sure that logT and logRho are within the
        equation of state boundaries.
        '''
        assert(self.logT_min <= logT <= self.logT_max), \
            'logT out of bounds'
        assert(self.logRho_min <= logRho <= self.logRho_max), \
            'logRho out of bounds'

    def ideal_mixture(self, logT, logRho, X, Z, debug=False):
        self.check_DT(logT, logRho)

        X, Y, Z = self.check_composition(X, Z)

        if Z == 1:
            conv = True
            logP = self.interpDT_logP_z(logT, logRho)
            logRho_x = self.logRho_min
            logRho_y = self.logRho_min
            logRho_z = logRho

        elif X == 1:
            conv = True
            logP = self.interpDT_logP_x(logT, logRho)
            logRho_x = logRho
            logRho_y = self.logRho_min
            logRho_z = self.logRho_min

        elif Y == 1:
            conv = True
            logP = self.interpDT_logP_y(logT, logRho)
            logRho_x = self.logRho_min
            logRho_y = logRho
            logRho_z = self.logRho_min

        elif Z > 0:
            x0 = logRho + np.log10(Z)
            sol = root_scalar(self.optimize_1d, args=(logRho, logT, X, Y, Z),
                              method='secant', x0=x0, x1=0)
            conv = sol.converged
            logRho_z = sol.root
            logP = self.interpDT_logP_z(logT, logRho_z)
            if X > 0:
                logRho_x = self.interpPT_logRho_x(logT, logP)
            else:
                logRho_x = self.logRho_min
            if Y > 0:
                logRho_y = self.interpPT_logRho_y(logT, logP)
            else:
                logRho_y = self.logRho_min

        elif X > 0:
            x0 = logRho + np.log10(X)
            sol = root_scalar(self.optimize_1d, args=(logRho, logT, X, Y, Z),
                              method='secant', x0=x0, x1=0)
            conv = sol.converged
            logRho_x = sol.root
            logP = self.interpDT_logP_x(logT, logRho_x)
            logRho_y = self.interpPT_logRho_y(logT, logP)
            logRho_z = self.logRho_min

        if(debug):
            res = self.optimize_1d(logRho_z, logRho, logT, X, Y, Z)
            logP_x = self.interpDT_logP_x(logT, logRho_x)
            logP_y = self.interpDT_logP_y(logT, logRho_y)

            print(sol)
            print('Residual: {:.2E}'.format(res))
            print('logP: {:.3f} {:.3f} {:.3f}'.format(logP, logP_x, logP_y))

        if not conv:
            pass
        elif not (self.logRho_min <= logRho_z <= self.logRho_max):
            conv = False
        elif not(self.logRho_min <= logRho_x <= self.logRho_max):
            conv = False
        elif not(self.logRho_min <= logRho_y <= self.logRho_max):
            conv = False

        return(conv, logRho_x, logRho_y, logRho_z, logP)

    def ideal_mixing_law(self, rho, rho_x, rho_y, rho_z, X, Y, Z):
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

        return(1 / rho - x_rho_x - y_rho_y - z_rho_z)

    def optimize_1d(self, xhat, logRho, logT, X, Y, Z):
        rho = 10**logRho
        if Z > 0:
            logP = self.interpDT_logP_z(logT, xhat)
            rho_z = 10**xhat
        elif (X > 0):
            logP = self.interpDT_logP_x(logT, xhat)
            rho_x = 10**xhat
            rho_z = 0

        if X == 0:
            rho_x = 0
        else:
            logRho_x = self.interpPT_logRho_x(logT, logP)
            rho_x = 10**logRho_x
        if Y == 0:
            rho_y = 0
        else:
            logRho_y = self.interpPT_logRho_y(logT, logP)
            rho_y = 10**logRho_y

        f = self.ideal_mixing_law(rho, rho_x, rho_y, rho_z, X, Y, Z)
        return f

    def evaluate(self, logT, logRho, X, Z):
        self.check_DT(logT, logRho)

        X, Y, Z = self.check_composition(X, Z)

        xhat = self.ideal_mixture(logT, logRho, X, Z)
        if not xhat[0]:
            exit('Failed in density root find')

        logRho_x = xhat[1]
        logRho_y = xhat[2]
        logRho_z = xhat[3]
        logP = xhat[4]

        # neglect entropy of mixing
        logS_x = self.interpDT_logS_x(logT, logRho_x)
        logS_y = self.interpDT_logS_y(logT, logRho_y)
        logS_z = self.interpDT_logS_z(logT, logRho_z)
        S = X * (10**logS_x) + Y * (10**logS_y) + Z * (10**logS_z)
        logS = np.log10(S)

        logU_x = self.interpDT_logU_x(logT, logRho_x)
        logU_y = self.interpDT_logU_y(logT, logRho_y)
        logU_z = self.interpDT_logU_z(logT, logRho_z)
        U = X * (10**logU_x) + Y * (10**logU_y) + Z * (10**logU_z)
        logU = np.log10(U)

        logP = np.float(logP)
        logS = np.float(logS)
        logU = np.float(logU)

        return np.array([logP, logS, logU])


if __name__ == '__main__':
    import argparse

    desc = 'Calculates logP, logS and logU of an ideal mixture'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('logT', type=float, help='logT [K]')
    parser.add_argument('logRho', type=float, help='logRho [g/cc]')
    parser.add_argument('X', type=float, help='Hydrogen fraction')
    parser.add_argument('Z', type=float, help='Heavy-element fraction')
    parser.add_argument('heavy_element', type=str, help='water, rock or iron')

    args = parser.parse_args()
    logRho = args.logRho
    logT = args.logT
    X = args.X
    Z = args.Z
    heavy_element = args.heavy_element

    T = TinyDTEos(heavy_element=heavy_element)
    res_eval = T.evaluate(logT, logRho, X, Z)
    res = np.array([logT, logRho, X, Z, res_eval[0],
                    res_eval[1], res_eval[2]])
    np.set_printoptions(precision=2)
    print('logT    logRho    X    Z     logP     logS    logE')
    print(res)
