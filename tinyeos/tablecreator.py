import os
import numpy as np
# from scipy.interpolate import NearestNDInterpolator
from fortranformat import FortranRecordWriter
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from tinyeos.tinydteos import TinyDT
from tinyeos.tinypteos import TinyPT
from tinyeos.support import get_1d_spline, NearestND


class TableCreatorDT(TinyDT):
    def __init__(
        self,
        which_heavy="water",
        which_hhe="cms",
        build_interpolants=False,
        fix_vals=True,
    ):
        super().__init__(
            which_heavy=which_heavy,
            which_hhe=which_hhe,
            build_interpolants=build_interpolants,
        )
        self.fix_vals = fix_vals
        self.smooth_vals = False
        self.check_if_file_exists = False
        self.do_single = False
        self.do_pure = False
        self.do_parallel = True
        self.debug = False

        # self.fix_method = fix_method  # nearest, intermediate, interpolate
        self.header1_line = FortranRecordWriter("(99(a14))")
        self.header2_line = FortranRecordWriter("(/,7x,a)")
        self.header3_line = FortranRecordWriter(
            "(a4,1x,3(a9,1x),7(a12,1x),1(a7,1x),1(a11),3(a9,1x),1(a9,1x))")

        self.header1 = [
            "version",
            "X",
            "Z",
            "num logTs",
            "logT min",
            "logT max",
            "del logT",
            "num logQs",
            "logQ min",
            "logQ max",
            "del logQ",
        ]
        self.header2 = ["logQ = logRho - 2*logT + 12"]
        self.header3 = [
            "logT",
            "logPgas",
            "logE",
            "logS",
            "chiRho",
            "chiT",
            "Cp",
            "Cv",
            "dE_dRho",
            "dS_dT",
            "dS_dRho",
            "mu",
            "log_free_e",
            "gamma1",
            "gamma3",
            "grad_ad",
            "eta",
        ]

        self.content1_line = FortranRecordWriter(
            "(i14,2f14.4,2(i10,4x,3(f14.4)))")
        self.content2_line = FortranRecordWriter("(2x,f14.6/)")
        self.content3_line = FortranRecordWriter(
            "(f4.2,3(f10.5),7(1pe13.5),1(0pf9.5),4(0pf10.5),1(0pf11.5))")

    def create_tables(
        self,
        del_logT,
        del_logQ,
        logT_min,
        logT_max,
        logQ_min,
        logQ_max,
        del_X,
        del_Z,
        fname_prefix,
    ):

        self.logT_min = logT_min
        self.logT_max = logT_max
        self.logQ_min = logQ_min
        self.logQ_max = logQ_max
        self.del_logT = del_logT
        self.del_logQ = del_logQ
        self.num_logTs = np.int(1 + np.ceil((logT_max - logT_min) / del_logT))
        self.num_logQs = np.int(1 + np.rint((logQ_max - logQ_min) / del_logQ))
        self.fname_prefix = fname_prefix

        if self.do_single:
            X = 0.50
            Z = 0.20
            print(f"Creating a single table for X = {X:.2f} and Z = {Z:.2f}")
            results = self.make_eos_files(X, Z)
            return results

        if self.do_pure:
            Xs = np.array([1, 0, 0])
            Zs = np.array([0, 0, 1])

        else:
            Zs = np.arange(0, 1 + del_Z, del_Z)
            Xs = [np.arange(0, (1 - Z) + del_X, del_X) for Z in Zs]
            Zs = [Zs[i] * np.ones(len(Xs[i])) for i in range(len(Zs))]
            Xs = np.concatenate(Xs)
            Zs = np.concatenate(Zs)

        if self.do_parallel:
            num_cores = cpu_count()
            print(f"Creating grid of tables with prefix {self.fname_prefix}",
                  f"using {num_cores:.0f} cores.")
            Parallel(n_jobs=num_cores)(
                delayed(self.make_eos_files)(X=Xs[i], Z=Zs[i])
                for i in range(len(Xs)))
        else:
            for i in range(len(Xs)):
                X = Xs[i]
                Z = Zs[i]
                print(f"Creating single table with prefix {self.fname_prefix}",
                      f"with X = {X:.2f} and Z = {Z:.2f}.")
                self.make_eos_files(X, Z)

    def make_eos_files(self, X, Z):
        assert X + Z <= 1
        X = np.round(X, 2)
        Z = np.round(Z, 2)

        # placeholders
        version_number = 11

        content1 = [
            version_number,
            X,
            Z,
            self.num_logTs,
            self.logT_min,
            self.logT_max,
            self.del_logT,
            self.num_logQs,
            self.logQ_min,
            self.logQ_max,
            self.del_logQ,
        ]

        if X >= 0.1:
            xtext = f"{100 * X:.0f}x.data"
        else:
            xtext = f"0{100 * X:.0f}x.data"

        if Z >= 0.1:
            ztext = f"{100 * Z:.0f}z"
        else:
            ztext = f"0{100 * Z:.0f}z"

        fname = "".join([self.fname_prefix, ztext, xtext])
        fdir = "data/mesa/eosDT"
        dst = os.path.join(fdir, fname)

        results = np.zeros((self.num_logQs, self.num_logTs, self.num_vals),
                           dtype=np.float32)
        if self.debug:
            dbg_arr = np.zeros((self.num_logQs, self.num_logTs, self.num_vals))

        for i in range(self.num_logQs):
            logQ = self.logQ_min + i * self.del_logQ
            for j in range(self.num_logTs):
                logT = self.logT_min + j * self.del_logT
                logRho = np.max([self.logRho_min, logQ + 2 * logT - 12])
                logRho = np.min([logRho, self.logRho_max])
                res = self.evaluate(logT, logRho, X, Z)

                if (np.all(res == -1) or np.any(np.isnan(res))
                        or np.any(np.isinf(res)) or res[self.i_cp] <= 0
                        or res[self.i_cv] <= 0 or res[self.i_dS_dT] <= 0
                        or res[self.i_logU] > 30 or res[self.i_cp] > 1e30
                        or res[self.i_cv] > 1e30 or res[self.i_chiRho] <= 0
                        or res[self.i_chiT] <= 0 or res[self.i_gamma1] >= 9.99
                        or res[self.i_gamma3] >= 9.99):

                    res[self.i_logT] = logT
                    res[self.i_logRho] = logRho
                    if self.debug:
                        dbg_arr[i, j, :] = res
                        dbg_arr[i, j, self.i_logT] = logT
                        dbg_arr[i, j, self.i_logP] = res[self.i_logP]
                        dbg_arr[i, j, self.i_logRho] = logRho

                    if self.fix_vals:
                        res[self.i_logRho + 1:] = np.nan
                    else:
                        res[np.where(np.isnan(res))] = -1
                        res[np.where(np.isinf(res))] = -1

                results[i, j, :] = res

        if self.fix_vals:
            x = results[:, :, self.i_logT].reshape(-1)
            y = results[:, :, self.i_logRho].reshape(-1)
            z = results[:, :, self.i_logRho + 1:].reshape(
                (-1, self.num_vals - 2))
            idcs = np.where(np.isfinite(z))[0]

            if self.debug:
                print(z.size, len(np.where(np.isnan(z))[0]))

            nnd_interp = NearestND((x[idcs], y[idcs]), z[idcs], rescale=True)

            zhat = nnd_interp((x, y), k=5, weights="distance").reshape(
                (self.num_logQs, self.num_logTs, -1))

            if self.smooth_vals:
                results[:, :, self.i_logRho + 1:] = zhat
            else:
                res2 = np.copy(results)
                res2[:, :, self.i_logRho + 1:] = zhat
                idcs = np.where(np.isfinite(results))
                res2[idcs] = results[idcs]
                results = np.copy(res2)
                del res2

            del x, y, z, zhat, idcs

        if self.check_if_file_exists:
            if os.path.isfile(dst):
                return

        with open(dst, "w") as file:
            file.write(self.header1_line.write(self.header1))
            file.write("\n")
            file.write(self.content1_line.write(content1))
            file.write("\n")

            for i in range(self.num_logQs):
                logQ = self.logQ_min + i * self.del_logQ
                file.write(self.header2_line.write(self.header2))
                file.write("\n")
                file.write(self.content2_line.write([logQ]))
                file.write("\n")
                file.write(self.header3_line.write(self.header3))
                file.write("\n")

                for j in range(self.num_logTs):
                    res = results[i, j, :]
                    content3 = [
                        res[self.i_logT],
                        res[self.i_logP],
                        res[self.i_logU],
                        res[self.i_logS],
                        res[self.i_chiRho],
                        res[self.i_chiT],
                        res[self.i_cp],
                        res[self.i_cv],
                        res[self.i_dE_dRho],
                        res[self.i_dS_dT],
                        res[self.i_dS_dRho],
                        res[self.i_mu],
                        res[self.i_lfe],
                        res[self.i_gamma1],
                        res[self.i_gamma3],
                        res[self.i_grad_ad],
                        res[self.i_eta],
                    ]
                    file.write(self.content3_line.write(content3))
                    file.write("\n")
                file.write("\n")
            file.write("\n")
            file.write("\n")
            file.close()

        if self.debug:
            debug_dir = os.path.join("data", "debug/eosDT")
            dname = fname.replace(self.fname_prefix, "debugDT_")
            dname = dname.replace(".data", ".npy")
            dst = os.path.join(debug_dir, dname)
            np.save(dst, dbg_arr)

        return results


class TableCreatorPT(TinyPT):
    def __init__(
        self,
        which_heavy="water",
        which_hhe="cms",
        build_interpolants=False,
        fix_vals=True,
    ):
        super().__init__(
            which_heavy=which_heavy,
            which_hhe=which_hhe,
            build_interpolants=build_interpolants,
        )
        self.fix_vals = fix_vals
        self.smooth_vals = False
        self.check_if_file_exists = False
        self.do_single = False
        self.do_pure = False
        self.do_parallel = True
        self.debug = False

        self.header1_line = FortranRecordWriter("(99(a14))")
        self.header2_line = FortranRecordWriter("(/,7x,a)")
        self.header3_line = FortranRecordWriter("(a7,10x,99(a11,7x))")

        self.header1 = [
            "version",
            "X",
            "Z",
            "num logTs",
            "logT min",
            "logT max",
            "del logT",
            "num logWs",
            "logW min",
            "logW max",
            "del logW",
        ]
        self.header2 = ["logW = logPgas - 4*logT"]
        self.header3 = [
            "logT",
            "logRho",
            "logE",
            "logS",
            "chiRho",
            "chiT",
            "Cp",
            "Cv",
            "dE_dRho",
            "dS_dT",
            "dS_dRho",
            "mu",
            "log_free_e",
            "gamma1",
            "gamma3",
            "grad_ad",
            "eta",
        ]

        self.content1_line = FortranRecordWriter(
            "(i14,2f14.4,2(i10,4x,3(f14.4)))")
        self.content2_line = FortranRecordWriter("(2x,f14.6/)")
        self.content3_line = FortranRecordWriter("(f11.6,99(e18.9))")

    def create_tables(
        self,
        del_logT,
        del_logW,
        logT_min,
        logT_max,
        logW_min,
        logW_max,
        del_X,
        del_Z,
        fname_prefix,
    ):

        self.logT_min = logT_min
        self.logT_max = logT_max
        self.logW_min = logW_min
        self.logW_max = logW_max
        self.del_logT = del_logT
        self.del_logW = del_logW
        self.num_logTs = np.int(1 + np.ceil((logT_max - logT_min) / del_logT))
        self.num_logWs = np.int(1 + np.rint((logW_max - logW_min) / del_logW))
        self.fname_prefix = fname_prefix

        if self.do_single:
            X = 0.80
            Z = 0.10
            print(f"Creating a single table for X = {X:.2f} and Z = {Z:.2f}")
            results = self.make_eos_files(Z, X)
            return results

        # for i, Z in enumerate(Zs):
        #     if Z != 0 and Z < 0.99:
        #         X_max = np.int(np.rint((1 - Z) / del_Z + 1))
        #         for j in range(X_max):
        #             self.make_eos_files(Z, Xs[j])
        #     elif Z > 0.99:
        #         self.make_eos_files(Z, 0)
        #     else:
        #         for X in Xs:
        #             self.make_eos_files(Z, X)

        if self.do_pure:
            Xs = np.array([1, 0, 0])
            Zs = np.array([0, 0, 1])

        else:
            Zs = np.arange(0, 1 + del_Z, del_Z)
            Xs = [np.arange(0, (1 - Z) + del_X, del_X) for Z in Zs]
            Zs = [Zs[i] * np.ones(len(Xs[i])) for i in range(len(Zs))]
            Xs = np.concatenate(Xs)
            Zs = np.concatenate(Zs)

        if self.do_parallel:
            num_cores = cpu_count()
            print(f"Creating grid of tables with prefix {self.fname_prefix}",
                  f"using {num_cores:.0f} cores.")
            Parallel(n_jobs=num_cores)(
                delayed(self.make_eos_files)(X=Xs[i], Z=Zs[i])
                for i in range(len(Xs)))
        else:
            for i in range(len(Xs)):
                X = Xs[i]
                Z = Zs[i]
                print(f"Creating single table with prefix {self.fname_prefix}",
                      f"with X = {X:.2f} and Z = {Z:.2f}.")
                self.make_eos_files(X, Z)

    def make_eos_files(self, Z, X):
        assert X + Z <= 1
        X = np.round(X, 2)
        Z = np.round(Z, 2)

        # placeholders
        version_number = 11

        content1 = [
            version_number,
            X,
            Z,
            self.num_logTs,
            self.logT_min,
            self.logT_max,
            self.del_logT,
            self.num_logWs,
            self.logW_min,
            self.logW_max,
            self.del_logW,
        ]

        if X >= 0.1:
            xtext = f"{100 * X:.0f}x.data"
        else:
            xtext = f"0{100 * X:.0f}x.data"

        if Z >= 0.1:
            ztext = f"{100 * Z:.0f}z"
        else:
            ztext = f"0{100 * Z:.0f}z"

        fname = "".join([self.fname_prefix, ztext, xtext])
        fdir = "data/mesa/eosPT"
        dst = os.path.join(fdir, fname)
        if self.check_if_file_exists:
            if os.path.isfile(dst):
                return

        results = np.zeros((self.num_logWs, self.num_logTs, self.num_vals),
                           dtype=np.float32)
        if self.debug:
            dbg_arr = np.zeros((self.num_logWs, self.num_logTs, self.num_vals))
        target_idcs = list(range(self.i_logRho, self.num_vals))
        target_idcs.remove(self.i_logP)

        for i in range(self.num_logWs):
            logW = self.logW_min + i * self.del_logW
            for j in range(self.num_logTs):
                logT = self.logT_min + j * self.del_logT
                logPgas = np.max([self.logP_min, logW + 4 * logT])
                logPgas = np.min([logPgas, self.logP_max])
                res = self.evaluate(logT, logPgas, X, Z)

                if (np.any(np.isnan(res)) or np.any(np.isinf(res))
                        or res[self.i_cp] <= 0 or res[self.i_cv] <= 0
                        or res[self.i_dS_dT] <= 0 or res[self.i_logU] > 30
                        or res[self.i_cp] > 1e30 or res[self.i_cv] > 1e30
                        or res[self.i_chiRho] <= 0 or res[self.i_chiT] <= 0
                        or res[self.i_gamma1] >= 9.99
                        or res[self.i_gamma3] >= 9.99):

                    res[self.i_logT] = logT
                    res[self.i_logP] = logPgas
                    if self.debug:
                        dbg_arr[i, j, :] = res
                        dbg_arr[i, j, self.i_logT] = logT
                        dbg_arr[i, j, self.i_logP] = logPgas
                        dbg_arr[i, j, self.i_logRho] = res[self.i_logRho]

                    if self.fix_vals:
                        res[self.i_logRho] = np.nan
                        res[self.i_logP + 1:] = np.nan
                    else:
                        res[np.where(np.isnan(res))] = -1
                        res[np.where(np.isinf(res))] = -1

                results[i, j, :] = res

        if self.fix_vals:
            x = results[:, :, self.i_logT].reshape(-1)
            y = results[:, :, self.i_logP].reshape(-1)
            z = results[:, :, target_idcs].reshape((-1, self.num_vals - 2))
            idcs = np.where(np.isfinite(z))[0]

            nnd_interp = NearestND((x[idcs], y[idcs]), z[idcs], rescale=True)
            zhat = nnd_interp((x, y), k=5, weights="distance").reshape(
                (self.num_logWs, self.num_logTs, -1))

            if self.smooth_vals:
                results[:, :, target_idcs] = zhat
            else:
                res2 = np.copy(results)
                res2[:, :, target_idcs] = zhat
                idcs = np.where(np.isfinite(results))
                res2[idcs] = results[idcs]
                results = np.copy(res2)
                del res2
            del x, y, z, zhat, idcs

        with open(dst, "w") as file:
            file.write(self.header1_line.write(self.header1))
            file.write("\n")
            file.write(self.content1_line.write(content1))
            file.write("\n")

            for i in range(self.num_logWs):
                logW = self.logW_min + i * self.del_logW
                file.write(self.header2_line.write(self.header2))
                file.write("\n")
                file.write(self.content2_line.write([logW]))
                file.write("\n")
                file.write(self.header3_line.write(self.header3))
                file.write("\n")

                for j in range(self.num_logTs):
                    res = results[i, j, :]
                    content3 = [
                        res[self.i_logT],
                        res[self.i_logRho],
                        res[self.i_logU],
                        res[self.i_logS],
                        res[self.i_chiRho],
                        res[self.i_chiT],
                        res[self.i_cp],
                        res[self.i_cv],
                        res[self.i_dE_dRho],
                        res[self.i_dS_dT],
                        res[self.i_dS_dRho],
                        res[self.i_mu],
                        res[self.i_lfe],
                        res[self.i_gamma1],
                        res[self.i_gamma3],
                        res[self.i_grad_ad],
                        res[self.i_eta],
                    ]
                    file.write(self.content3_line.write(content3))
                    file.write("\n")
                file.write("\n")
            file.write("\n")
            file.write("\n")
            file.close()

        if self.debug:
            debug_dir = os.path.join("data", "debug/eosPT")
            dname = fname.replace(self.fname_prefix, "debugPT_")
            dname = dname.replace(".data", ".npy")
            dst = os.path.join(debug_dir, dname)
            np.save(dst, dbg_arr)
        return results


if __name__ == "__main__":
    del_logT = 0.02
    del_logQ = 0.03
    del_logW = 0.2
    logT_min = 2.00
    logT_max = 6.00
    logQ_min = -4.00
    logQ_max = 5.50
    logW_min = -19  # logP_min - 4 * logT_max = -19
    logW_max = 7  # logP_max - 4 * logT_min = 7
    del_X = 0.1
    del_Z = 0.1


    # example
    Tdt = TableCreatorDT(which_heavy="water",
                         which_hhe="scvh",
                         build_interpolants=False,
                         fix_vals=True)
    Tdt.do_parallel = False
    Tdt.smooth_vals = True
    # for debugging
    del_logT = 0.2
    del_logQ = 0.2
    Tdt.debug = False

    Tdt.create_tables(
        del_logT,
        del_logQ,
        logT_min,
        logT_max,
        logQ_min,
        logQ_max,
        del_X,
        del_Z,
        "scvh_h2o_smoothed-eosDT_",
    )
