import os
from multiprocessing import cpu_count

import numpy as np
from fortranformat import FortranRecordWriter
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from tinyeos.definitions import (
    eos_num_vals,
    i_chiRho,
    i_chiT,
    i_cp,
    i_cv,
    i_dE_dRho,
    i_dS_dRho,
    i_dS_dT,
    i_eta,
    i_gamma1,
    i_gamma3,
    i_grad_ad,
    i_lfe,
    i_logP,
    i_logRho,
    i_logS,
    i_logT,
    i_logU,
    i_mu,
)
from tinyeos.support import NearestND
from tinyeos.tinydteos import TinyDirectDT, TinyDT


def build_mesa_tables(
    del_logT: float = 0.02,
    del_logQ: float = 0.05,
    logT_min: float = 2.00,
    logT_max: float = 6.00,
    logQ_min: float = -6.00,
    logQ_max: float = 6.00,
    del_X: float = 0.10,
    del_Z: float = 0.10,
    min_Z: float = 0.0,
    max_Z: float = 1.0,
    fname_prefix: str = "custom-eosDT",
    output_path: str = "tables",
    do_only_pure: bool = False,
    do_only_single: bool = False,
    which_hhe: str = "cms",
    which_heavy: str = "h2o",
    Z1: float = 0.5,
    Z2: float = 0.5,
    Z3: float = 0.0,
    use_helium_for_heavy_elements: bool = False,
    use_pt_eos: bool = True,
    include_hhe_interactions: bool = True,
    set_custom_eos_boundaries: bool = False,
    logT_min_eos: float = 2.0,
    logT_max_eos: float = 6.0,
    logRho_min_eos: float = -8.0,
    logRho_max_eos: float = 2.0,
    logP_min_eos: float = 1.0,
    logP_max_eos: float = 17.0,
    use_smoothed_xy_tables: bool = False,
    use_smoothed_z_tables: bool = False,
    build_interpolants: bool = False,
    do_simple_smoothing: bool = False,
    do_extra_smoothing: bool = False,
    num_extra_smoothing_rounds: int = 2,
    fix_bad_values: bool = True,
    do_parallel: bool = True,
    num_cores: int = cpu_count() - 2,
    debug: bool = False,
) -> ArrayLike:
    """Wrapper function to create density-temperature
    equation of state tables for MESA.

    Args:
        del_logT (float, optional): step-size for logT.
            Defaults to 0.02.
        del_logQ (float, optional): step-size for logQ.
            Defaults to 0.05.
        logT_min (float, optional): minimum logT.
            Defaults to 2.00.
        logT_max (float, optional): maximum logT.
            Defaults to 6.00.
        logQ_min (float, optional): minimum logQ.
            Defaults to -8.00.
        logQ_max (float, optional): maximum logQ.
            Defaults to 8.00.
        del_X (float, optional): step-size for hydrogen.
            Defaults to 0.10.
        del_Z (float, optional): step-size for heavy-elements.
            Defaults to 0.10.
        min_Z (float, optional): minimum heavy-element mass-fraction.
            Defaults to 0.0.
        max_Z (float, optional): maximum heavy-element mass-fraction.
            Defaults to 1.0.
        fname_prefix (str, optional): table prefix.
            Defaults to "cms_qeos-eosDT".
        do_only_pure (bool, optional): only create tables of
            pure substances. Defaults to False.
        do_only_single (bool, optional):  create a single table
            for debugging. Defaults to False.
        which_hhe (str, optional): hydrogen-helium
            equation of state to use. Defaults to "cms".
        which_heavy (str, optional): heavy-element
            equation of state to use. Defaults to "h2o".
        Z1 (float, optional): mass-fraction of the first heavy element.
            Defaults to 0.5
        Z2 (float, optional): mass-fraction of the second heavy element.
            Defaults to 0.5.
        Z3 (float, optional): mass-fraction of the third heavy element.
            Defaults to 0.0.
        use_helium_for_heavy_elements (bool, optional): use helium
            in place of heavy-elements. Defaults to False.
        use_pt_eos (bool, optional): use the pressure-temperature
            equation of state as the basis. Defaults to False.
        include_hhe_interactions (bool, optional): include
            hydrogen-helium interactions. Defaults to False.
        set_custom_eos_boundaries (bool, optional): set custom
            equation of state boundaries. Defaults to False.
        logT_min_eos (float, optional): minimum logT
            for the equation of state. Defaults to 2.0.
        logT_max_eos (float, optional): maximum logT
            for the equation of state. Defaults to 6.0.
        logRho_min_eos (float, optional): minimum logRho
            for the equation of state. Defaults to -8.0.
        logRho_max_eos (float, optional): maximum logRho
            for the equation of state. Defaults to 2.0.
        logP_min_eos (float, optional): minimum logP
            for the equation of state. Defaults to 1.0.
        logP_max_eos (float, optional): maximum logP
            for the equation of state. Defaults to 17.0.
        use_smoothed_xy_tables (bool, optional): use smoothed
            hydrogen and helium tables. Defaults to False.
        use_smoothed_z_tables (bool, optional): use smoothed
            heavy-element tables. Defaults to False.
        build_interpolants (bool, optional): build interpolants.
            Defaults to False.
        do_simple_smoothing (bool, optional): smooth tables
            with a nearest-neighbour method. Defaults to False.
        do_extra_smoothing (bool, optional): do extra smoothing
            by using the average of the five nearest points.
            Defaults to False.
        num_extra_smoothing_rounds (int, optional): how many times to
            do extra smoothing. Defaults to 2.
        fix_bad_values (bool, optional): fix bad equation of
            state results with a nearest-neighbour interpolation.
            Defaults to True.
        do_parallel (bool, optional): build tables in parallel.
            Defaults to True.
        num_cores (int, optional): how many cores to use in parallel.
            Defaults to cpu_count() - 2.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        ArrayLike: equation of state tables.
    """

    table_builder = TableBuilder(
        which_hhe=which_hhe,
        which_heavy=which_heavy,
        Z1=Z1,
        Z2=Z2,
        Z3=Z3,
        use_pt_eos=use_pt_eos,
        include_hhe_interactions=include_hhe_interactions,
        set_custom_eos_boundaries=set_custom_eos_boundaries,
        logT_min_eos=logT_min_eos,
        logT_max_eos=logT_max_eos,
        logRho_min_eos=logRho_min_eos,
        logRho_max_eos=logRho_max_eos,
        logP_min_eos=logP_min_eos,
        logP_max_eos=logP_max_eos,
        use_smoothed_xy_tables=use_smoothed_xy_tables,
        use_smoothed_z_tables=use_smoothed_z_tables,
        build_interpolants=build_interpolants,
        do_simple_smoothing=do_simple_smoothing,
        do_extra_smoothing=do_extra_smoothing,
        num_extra_smoothing_rounds=num_extra_smoothing_rounds,
        fix_bad_values=fix_bad_values,
        debug=debug,
    )

    num_tables, Xs, Zs = table_builder.set_params(
        del_logT=del_logT,
        del_logQ=del_logQ,
        logT_min=logT_min,
        logT_max=logT_max,
        logQ_min=logQ_min,
        logQ_max=logQ_max,
        del_X=del_X,
        del_Z=del_Z,
        min_Z=min_Z,
        max_Z=max_Z,
        fname_prefix=fname_prefix,
        output_path=output_path,
        do_only_pure=do_only_pure,
    )
    num_logQs = table_builder.num_logQs
    num_logTs = table_builder.num_logTs
    num_vals = table_builder.eos_num_vals

    def parallelWrapper(X: float, Z: float) -> NDArray:
        return table_builder.build_tables(
            X=X, Z=Z, use_helium_for_heavy_elements=use_helium_for_heavy_elements
        )

    if do_only_single:
        X = 0.5
        Z = 0.2
        print(
            f"Creating single table with prefix {fname_prefix}",
            f"with X = {X} and Z = {Z}.",
        )
        comp_info, results = parallelWrapper(0.5, 0.2)
        return comp_info, results

    if do_parallel:
        print(
            f"Creating grid of tables with prefix {fname_prefix}",
            f"using {num_cores:.0f} cores.",
        )
        comp_info = np.zeros((num_tables, 2))
        results = np.zeros((num_tables, num_logQs, num_logTs, num_vals))
        inputs = range(num_tables)
        parallel_results = Parallel(n_jobs=num_cores, verbose=10)(
            delayed(parallelWrapper)(X=Xs[i], Z=Zs[i]) for i in inputs
        )
        for i, pr in enumerate(parallel_results):
            comp_info[i] = pr[0]
            results[i] = pr[1]
    else:
        comp_info = np.zeros((num_tables, 2))
        results = np.zeros((num_tables, num_logQs, num_logTs, num_vals))
        for i in range(len(Xs)):
            print(
                f"Creating single table with prefix {fname_prefix}",
                f"with X = {Xs[i]:.2f} and Z = {Zs[i]:.2f}.",
            )
            comp_info[i], results[i] = parallelWrapper(Xs[i], Zs[i])
    return comp_info, results


class TableBuilder:
    """Creates density-temperature tables for use with the stellar evolution
    code MESA.
    """

    def __init__(
        self,
        which_hhe: str = "cms",
        which_heavy: str = "h2o",
        Z1: float = 0.5,
        Z2: float = 0.5,
        Z3: float = 0.0,
        use_pt_eos: bool = True,
        include_hhe_interactions: bool = True,
        set_custom_eos_boundaries: bool = False,
        logT_min_eos: float = 2.0,
        logT_max_eos: float = 6.0,
        logRho_min_eos: float = -8.0,
        logRho_max_eos: float = 2.0,
        logP_min_eos: float = 1.0,
        logP_max_eos: float = 17.0,
        use_smoothed_xy_tables: bool = False,
        use_smoothed_z_tables: bool = False,
        build_interpolants: bool = False,
        do_simple_smoothing: bool = False,
        do_extra_smoothing: bool = False,
        num_extra_smoothing_rounds: int = 2,
        fix_bad_values: bool = True,
        debug: bool = False,
    ) -> None:
        """__init__ method. Sets up the parameters to use
        for creating the equation of state tables.

        Args:
            which_hhe (str, optional): hydrogen-helium
                equation of state to use. Defaults to "cms".
            which_heavy (str, optional): heavy-element
                equation of state to use. Defaults to "h2o".
            Z1 (float, optional): mass-fraction of the first heavy element.
                Defaults to 0.5
            Z2 (float, optional): mass-fraction of the second heavy element.
                Defaults to 0.5.
            Z3 (float, optional): mass-fraction of the third heavy element.
                Defaults to 0.0.
            use_pt_eos (bool, optional): use the pressure-temperature
                equation of state as the basis. Defaults to False.
            include_hhe_interactions (bool, optional): include
                hydrogen-helium interactions. Defaults to False.
            set_custom_eos_boundaries (bool, optional): set custom
                equation of state boundaries. Defaults to False.
            logT_min_eos (float, optional): minimum logT
                for the equation of state. Defaults to 2.0.
            logT_max_eos (float, optional): maximum logT
                for the equation of state. Defaults to 6.0.
            logRho_min_eos (float, optional): minimum logRho
                for the equation of state. Defaults to -8.0.
            logRho_max_eos (float, optional): maximum logRho
                for the equation of state. Defaults to 2.0.
            logP_min_eos (float, optional): minimum logP
                for the equation of state. Defaults to 1.0.
            logP_max_eos (float, optional): maximum logP
                for the equation of state. Defaults to 17.0.
            use_smoothed_xy_tables (bool, optional): use smoothed
                hydrogen and helium tables. Defaults to False.
            use_smoothed_z_tables (bool, optional): use smoothed
                heavy-element tables. Defaults to False.
            build_interpolants (bool, optional): build interpolants.
                Defaults to False.
            do_simple_smoothing (bool, optional): smooth tables
                with a nearest-neighbour method. Defaults to False.
            do_extra_smoothing (bool, optional): do extra smoothing
                by using the average of the five nearest points.
                Defaults to False.
            num_extra_smoothing_rounds (int, optional): how many times to
                do extra smoothing. Defaults to 2.
            fix_bad_values (bool, optional): fix bad equation of
                state results with a nearest-neighbour interpolation.
                Defaults to True.
            debug (bool, optional): enable debugging mode for
                additional output. Defaults to False.
        """
        eos = TinyDT if use_pt_eos else TinyDirectDT
        self.eos = eos(
            which_hhe=which_hhe,
            which_heavy=which_heavy,
            Z1=Z1,
            Z2=Z2,
            Z3=Z3,
            include_hhe_interactions=include_hhe_interactions,
            use_smoothed_xy_tables=use_smoothed_xy_tables,
            use_smoothed_z_tables=use_smoothed_z_tables,
            build_interpolants=build_interpolants,
        )

        if set_custom_eos_boundaries:
            if use_pt_eos:
                self.eos.tpt.logT_min = logT_min_eos
                self.eos.tpt.logT_max = logT_max_eos
                self.eos.tpt.logP_min = logP_min_eos
                self.eos.tpt.logP_max = logP_max_eos
                self.eos.tpt.logRho_min = logRho_min_eos
                self.eos.tpt.logRho_max = logRho_max_eos

            else:
                self.eos.logT_min = logT_min_eos
                self.eos.logT_max = logT_max_eos
                self.eos.logP_min = logP_min_eos
                self.eos.logP_max = logP_max_eos
                self.eos.logRho_min = logRho_min_eos
                self.eos.logRho_max = logRho_max_eos

        self.fix_bad_values = fix_bad_values
        self.do_simple_smoothing = do_simple_smoothing
        self.do_extra_smoothing = do_extra_smoothing
        self.num_extra_smoothing_rounds = num_extra_smoothing_rounds
        self.debug = debug

        # for debugging purposes
        self.check_if_file_exists = False

        self.header1_line = FortranRecordWriter("(99(a14))")
        self.header2_line = FortranRecordWriter("(/,7x,a)")
        self.header3_line = FortranRecordWriter(
            "(a4,1x,3(a9,1x),7(a12,1x),1(a7,1x),1(a11),3(a9,1x),1(a9,1x))"
        )

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

        self.content1_line = FortranRecordWriter("(i14,2f14.4,2(i10,4x,3(f14.4)))")
        self.content2_line = FortranRecordWriter("(2x,f14.6/)")
        self.content3_line = FortranRecordWriter(
            "(f4.2,3(f10.5),7(1pe13.5),1(0pf9.5),4(0pf10.5),1(0pf11.5))"
        )

    def set_params(
        self,
        del_logT: float,
        del_logQ: float,
        logT_min: float,
        logT_max: float,
        logQ_min: float,
        logQ_max: float,
        del_X: float,
        del_Z: float,
        min_Z: float,
        max_Z: float,
        fname_prefix: str,
        output_path: str,
        do_only_pure: bool = False,
    ) -> tuple[int, NDArray, NDArray]:
        """Sets the table parameters.

        Args:
            del_logT (float): step size for logT.
            del_logQ (float): step size for logQ.
            logT_min (float): minimum logT.
            logT_max (float): maximum logT.
            logQ_min (float): minimum logQ.
            logQ_max (float): maximum logQ.
            del_X (float): step size for hydrogen.
            del_Z (float): step size for heavy-elements.
            fname_prefix (str): table prefix.
            output_path (str): path to where the tables
                are saved.
            do_only_pure (bool, optional): only create
                tables of pure substances. Defaults to False.

        Returns:
            tuple[int, NDArray, NDarray]: number of tables and
                mass fractions.
        """
        self.eos_num_vals = eos_num_vals
        self.logT_min = logT_min
        self.logT_max = logT_max
        self.logQ_min = logQ_min
        self.logQ_max = logQ_max
        self.del_logT = del_logT
        self.del_logQ = del_logQ
        self.num_logTs = np.int32(1 + np.ceil((logT_max - logT_min) / del_logT))
        self.num_logQs = np.int32(1 + np.rint((logQ_max - logQ_min) / del_logQ))
        self.fname_prefix = fname_prefix
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if do_only_pure:
            self.Xs = np.array([1, 0, 0])
            self.Zs = np.array([0, 0, 1])
        else:
            if min_Z == max_Z or del_Z == 0:
                Zs = np.array([min_Z])
            else:
                Zs = np.arange(min_Z, max_Z + del_Z, del_Z)
            Xs = [np.arange(0, (1 - Z) + del_X, del_X) for Z in Zs]
            Zs = [Zs[i] * np.ones(len(Xs[i])) for i in range(len(Zs))]
            self.Xs = np.concatenate(Xs)
            self.Zs = np.concatenate(Zs)
        self.num_tables = len(self.Xs)
        return (self.num_tables, self.Xs, self.Zs)

    def build_tables(
        self,
        X: float,
        Z: float,
        use_helium_for_heavy_elements: bool = False,
    ) -> NDArray:
        """Wrapper function for __make_eos_files.
        Calculates the equation of state table for a mixture of
        hydrogen, helium and a heavy-element.

        Args:
            X (float): hydrogen mass fraction.
            Z (float): heavy-element mass fraction.
            use_helium_for_heavy_elements (bool, optional): use helium
                in place of heavy-elements. Defaults to False.

        Returns:
            ArrayLike: equation of state tables.
        """
        return self.__make_eos_files(X, Z, use_helium_for_heavy_elements)

    def __smooth_table(self, table: NDArray) -> NDArray:
        """Smoothes the equation of state tables by taking
        the average of the original point and the four nearest neighbours
        on the two-dimensional grid.

        Args:
            table (NDArray): original unsmoothed table.

        Returns:
            NDArray: smoothed table.
        """
        out_table = np.copy(table)
        for i in range(self.num_logQs):
            if i < 1:
                continue
            elif i > self.num_logQs - 2:
                break
            for j in range(self.num_logTs):
                if j < 1:
                    continue
                elif j > self.num_logTs - 2:
                    break
                out_table[i, j, i_logP:] = (
                    table[i, j, i_logP:]
                    + table[i + 1, j, i_logP:]
                    + table[i - 1, j, i_logP:]
                    + table[i, j - 1, i_logP:]
                    + table[i, j + 1, i_logP:]
                ) / 5
        return out_table

    def __make_eos_files(
        self, X: float, Z: float, use_helium_for_heavy_elements: bool = False
    ) -> NDArray:
        """Calculates the equation of state table for a mixture of
        hydrogen, helium and a heavy-element.

        Args:
            X (float): hydrogen mass-fraction.
            Z (float): heavy-element mass-fraction.
            use_helium_for_heavy_elements (bool, optional): use helium
                in place of heavy-elements. Defaults to False.

        Returns:
            NDArray: equation of state table.
        """
        assert X + Z <= 1
        X = np.round(X, 2)
        Z = np.round(Z, 2)

        if use_helium_for_heavy_elements:
            X_for_eos = X - Z
            Z_for_eos = 0
        else:
            X_for_eos = X
            Z_for_eos = Z

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
        xtext = f"{100 * X:.0f}x.data" if X >= 0.1 else f"0{100 * X:.0f}x.data"
        ztext = f"{100 * Z:.0f}z" if Z >= 0.1 else f"0{100 * Z:.0f}z"

        fname = "".join([self.fname_prefix, ztext, xtext])
        dst = os.path.join(self.output_path, fname)

        results = np.zeros(
            (self.num_logQs, self.num_logTs, eos_num_vals), dtype=np.float32
        )
        if self.debug:
            dbg_arr = np.zeros((self.num_logQs, self.num_logTs, eos_num_vals))

        for i in range(self.num_logQs):
            logQ = self.logQ_min + i * self.del_logQ
            for j in range(self.num_logTs):
                logT = self.logT_min + j * self.del_logT
                # make sure to stay within equation of state
                # logRho boundaries
                logRho = np.max([self.eos.logRho_min, logQ + 2 * logT - 12])
                logRho = np.min([logRho, self.eos.logRho_max])
                res = self.eos.evaluate(
                    logT=logT, logRho=logRho, X=X_for_eos, Z=Z_for_eos
                )
                if (
                    np.any(np.isnan(res))
                    or np.any(np.isinf(res))
                    or res[i_logU] > 20
                    or res[i_logS] > 15
                    or res[i_chiRho] <= 0
                    or res[i_chiT] <= 0
                    or res[i_grad_ad] <= 0
                    or res[i_cp] <= 0
                    or res[i_cv] <= 0
                    or res[i_cp] > 1e20
                    or res[i_cv] > 1e20
                    or res[i_gamma1] >= 9.99
                    or res[i_gamma1] <= 0
                    or res[i_gamma3] >= 9.99
                    or res[i_gamma3] <= 0
                    or res[i_dS_dT] <= 0
                ):
                    res[i_logT] = logT
                    res[i_logRho] = logRho
                    if self.debug:
                        dbg_arr[i, j, :] = res
                        dbg_arr[i, j, i_logT] = logT
                        dbg_arr[i, j, i_logP] = res[i_logP]
                        dbg_arr[i, j, i_logRho] = logRho
                    if self.fix_bad_values:
                        res[i_logRho + 1 :] = np.nan
                results[i, j, :] = res

        if self.fix_bad_values:
            x = results[:, :, i_logT].reshape(-1)
            y = results[:, :, i_logRho].reshape(-1)
            z = results[:, :, i_logRho + 1 :].reshape((-1, eos_num_vals - 2))
            idcs = np.where(np.isfinite(z))[0]

            if self.debug:
                print(z.size, len(np.where(np.isnan(z))[0]))

            nnd_interp = NearestND((x[idcs], y[idcs]), z[idcs], rescale=True)

            zhat = nnd_interp((x, y), k=5, weights="distance").reshape(
                (self.num_logQs, self.num_logTs, -1)
            )

            if self.do_simple_smoothing:
                results[:, :, i_logRho + 1 :] = zhat
            else:
                res2 = np.copy(results)
                res2[:, :, i_logRho + 1 :] = zhat
                idcs = np.where(np.isfinite(results))
                res2[idcs] = results[idcs]
                results = np.copy(res2)
                del res2

            del x, y, z, zhat, idcs

        if self.do_extra_smoothing:
            for _ in range(self.num_extra_smoothing_rounds):
                results = self.__smooth_table(results)

        if self.check_if_file_exists and os.path.isfile(dst):
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
                        res[i_logT],
                        res[i_logP],
                        res[i_logU],
                        res[i_logS],
                        res[i_chiRho],
                        res[i_chiT],
                        res[i_cp],
                        res[i_cv],
                        res[i_dE_dRho],
                        res[i_dS_dT],
                        res[i_dS_dRho],
                        res[i_mu],
                        res[i_lfe],
                        res[i_gamma1],
                        res[i_gamma3],
                        res[i_grad_ad],
                        res[i_eta],
                    ]
                    file.write(self.content3_line.write(content3))
                    file.write("\n")
                file.write("\n")
            file.write("\n")
            file.write("\n")
            file.close()

        if self.debug:
            dname = fname.replace(".data", ".npy")
            dst = os.path.join(self.output_path, dname)
            np.save(dst, dbg_arr)

        return ([X, Z], results)
