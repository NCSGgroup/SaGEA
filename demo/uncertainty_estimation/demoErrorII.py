import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.load_file.LoadGIA import LoadGIA
from pysrc.auxiliary.load_file.LoadL2LowDeg import LoadLowDegree
from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH, load_SHC
from pysrc.auxiliary.preference.EnumClasses import L2InstituteType, L2LowDegreeType, L2LowDegreeFileID, \
    FieldPhysicalQuantity
from pysrc.auxiliary.scripts.PlotGrids import plot_grids
from pysrc.data_class.DataClass import SHC, GRID
from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC
from pysrc.post_processing.extract_basin_signal.ExtractSpatialSignal import ExtractSpatial
from pysrc.post_processing.filter.Gaussian import Gaussian
from pysrc.post_processing.GIA_correction.GIACorrectionSpectral import GIACorrectionSpectral
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.post_processing.replace_low_deg.ReplaceLowDegree import ReplaceLowDegree
from pysrc.time_series_analysis.OrdinaryLeastSquare import OLS
from pysrc.time_series_analysis.WeightedLeastSquare import WLS
from pysrc.uncertainty_estimating.covaiance_propagation.BasinSumPropagation import BasinSumPropagation
from pysrc.uncertainty_estimating.covaiance_propagation.ConvertSHCPropagation import ConvertSHCPropagation
from pysrc.uncertainty_estimating.covaiance_propagation.GaussianPropagation import GaussianPropagation
from pysrc.uncertainty_estimating.covaiance_propagation.HarmonicPropagation import HarmonicPropagation


def demo():
    """GMOM estimate with error information: get varEWH"""

    '''load GRACE L2 SH products'''
    begin_date = datetime.date(2005, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    institute = L2InstituteType.CSR
    lmax = 60

    print('loading files...')
    load = LoadL2SH()

    load.configuration.set_begin_date(begin_date)
    load.configuration.set_end_date(end_date)
    load.configuration.set_institute(institute)
    load.configuration.set_lmax(lmax)

    shc, dates = load.get_shc(with_dates=True)
    ave_dates_GRACE = TimeTool.get_average_dates(*dates)

    sigma_shc = load.get_shc(get_sigma=True)

    cov_matrices_list = [np.diag((sigma_shc.value[i])) ** 2 for i in range(len(sigma_shc.value))]

    '''load and replace low degrees'''
    degree1_or_not = True
    degree1_file_id = L2LowDegreeFileID.TN13

    c20_or_not = True
    c20_file_id = L2LowDegreeFileID.TN14

    c30_or_not = True
    c30_file_id = L2LowDegreeFileID.TN14

    low_degs = {}

    if degree1_or_not:
        load_deg1 = LoadLowDegree()
        load_deg1.configuration.set_file_id(degree1_file_id).set_institute(institute)
        low_degs.update(load_deg1.get_degree1())

    if c20_or_not:
        load_c20 = LoadLowDegree()
        load_c20.configuration.set_file_id(c20_file_id)
        low_degs.update(load_c20.get_c20())

    if c30_or_not:
        load_c30 = LoadLowDegree()
        load_c30.configuration.set_file_id(c30_file_id)
        low_degs.update(load_c30.get_c30())

    rep = ReplaceLowDegree()
    rep.configuration.set_replace_deg1(degree1_or_not).set_replace_c20(c20_or_not).set_replace_c30(c30_or_not)
    rep.set_low_degrees(low_degs)

    shc = rep.apply_to(shc, dates[0], dates[1])

    '''deduct background'''
    clm_bg, slm_bg = load_SHC(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=60,
        lmcs_in_queue=(2, 3, 4, 5)
    )
    shc.de_background(SHC(clm_bg, slm_bg))

    '''loop for each cov_cs'''
    varEWH_list = []
    for i in range(len(cov_matrices_list)):
        print(f'\n{i + 1} / {len(cov_matrices_list)}')

        VGC_id = '300_500'

        filename = (
            f'{str(dates[0][i].year)}'
            f'{str(dates[0][i].month).rjust(2, "0")}'
            f'{str(dates[0][i].day).rjust(2, "0")}'
            f'-'
            f'{str(dates[1][i].year)}'
            f'{str(dates[1][i].month).rjust(2, "0")}'
            f'{str(dates[1][i].day).rjust(2, "0")}'
        )
        # save_path = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/VGC{VGC_id}/{filename}.hdf5')
        # save_path = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/GS300/{filename}.hdf5')
        # save_path = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/NonFilter/{filename}.hdf5')
        save_path = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/GS500/{filename}.hdf5')
        if save_path.exists():
            continue

        cov_cs = cov_matrices_list[i]

        '''convert to EWH'''
        print('convert to EWH')
        LN = LoveNumber()
        LN.configuration.set_lmax(lmax)
        ln = LN.get_Love_number()

        convert_prop = ConvertSHCPropagation()
        convert_prop.configuration.set_input_type(FieldPhysicalQuantity.Dimensionless)
        convert_prop.configuration.set_output_type(FieldPhysicalQuantity.EWH)
        convert_prop.configuration.set_Love_number(ln)

        cov_cs = convert_prop.apply_to(cov_cs)

        '''Gaussian filter'''
        print('Gaussian filter')
        radius = 500

        gs_prop = GaussianPropagation()
        gs_prop.configuration.set_lmax(lmax).set_filtering_radius(radius)

        cov_cs = gs_prop.apply_to(cov_cs)

        '''VGC filter'''
        # print('VGC filter')
        # with h5py.File(FileTool.get_project_dir(f'data/vgc_data/VGC{VGC_id}.hdf5'), 'r') as f:
        #     vgc_mat = np.zeros(((lmax + 1) ** 2, (lmax + 1) ** 2))
        #     vgc_mat[1:, 1:] = np.array(f['matrix'])

        # print('VGC filter')
        # vgc_npz = np.load(
        #     FileTool.get_project_dir(f'data/vgc_data_experimental/vgc_spectral_filtering_matrix_300_500_1.npz'))
        # vgc_mat = vgc_npz['grids']
        # cov_cs = vgc_mat @ cov_cs @ vgc_mat.T

        '''extract ocean signal'''
        ocean_mask_grid = GRID(
            np.load('../../temp/ocean_300km-buffer(360,720))_Uebbing.npy'),
            *MathTool.get_global_lat_lon_range(0.5)
        )
        ocean_mask_shc = ocean_mask_grid.to_SHC(lmax)

        # ocean_mask_cs = load_SH_simple(
        #     FileTool.get_project_dir('data/auxiliary/ocean360_grndline.sh'),
        #     key='',
        #     lmcs_in_queue=(1, 2, 3, 4),
        #     lmax=lmax
        # )
        # ocean_mask_shc = SHC(*ocean_mask_cs)
        # gs = Gaussian()
        # gs.configuration.set_lmax(lmax)
        # gs.configuration.set_filtering_radius(radius)
        # ocean_mask_shc = gs.apply_to(ocean_mask_shc)

        extract_prop = BasinSumPropagation()
        extract_prop.set_cov_mat(cov_cs)
        extract_prop.set_basin(ocean_mask_shc)

        var_ewha_ocean = extract_prop.get_average()
        print(np.sqrt(var_ewha_ocean) * 100)  # cm
        varEWH_list.append(var_ewha_ocean)

        # continue

        '''harmonic to grids'''
        print('harmonic to grids')
        grid_space = 3
        lat, lon = MathTool.get_global_lat_lon_range(grid_space)
        har_prop = HarmonicPropagation(lat, lon, lmax, option=1)

        varEWH = har_prop.synthesis_var(cov_cs)

        print(np.sqrt(MathTool.global_integral(varEWH, for_square=True)) / MathTool.get_acreage(
            np.load('../../temp/ocean_300km-buffer(360,720))_Uebbing.npy')))
        varEWH_list.append(varEWH)

        # break

        '''save hdf5 file'''
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('lat', data=lat)
            f.create_dataset('lon', data=lon)
            f.create_dataset('data', data=varEWH)

        print(f'file {save_path.name} has been saved!')

    # year_fraction = TimeTool.convert_date_format(
    #     dates[0],
    #     input_type=TimeTool.DateFormat.ClassDate,
    #     output_type=TimeTool.DateFormat.YearFraction
    # )
    #
    # plt.plot(year_fraction, np.sqrt(np.array(varEWH_list)) * 100)
    #
    # np.savez(
    #     FileTool.get_project_dir(
    #         'results/GMOM_estimation_with_error/std_ocean_VGC300_500.npz'
    #     ),
    #     time=np.array(year_fraction),
    #     data=np.array(np.sqrt(np.array(varEWH_list)))
    # )

    # plt.matshow(
    #     np.sqrt(np.array(varEWH_list)[0])
    #     , vmin=0, vmax=0.015
    # )
    # plt.colorbar()
    # plt.style.use('science')
    #
    # plt.show()

    pass


def demo2():
    """GMOM estimate with error information: get sigma-weighted EWHA"""

    '''load GRACE L2 SH products'''
    begin_date = datetime.date(2005, 1, 1)
    end_date = datetime.date(2015, 12, 31)
    institute = L2InstituteType.CSR
    lmax = 60

    print('loading files...')
    load = LoadL2SH()

    load.configuration.set_begin_date(begin_date)
    load.configuration.set_end_date(end_date)
    load.configuration.set_institute(institute)
    load.configuration.set_lmax(lmax)

    shc, dates = load.get_shc(with_dates=True)
    ave_dates_GRACE = TimeTool.get_average_dates(*dates)

    '''load and replace low degrees'''
    degree1_or_not = True
    degree1_file_id = L2LowDegreeFileID.TN13

    c20_or_not = True
    c20_file_id = L2LowDegreeFileID.TN14

    c30_or_not = True
    c30_file_id = L2LowDegreeFileID.TN14

    low_degs = {}

    if degree1_or_not:
        load_deg1 = LoadLowDegree()
        load_deg1.configuration.set_file_id(degree1_file_id).set_institute(institute)
        low_degs.update(load_deg1.get_degree1())

    if c20_or_not:
        load_c20 = LoadLowDegree()
        load_c20.configuration.set_file_id(c20_file_id)
        low_degs.update(load_c20.get_c20())

    if c30_or_not:
        load_c30 = LoadLowDegree()
        load_c30.configuration.set_file_id(c30_file_id)
        low_degs.update(load_c30.get_c30())

    rep = ReplaceLowDegree()
    rep.configuration.set_replace_deg1(degree1_or_not).set_replace_c20(c20_or_not).set_replace_c30(c30_or_not)
    rep.set_low_degrees(low_degs)

    shc = rep.apply_to(shc, dates[0], dates[1])

    '''deduct background'''
    # clm_bg, slm_bg = load_SH_simple(
    #     FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
    #     key='gfc',
    #     lmax=60,
    #     lmcs_in_queue=(2, 3, 4, 5)
    # )
    # shc.de_background(SHC(clm_bg, slm_bg))
    shc.de_background()

    '''GIA correction'''
    load_gia = LoadGIA()
    load_gia.configuration.set_filepath(FileTool.get_project_dir() / 'data/GIA/GIA.ICE-6G_D.txt')
    load_gia.configuration.set_lmax(lmax)

    shc_gia_trend = load_gia.get_shc()

    gia = GIACorrectionSpectral()
    gia.configuration.set_times(ave_dates_GRACE)
    gia.configuration.set_gia_trend(shc_gia_trend)

    shc = gia.apply_to(shc)

    '''convert to EWH'''
    print('convert to EWH')
    LN = LoveNumber()
    LN.configuration.set_lmax(lmax)
    ln = LN.get_Love_number()

    convert = ConvertSHC()
    convert.configuration.set_input_type(FieldPhysicalQuantity.Dimensionless)
    convert.configuration.set_output_type(FieldPhysicalQuantity.EWH)
    convert.configuration.set_Love_number(ln)

    shc = convert.apply_to(shc)

    '''Gaussian filter'''
    print('Gaussian filter')
    radius = 300

    gs = Gaussian()
    gs.configuration.set_lmax(lmax).set_filtering_radius(radius)

    shc = gs.apply_to(shc)

    '''VGC filter'''
    # print('VGC filter')
    # vgc_npz = np.load(
    #     FileTool.get_project_dir(f'data/vgc_data_experimental/vgc_spectral_filtering_matrix_300_500_1.npz'))
    # vgc_mat = vgc_npz['grids']
    #
    # shc.cs = shc.cs @ vgc_mat

    '''harmonic to grids'''
    print('harmonic to grids')
    grid_space = 3
    lat, lon = MathTool.get_global_lat_lon_range(grid_space)
    har = Harmonic(lat, lon, lmax, option=1)

    grid = har.synthesis(shc)

    '''load sigma file'''
    sigma_EWH_list_vgc = []
    sigma_EWH_list_gs = []
    sigma_EWH_list_non_filter = []
    filenames = [
        f'{str(dates[0][i].year)}'
        f'{str(dates[0][i].month).rjust(2, "0")}'
        f'{str(dates[0][i].day).rjust(2, "0")}'
        f'-'
        f'{str(dates[1][i].year)}'
        f'{str(dates[1][i].month).rjust(2, "0")}'
        f'{str(dates[1][i].day).rjust(2, "0")}'
        for i in range(len(dates[0]))
    ]

    for i in range(len(filenames)):
        filepath_vgc = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/VGC300_500/{filenames[i]}.hdf5')
        # filepath_gs = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/GS500/{filenames[i]}.hdf5')
        filepath_gs = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/GS300/{filenames[i]}.hdf5')
        filepath_non_filter = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/NonFilter/{filenames[i]}.hdf5')
        with h5py.File(filepath_gs, 'r') as f:
            sigma_EWH = np.sqrt(np.array(f['data']))
            sigma_EWH_list_gs.append(sigma_EWH)

        with h5py.File(filepath_vgc, 'r') as f:
            sigma_EWH = np.sqrt(np.array(f['data']))
            sigma_EWH_list_vgc.append(sigma_EWH)

        with h5py.File(filepath_non_filter, 'r') as f:
            sigma_EWH_non_filter = np.sqrt(np.array(f['data']))
            sigma_EWH_list_non_filter.append(sigma_EWH_non_filter)

    sigma_non_filter_array = np.array(sigma_EWH_list_non_filter)

    sigma_weight_gs = 1 / np.array(sigma_EWH_list_gs)
    sigma_weight_vgc = 1 / np.array(sigma_EWH_list_vgc)

    smoothing_scale = np.array(sigma_EWH_list_vgc) / np.array(sigma_EWH_list_gs)

    # smoothing_scale /= (np.sum(smoothing_scale[:, 29:31, :], axis=1)[:, None, :] / 2)
    '''load'''

    '''get GMOM'''
    print('get GMOM')
    extract = ExtractSpatial()
    extract.configuration.set_lat_lon_range(*MathTool.get_global_lat_lon_range(grid_space))

    # ocean_mask = np.load('../../temp/ocean_300km-buffer(360,720))_Uebbing.npy')
    ocean_mask = np.load('../../temp/ocean_500km-buffer(360,720).npy')
    ocean_mask = MathTool.shrink(ocean_mask, 60, 120) / 36

    ocean_mask[np.where(ocean_mask >= 0.5)] = 1
    ocean_mask[np.where(ocean_mask < 0.5)] = 0

    # ocean_mask[:7] = 0  # +- 67 deg
    # ocean_mask[-7:] = 0

    extract.set_basin(ocean_mask)
    extract.set_signal(grid)
    ewha_classical = extract.get_average()

    '''error-scaled'''
    pass
    # scale = sigma_weight / (np.sum(sigma_weight[:, 29:31, :], axis=1)[:, None, :] / 2)
    grid.value *= smoothing_scale

    extract.set_signal(grid)
    ewha_se_weighted = extract.get_average()

    # extract.set_weight(var_weight)
    # extract.set_weight(sigma_weight)
    # ewha_se_weighted = extract.get_average()

    '''curve fit'''

    year_fraction = TimeTool.convert_date_format(
        ave_dates_GRACE,
        input_type=TimeTool.DateFormat.ClassDate,
        output_type=TimeTool.DateFormat.YearFraction
    )
    year_fraction = np.array(year_fraction)

    ols = OLS.OLSFor1d()

    ols.setSignals(year_fraction, ewha_classical)
    trend, sigma_trend = ols.get_trend(with_sigma=True)
    an_a, sigma_an_a = ols.get_annual_amplitude(with_sigma=True)
    an_p, sigma_an_p = ols.get_annual_phase(with_sigma=True)
    semian_a, sigma_semian_a = ols.get_semiannual_amplitude(with_sigma=True)
    semian_p, sigma_semian_p = ols.get_semiannual_phase(with_sigma=True)
    print('unscaled')
    print(f'{round(trend * 1000, 2)}±{round(sigma_trend * 1000, 2)}', end='\t')
    print(f'{round(an_a * 1000, 2)}±{round(sigma_an_a * 1000, 2)}', end='\t')
    print(f'{round(an_p, 2)}±{round(sigma_an_p, 2)}', end='\t')
    print(f'{round(semian_a * 1000, 2)}±{round(sigma_semian_a * 1000, 2)}', end='\t')
    print(f'{round(semian_p, 2)}±{round(sigma_semian_p, 2)}')
    print()

    ols.setSignals(year_fraction, ewha_se_weighted)
    trend, sigma_trend = ols.get_trend(with_sigma=True)
    an_a, sigma_an_a = ols.get_annual_amplitude(with_sigma=True)
    an_p, sigma_an_p = ols.get_annual_phase(with_sigma=True)
    semian_a, sigma_semian_a = ols.get_semiannual_amplitude(with_sigma=True)
    semian_p, sigma_semian_p = ols.get_semiannual_phase(with_sigma=True)
    print('error-scaled')
    print(f'{round(trend * 1000, 2)}±{round(sigma_trend * 1000, 2)}', end='\t')
    print(f'{round(an_a * 1000, 2)}±{round(sigma_an_a * 1000, 2)}', end='\t')
    print(f'{round(an_p, 2)}±{round(sigma_an_p, 2)}', end='\t')
    print(f'{round(semian_a * 1000, 2)}±{round(sigma_semian_a * 1000, 2)}', end='\t')
    print(f'{round(semian_p, 2)}±{round(sigma_semian_p, 2)}')

    # fit_signal_wls = ols.get_fitting_signal()
    #
    # wls = WLS.WLSFor1d()
    #
    # # load_std = np.load(FileTool.get_project_dir('results/GMOM_estimation_with_error/std_ocean_GS300.npz'))
    # load_std = np.load(FileTool.get_project_dir('results/GMOM_estimation_with_error/std_ocean_VGC300_500.npz'))
    # begin_date_year_fraction = TimeTool.convert_date_format(
    #     begin_date,
    #     input_type=TimeTool.DateFormat.ClassDate,
    #     output_type=TimeTool.DateFormat.YearFraction
    # )
    # end_date_year_fraction = TimeTool.convert_date_format(
    #     end_date,
    #     input_type=TimeTool.DateFormat.ClassDate,
    #     output_type=TimeTool.DateFormat.YearFraction
    # )
    #
    # std_index = np.where((year_fraction >= begin_date_year_fraction) & (year_fraction <= end_date_year_fraction))
    # std = load_std['data'][std_index]
    #
    # weight = 1 / std ** 2
    # wls.setSignals(year_fraction, ewha_classical, weight=weight)
    # trend, sigma_trend = wls.get_trend(with_sigma=True)
    # an_a, sigma_an_a = wls.get_annual_amplitude(with_sigma=True)
    # an_p, sigma_an_p = wls.get_annual_phase(with_sigma=True)
    # semian_a, sigma_semian_a = wls.get_semiannual_amplitude(with_sigma=True)
    # semian_p, sigma_semian_p = wls.get_semiannual_phase(with_sigma=True)
    #
    # fit_signal_wls = wls.get_fitting_signal()
    # print(f'{round(trend * 1000, 2)}±{round(sigma_trend * 1000, 2)}', end='\t')
    # print(f'{round(an_a * 1000, 2)}±{round(sigma_an_a * 1000, 2)}', end='\t')
    # print(f'{round(an_p, 2)}±{round(sigma_an_p, 2)}', end='\t')
    # print(f'{round(semian_a * 1000, 2)}±{round(sigma_semian_a * 1000, 2)}', end='\t')
    # print(f'{round(semian_p, 2)}±{round(sigma_semian_p, 2)}')
    #
    # wls.setSignals(year_fraction, ewha_se_weighted, weight=weight)
    # trend, sigma_trend = wls.get_trend(with_sigma=True)
    # an_a, sigma_an_a = wls.get_annual_amplitude(with_sigma=True)
    # an_p, sigma_an_p = wls.get_annual_phase(with_sigma=True)
    # semian_a, sigma_semian_a = wls.get_semiannual_amplitude(with_sigma=True)
    # semian_p, sigma_semian_p = wls.get_semiannual_phase(with_sigma=True)
    # print(f'{round(trend * 1000, 2)}±{round(sigma_trend * 1000, 2)}', end='\t')
    # print(f'{round(an_a * 1000, 2)}±{round(sigma_an_a * 1000, 2)}', end='\t')
    # print(f'{round(an_p, 2)}±{round(sigma_an_p, 2)}', end='\t')
    # print(f'{round(semian_a * 1000, 2)}±{round(sigma_semian_a * 1000, 2)}', end='\t')
    # print(f'{round(semian_p, 2)}±{round(sigma_semian_p, 2)}')

    # '''plot EWHA'''
    # index = -10
    # ocean_mask_to_plot = ocean_mask.copy()
    # ocean_mask_to_plot[np.where(ocean_mask_to_plot) == 0] = np.nan
    # plot_grids(
    #     np.array([
    #         # var_weight[index] * ocean_mask_to_plot * 1000,
    #         sigma_EWH_list[index] * ocean_mask_to_plot * 1000,
    #         # grid.data[index] * ocean_mask_to_plot * 1000
    #     ]),
    #     lat, lon,
    #     0, 11
    # )

    plt.plot(year_fraction, ewha_classical * 1000, label='unscaled', color='green')
    plt.plot(year_fraction, ewha_se_weighted * 1000, label='error-scaled', color='red')

    # plt.plot(year_fraction, ewha_classical * 1000, color='green')
    # plt.plot(year_fraction, (year_fraction - np.mean(year_fraction)) * trend * 1000, ls='--', color='green',
    #          label=r'OLS')
    # plt.plot(year_fraction, (year_fraction - np.mean(year_fraction)) * trend * 1100, ls='--', color='red',
    #          label=r'WLS')

    # for i in range(0, len(year_fraction), 2):
    #     plt.plot(
    #         [year_fraction[i]] * 2,
    #         (ewha_classical[i] + 20 * np.array([std[i], -std[i]])) * 1000,
    #         color='red',
    #         markersize=10,
    #         markeredgewidth=10,
    #     )
    plt.legend()
    plt.show()


def demo3():
    sigma_EWH_filepath = FileTool.get_project_dir(f'results/GlobalVarEWH_20230322/VGC500/20050101-20050131.hdf5')
    with h5py.File(sigma_EWH_filepath, 'r') as f:
        sigma_EWH = np.sqrt(np.array(f['data']))

    var_weight = 1 / sigma_EWH ** 2
    var_weight /= np.sum(var_weight)

    ocean_mask = np.load('../../temp/ocean_300km-buffer(360,720))_Uebbing.npy')
    ocean_mask = ocean_mask[::6, ::6]
    ocean_mask[np.where(ocean_mask >= 0.5)] = 1
    ocean_mask[np.where(ocean_mask < 0.5)] = 0

    ocean_mask[:7] = 0  # +- 67 deg
    ocean_mask[-7:] = 0
    ocean_mask_to_plot = ocean_mask.copy()
    ocean_mask_to_plot[np.where(ocean_mask_to_plot) == 0] = np.nan
    lat, lon = MathTool.get_global_lat_lon_range(3)
    plot_grids(
        np.array(
            [var_weight * ocean_mask_to_plot,
             sigma_EWH * ocean_mask_to_plot,
             sigma_EWH]
        ),
        lat, lon,
    )


if __name__ == '__main__':
    demo2()
    demo()
