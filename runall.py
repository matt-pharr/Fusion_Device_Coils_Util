coilnameslist = """ITER_2cmshift_CS1L.dat
ITER_2cmshift_CS1U.dat
ITER_2cmshift_CS2L.dat
ITER_2cmshift_CS2U.dat
ITER_2cmshift_CS3L.dat
ITER_2cmshift_CS3U.dat
ITER_2cmshift_PF1.dat
ITER_2cmshift_PF2.dat
ITER_2cmshift_PF3.dat
ITER_2cmshift_PF4.dat
ITER_2cmshift_PF5.dat
ITER_2cmshift_PF6.dat
ITER_5mrtilt_CS1L.dat
ITER_5mrtilt_CS1U.dat
ITER_5mrtilt_CS2L.dat
ITER_5mrtilt_CS2U.dat
ITER_5mrtilt_CS3L.dat
ITER_5mrtilt_CS3U.dat
ITER_5mrtilt_PF1.dat
ITER_5mrtilt_PF2.dat
ITER_5mrtilt_PF3.dat
ITER_5mrtilt_PF4.dat
ITER_5mrtilt_PF5.dat
ITER_5mrtilt_PF6.dat
ITER_CS1L_2cmshift_CS1L.dat
ITER_CS1L_5mrtilt_CS1L.dat
ITER_CS1U_2cmshift_CS1U.dat
ITER_CS1U_5mrtilt_CS1U.dat
ITER_CS2L_2cmshift_CS2L.dat
ITER_CS2L_5mrtilt_CS2L.dat
ITER_CS2U_2cmshift_CS2U.dat
ITER_CS2U_5mrtilt_CS2U.dat
ITER_CS3L_2cmshift_CS3L.dat
ITER_CS3L_5mrtilt_CS3L.dat
ITER_CS3U_2cmshift_CS3U.dat
ITER_CS3U_5mrtilt_CS3U.dat
ITER_PF1_2cmshift_PF1.dat
ITER_PF1_5mrtilt_PF1.dat
ITER_PF1_5mrtilt_PF1s.dat
ITER_PF2_2cmshift_PF2.dat
ITER_PF2_5mrtilt_PF2.dat
ITER_PF3_2cmshift_PF3.dat
ITER_PF3_5mrtilt_PF3.dat
ITER_PF4_2cmshift_PF4.dat
ITER_PF4_5mrtilt_PF4.dat
ITER_PF5_2cmshift_PF5.dat
ITER_PF5_5mrtilt_PF5.dat
ITER_PF6_2cmshift_PF6.dat
ITER_PF6_5mrtilt_PF6.dat
ITER_unper_PF1s.dat""".splitlines()

equilibriatrimmed = """pfpo1_dina_2MA_k1.0_iter
pfpo1_dina_3.5MA_k1.7_iter
pfpo1_dina_3MA_k1.1_iter
pfpo1_dina_3MA_k1.2_iter
pfpo1_dina_3MA_k1.3_iter
pfpo1_dina_3MA_k1.4_iter
pfpo1_dina_3MA_k1.5_iter
pfpo1_dina_3MA_k1.6_iter""".splitlines()

equil_location = '/p/gpec/users/jpark/data/equilibria/efit/'



for coil in coilnameslist:
    
    # Load in appropriate coils files
    root['GPEC']['COILS']['SETTINGS']['PHYSICS']['coil_inputs_key'] = coil[5:-4]
    root['GPEC']['COILS']['SETTINGS']['PHYSICS']['coil_location'] = '/u/mpharr/Projects/ITER_erf/coils/' + coil
    root['GPEC']['COILS']['SCRIPTS']['load_coils'].run()

    # Set coil currents
    root['GPEC']['SETTINGS']['PHYSICS']['current'][coil[5:-4]]['current'] = [2e5]*9

    
    # Loop through equilibria
    for eq in equilibriatrimmed:

        # Load in equilibria
        eqloc = equil_location + eq
        eqtimes = [1]
        root['GPEC']['SCRIPTS']['load_equilibrium'].run(location=eqloc, times=eqtimes)

        # Set run name
        runname = eq[11:-5] + '_' + coil[5:-4]

        # Run
        root['SETTINGS']['PHYSICS']['run_times'] = times
        root['SETTINGS']['PHYSICS']['run_dcon'] = True
        root['SETTINGS']['PHYSICS']['run_gpec'] = True
        root['SETTINGS']['PHYSICS']['run_qsub'] = True
        root['SCRIPTS']['run_exes'].run(run_key=runname)