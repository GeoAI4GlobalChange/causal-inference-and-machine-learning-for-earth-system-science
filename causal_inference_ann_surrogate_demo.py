import numpy as np
import os
from tigramite import data_processing_ET as pp
from tigramite.pcmci_ET import PCMCI
from tigramite.independence_tests import  CMIknn
import copy
import random
from sklearn.neural_network import MLPRegressor
import pickle
from sklearn.metrics import explained_variance_score
import pandas as pd
def causal_inference_obs():
    path = r'observation_IGBP_combine/'#the path of the data that includes all the observed data with different IGBP types.
    files = os.listdir(path)
    max_lag=1
    len_limit=100
    indexes=[]
    for lag in range(0,max_lag+1):
        indexes.append(f't_{lag}')
    #for file in files:
    for file_idx in range(0,5):
        file=files[file_idx]
        data_path = path + file#files[index]
        df=pd.read_csv(data_path)
        var_names = ['ET_fluxunit_detrend', 'LAI_detrend', 'GPP_NT_VUT_MEAN_detrend','SWC_F_MDS_1_detrend', 'VPD_ERA_detrend', 'TA_ERA_detrend',
                     'P_ERA_detrend', 'SW_IN_ERA_detrend', ]
        data=df[var_names].values
        for col_idx in range(data.shape[1]):
            temp_data=data[:,col_idx]
            temp_data_per=temp_data[~np.isnan(temp_data)]
            min_value=np.percentile(temp_data_per,5)
            max_value = np.percentile(temp_data_per, 95)
            print(var_names[col_idx],min_value,max_value)
            temp_data[temp_data<min_value]=np.nan
            temp_data[temp_data > max_value] = np.nan
            data[:, col_idx]=temp_data
        not_nan_len=len(data)-np.sum(np.any(np.isnan(data),axis=1))
        print(file)
        print('not nan len',not_nan_len)

        if len(data.shape)==2 and 'ET_fluxunit_detrend' in var_names and not_nan_len>len_limit:
            if var_names.index('ET_fluxunit_detrend')==0:
                flag_value=-9999
                data[np.isnan(data)]=flag_value
                dataframe = pp.DataFrame(data, var_names=var_names,missing_flag=flag_value)
                parcorr = CMIknn(significance='shuffle_test',sig_samples=100,)#significance='fixed_thres',fixed_thres=0.001 significance='shuffle_test',fixed_thres=0.005,
                pcmci_parcorr = PCMCI(
                    dataframe=dataframe,
                    cond_ind_test=parcorr,
                    selected_variables=[0],
                    verbosity=0)
                results = pcmci_parcorr.run_pcmci(tau_min=0,tau_max=max_lag, pc_alpha=0.05,max_combinations=100,first_not_driver=True,max_conds_dim=1,max_conds_py=1,max_conds_px=1)#max_combinations=max_lag*len(var_names),
                val_matrix = results['val_matrix'][:,0]
                p_matrix = results['p_matrix'][:,0]
                df_result=pd.DataFrame(val_matrix,columns=indexes,index=var_names)
                df_sig = pd.DataFrame(p_matrix, columns=indexes, index=var_names)
                save_path=f'ET_condim_1_'+file
                #save_path = file
                df_result.to_csv(save_path)
                sig_path=f'ET_condim_1_sig_'+file
                #save_path = file
                df_sig.to_csv(sig_path)
def causal_inference_cmip():
    model_dirs = os.listdir(r'CMIP_IGBP_combine/')# The path includes all the cmip models used in the experiment
    for model_dir in model_dirs:
        path = f'CMIP_IGBP_combine/{model_dir}/'#The path includes the datasets that includes all the cmip model simulated data with different IGBP types.
        files = os.listdir(path)
        max_lag = 1
        len_limit = 100
        indexes = []
        print(model_dir)

        for lag in range(0, max_lag + 1):
            indexes.append(f't_{lag}')
        for file in files:
            data_path = path + file
            df = pd.read_csv(data_path)
            var_names = ['ET_fluxunit_detrend', 'lai_detrend', 'gpp_detrend', 'mrso_detrend', 'VPD_detrend',
                         'tas_detrend', 'pr_detrend',
                         'rsds_detrend']
            data = df[var_names].values

            not_nan_len = len(data) - np.sum(np.any(np.isnan(data), axis=1))
            print(file)
            print('not nan len', not_nan_len)

            if len(data.shape) == 2 and 'ET_fluxunit_detrend' in var_names and not_nan_len > len_limit:
                if var_names.index('ET_fluxunit_detrend') == 0:
                    flag_value = -9999
                    data[np.isnan(data)] = flag_value
                    dataframe = pp.DataFrame(data, var_names=var_names, missing_flag=flag_value)
                    parcorr = CMIknn(significance='shuffle_test',
                                     sig_samples=100, )
                    pcmci_parcorr = PCMCI(
                        dataframe=dataframe,
                        cond_ind_test=parcorr,
                        selected_variables=[0],
                        verbosity=0)
                    results = pcmci_parcorr.run_pcmci(tau_min=0, tau_max=max_lag, pc_alpha=0.05, max_combinations=100,
                                                      first_not_driver=True, max_conds_dim=1, max_conds_py=1,
                                                      max_conds_px=1)
                    val_matrix = results['val_matrix'][:, 0]
                    p_matrix = results['p_matrix'][:, 0]
                    df_result = pd.DataFrame(val_matrix, columns=indexes, index=var_names)
                    df_sig = pd.DataFrame(p_matrix, columns=indexes, index=var_names)
                    save_path = f'{model_dir}/ET_condim_1_' + file
                    df_result.to_csv(save_path)
                    sig_path = f'{model_dir}/ET_condim_1_sig_' + file
                    df_sig.to_csv(sig_path)
def CMIP_ANN_surrogate():
    test_precent = 0.1
    df_result = pd.DataFrame(columns=['model', 'exp_id', 'IGBP', 'explained_variance'])
    df_idx = 0
    site_types = ['EBF', 'ENF', 'DBF', 'SAV', 'GRA']
    Fea_importance = pd.DataFrame(columns=['model', 'seed', 'type',
                                           'LAI', 'GPP',
                                           'TA', 'RSDS',
                                           'SWC',
                                           'VPD', 'Pr'])
    dir = r'CMIP_sites/'# the path of data that includes all the cmip model simulated data in different sites.
    models = os.listdir(dir)
    Fea_idx = 0
    for seed in range(1):
        random.seed(seed)
        for model in models:
            dir_data = dir + model + '/'
            files = os.listdir(dir_data)
            target = 'EF'
            cols = [
                'lai',
                'gpp',
                'tas',
                'rsds',
                'mrso',
                'VPD',
                'pr',
            ]
            train_datasets = {}
            val_datasets = {}
            test_datasets = {}

            site_type_dict = {}
            df = pd.read_csv(r'Site_information.csv')[
                ['SITE_NAME', 'IGBP']].values
            for idx in range(df.shape[0]):
                site_type_dict[df[idx, 0]] = df[idx, 1]
            site_start = {}
            for item in site_types:
                site_start[item] = True
            for file in files:
                print(file)
                temp_type = site_type_dict[file.split('.csv')[0]]
                if temp_type in site_types:
                    # temp_type='all'#merge all types
                    file_path = dir_data + file
                    data = pd.read_csv(file_path)
                    site_colms = data.columns.values.tolist()
                    site_final_cols = copy.deepcopy(cols)
                    length = data.shape[0]
                    data1 = data
                    X_train1 = np.zeros((len(data1), len(site_final_cols)))
                    for i, name in enumerate(site_final_cols):
                        driver = data1[name]
                        if name == 'pr':
                            driver = driver * 3600 * 24  # kg/m2/s->mm/day
                        if name == 'mrso':
                            driver = driver / 100  # kg/m2->%
                        X_train1[:, i] = driver  # .fillna(method="bfill")
                    y_train1 = np.array(data1[target])  # .fillna(method='ffill'))

                    y_nan_mask = np.isnan(y_train1) | np.isinf(y_train1)
                    y_extreme_mask = np.abs(y_train1) > pow(10, 5)
                    y_nan_mask = (y_nan_mask | y_extreme_mask)
                    x_temp = X_train1.copy()
                    x_nan_mask = np.any(np.isnan(x_temp), axis=1) | np.any(np.isinf(x_temp), axis=1)
                    x_extreme_mask = np.any(np.abs(x_temp) > pow(10, 5), axis=1)
                    x_nan_mask = (x_nan_mask | x_extreme_mask)
                    nan_mask = (y_nan_mask | x_nan_mask)
                    if np.sum(nan_mask == False) > 20:
                        X_train1 = X_train1[nan_mask == False]
                        y_train1 = y_train1[nan_mask == False]
                        all_idxs = [item_idx for item_idx in range(y_train1.shape[0])]
                        test_idxs = random.sample(all_idxs, int(test_precent * len(all_idxs)))
                        train_idxs = np.array(list(set(all_idxs).difference(set(test_idxs))))
                        test_idxs = np.array(test_idxs)
                        if site_start[temp_type]:
                            train_datasets[temp_type] = {}
                            train_datasets[temp_type]['X'] = X_train1[train_idxs]
                            train_datasets[temp_type]['Y'] = y_train1[train_idxs]
                            test_datasets[temp_type] = {}
                            test_datasets[temp_type]['X'] = X_train1[test_idxs]
                            test_datasets[temp_type]['Y'] = y_train1[test_idxs]
                            site_start[temp_type] = False
                        else:
                            train_datasets[temp_type]['X'] = np.concatenate(
                                [train_datasets[temp_type]['X'], X_train1[train_idxs]], axis=0)
                            train_datasets[temp_type]['Y'] = np.concatenate(
                                [train_datasets[temp_type]['Y'], y_train1[train_idxs]], axis=0)
                            test_datasets[temp_type]['X'] = np.concatenate(
                                [test_datasets[temp_type]['X'], X_train1[test_idxs]], axis=0)
                            test_datasets[temp_type]['Y'] = np.concatenate(
                                [test_datasets[temp_type]['Y'], y_train1[test_idxs]], axis=0)
            print(train_datasets.keys())

            for type in train_datasets:
                x_train = train_datasets[type]['X']
                y_train = train_datasets[type]['Y']

                x_test = test_datasets[type]['X']
                y_test = test_datasets[type]['Y']

                x_all = np.concatenate((x_train, x_test), axis=0)
                y_all = np.concatenate((y_train, y_test), axis=0)

                means = np.mean(x_all, axis=0)
                stds = np.std(x_all, axis=0)
                y_mean = np.mean(y_all, axis=0)
                y_std = np.std(y_all, axis=0)
                np.save(f'{model}_{type}_x_mean.npy', means)
                np.save(f'{model}_{type}_x_std.npy', stds)
                np.save(f'{model}_{type}_y_mean.npy', y_mean)
                np.save(f'{model}_{type}_y_std.npy', y_std)

                x_train = (x_train - means) / (stds + pow(10, -8))
                y_train = (y_train - y_mean) / (y_std + pow(10, -8))

                x_all = (x_all - means) / (stds + pow(10, -8))
                y_all = (y_all - y_mean) / (y_std + pow(10, -8))
                x_test = x_all
                y_test = y_all * (y_std + pow(10, -8)) + y_mean

                regressor = MLPRegressor(hidden_layer_sizes=(50, 40, 5), activation='tanh',  # early_stopping=True,
                                         solver='adam', beta_1=0.9, beta_2=0.999, tol=pow(10, -5), n_iter_no_change=20,
                                         learning_rate='invscaling', max_iter=5000, alpha=0.0001, batch_size=16,
                                         learning_rate_init=0.001)
                my_model = regressor
                my_model.fit(x_train, y_train)
                preds = my_model.predict(x_test) * (y_std + pow(10, -8)) + y_mean
                # save the model to disk
                para_filename = f'{model}_{type}_{seed}.para'
                pickle.dump(my_model, open(para_filename, 'wb'))

                # # load the model from disk
                # loaded_model = pickle.load(open(filename), 'rb')
                true = y_test
                # mse = mean_squared_error(true, preds)
                # mae = mean_absolute_error(true, preds)
                # r=stats.pearsonr(true, preds)[0]
                exp_vars = explained_variance_score(true, preds)
                print(model, type, seed, exp_vars)
                df_result.loc[df_idx] = [model, type, seed, exp_vars]
                df_idx += 1
    df_result.to_csv(r'ANN_performance_stats_all_train.csv')
def CMIP_ANN_surrogate_finetune():
    test_precent = 0.1
    site_types = ['EBF', 'ENF', 'DBF', 'SAV', 'GRA']
    dir = r'CMIP_sites/'# The path of data that includes all the cmip model simulated data in different sites
    flux_cols = ['EF_noExtreme', 'GPP_NT_VUT_MEAN', 'VPD_ERA', 'TA_ERA', 'SW_IN_ERA', 'P_ERA', 'LAI']
    flux_cmip_cols = ['EF', 'gpp', 'VPD', 'tas', 'rsds', 'pr', 'lai', ]  # colums to surrogate with flux data
    df_result = pd.DataFrame(columns=['model', 'exp_id', 'IGBP', 'explained_variance'])
    df_idx = 0
    models = os.listdir(dir)
    for seed in range(1):
        random.seed(seed)
        for model in models:
            dir_data = dir + model + '/'
            files = os.listdir(dir_data)
            target = 'EF'
            cols = [
                'lai',
                'gpp',
                'tas',
                'rsds',
                'mrso',
                'VPD',
                'pr',
            ]  # 'ps',
            train_datasets = {}
            val_datasets = {}
            test_datasets = {}

            site_type_dict = {}
            df = pd.read_csv(r'Site_information.csv')[
                ['SITE_NAME', 'IGBP']].values
            for idx in range(df.shape[0]):
                site_type_dict[df[idx, 0]] = df[idx, 1]
            site_start = {}
            for item in site_types:
                site_start[item] = True
            # site_start['all'] = True  ##merge all types
            for file in files:
                print(file)
                temp_type = site_type_dict[file.split('.csv')[0]]
                if temp_type in site_types:
                    # temp_type = 'all'  # merge all types
                    file_path = dir_data + file
                    data = pd.read_csv(file_path)
                    flux_net_file_path = r'Monthly_site_withLAI_noExEF/' + file  # f'{file.split(".csv")[0]}_day2mon.csv'
                    flux_data = pd.read_csv(flux_net_file_path)
                    #########################################
                    # time cut
                    start_year = flux_data['Year'].values[0]
                    start_month = flux_data['Month'].values[0]
                    start_idx = data[(data['Year'] == start_year) & (data['Month'] == start_month)].index.tolist()
                    data = data.iloc[start_idx[0]:(start_idx[0] + len(flux_data)), :]
                    site_final_cols = copy.deepcopy(cols)
                    data1 = data
                    y_train2 = np.array(data1[target])  # cmip target
                    x_train2 = np.array(data1[site_final_cols])  # cmip drivers
                    x_train2[:, site_final_cols.index('pr')] = x_train2[:, site_final_cols.index(
                        'pr')] * 3600 * 24  # kg/m2/s->mm/day
                    x_train2[:, site_final_cols.index('mrso')] = x_train2[:,
                                                                 site_final_cols.index('mrso')] / 100  # kg/m2->%
                    for name in flux_cols:
                        data1[flux_cmip_cols[flux_cols.index(name)]] = flux_data[
                            name].values  # surrogate cmip drivers with flux drivers
                    X_train1 = np.zeros((len(data1), len(site_final_cols)))  ##flux drivers
                    for i, name in enumerate(site_final_cols):
                        X_train1[:, i] = data1[name]  # .fillna(method="bfill")##observed flux drivers
                    y_train1 = np.array(data1[target])  # .fillna(method='ffill')) #observed EF
                    y_nan_mask = (np.isnan(y_train1)) | (np.isinf(y_train1)) | (np.isnan(y_train2))
                    y_extreme_mask = np.abs(y_train1) > pow(10, 5)
                    y_nan_mask = (y_nan_mask | y_extreme_mask)
                    x_temp = X_train1.copy()
                    x_nan_mask = np.any(np.isnan(x_temp), axis=1) | np.any(np.isinf(x_temp), axis=1)
                    x_extreme_mask = np.any(np.abs(x_temp) > pow(10, 5), axis=1)
                    x_nan_mask = (x_nan_mask | x_extreme_mask)
                    nan_mask = (y_nan_mask | x_nan_mask)
                    if np.sum(nan_mask == False) > 20:  # to exclude sites with limited lengths
                        X_train1 = X_train1[nan_mask == False]  # flux drivers
                        y_train1 = y_train1[nan_mask == False]  # flux targets
                        all_idxs = [item_idx for item_idx in range(y_train1.shape[0])]
                        test_idxs = random.sample(all_idxs, int(test_precent * len(all_idxs)))
                        train_idxs = np.array(list(set(all_idxs).difference(set(test_idxs))))
                        test_idxs = np.array(test_idxs)
                        if site_start[temp_type]:
                            train_datasets[temp_type] = {}
                            train_datasets[temp_type]['X'] = X_train1[train_idxs]
                            train_datasets[temp_type]['Y'] = y_train1[train_idxs]
                            test_datasets[temp_type] = {}
                            test_datasets[temp_type]['X'] = X_train1[test_idxs]
                            test_datasets[temp_type]['Y'] = y_train1[test_idxs]
                            site_start[temp_type] = False
                        else:
                            train_datasets[temp_type]['X'] = np.concatenate(
                                [train_datasets[temp_type]['X'], X_train1[train_idxs]], axis=0)
                            train_datasets[temp_type]['Y'] = np.concatenate(
                                [train_datasets[temp_type]['Y'], y_train1[train_idxs]], axis=0)
                            test_datasets[temp_type]['X'] = np.concatenate(
                                [test_datasets[temp_type]['X'], X_train1[test_idxs]], axis=0)
                            test_datasets[temp_type]['Y'] = np.concatenate(
                                [test_datasets[temp_type]['Y'], y_train1[test_idxs]], axis=0)
                print(train_datasets.keys())
            for type in train_datasets:
                x_train = train_datasets[type]['X']
                y_train = train_datasets[type]['Y']
                x_test = test_datasets[type]['X']
                y_test = test_datasets[type]['Y']
                x_all = np.concatenate((x_train, x_test), axis=0)
                y_all = np.concatenate((y_train, y_test), axis=0)
                means = np.load(f'obs_{temp_type}_x_mean.npy')
                stds = np.load(f'obs_{temp_type}_x_std.npy')
                y_mean = np.load(f'obs_{temp_type}_y_mean.npy')
                y_std = np.load(f'obs_{temp_type}_y_std.npy')
                x_train = (x_train - means) / (stds + pow(10, -8))
                y_train = (y_train - y_mean) / (y_std + pow(10, -8))

                x_all = (x_all - means) / (stds + pow(10, -8))
                y_all = (y_all - y_mean) / (y_std + pow(10, -8))
                x_test = x_all
                y_test = y_all * (y_std + pow(10, -8)) + y_mean

                para_filename = f'{model}_{temp_type}_{seed}.para'  # {model}_{type}_{seed}.para #{model}_all_{seed}.para
                f = open(para_filename, 'rb')
                my_model = pickle.load(f)
                my_model.learning_rate_init = pow(10, -5)
                my_model.fit(x_train, y_train)
                preds = my_model.predict(x_test) * (y_std + pow(10, -8)) + y_mean
                preds[preds > 1] = 1
                preds[preds < 0] = 0
                # save the model to disk
                para_filename = f'finetuned_{model}_{type}_{seed}.para'
                pickle.dump(my_model, open(para_filename, 'wb'))
                # # load the model from disk
                # loaded_model = pickle.load(open(filename), 'rb')
                true = y_test
                exp_vars = explained_variance_score(true, preds)
                print(model, type, seed, exp_vars)
                df_result.loc[df_idx] = [model, type, seed, exp_vars]
                df_idx += 1
    df_result.to_csv(
        r'finetuned_ANN_performance_stats_all_train.csv')
if __name__=='__main__':
    causal_inference_obs()# causal inference for observations
    causal_inference_cmip()# causal inference for cmip models
    CMIP_ANN_surrogate()#use ANN to surrogate cmip models
    CMIP_ANN_surrogate_finetune()#use observation to finetune the ANN surrogate model


