# %%

from pathlib import Path

# cwd = Path().resolve()
# print(cwd)
import os
import sys


file_dir = 'E:\Projects\Tempora Graph\GNN_forecasting\Final_check'

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from AGCRN import AGCRN as Network
from BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_KoreaRail_dataloader,get_combinations
from lib.TrainInits import print_model_parameters
from torchsummary import summary
from pathlib import Path
import json
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from lib.postprocessing import make_final_prediction as mfp
from lib.postprocessing import RMSE_MAE as RMSE_MAE
from lib.postprocessing import RMSE_MAE2 as imput_perf
from lib.postprocessing import get_mase_sensor as mase_metric

# %%

# *************************************************************************#
Mode = 'test'
DEBUG = 'False'
DATASET = 'Koreaair'
DEVICE = 'cuda:0'
MODEL = 'DAGCGN'
t_stations = 18
# %%

# get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)

config = configparser.ConfigParser()
config.read(config_file)

# %%

from lib.metrics import MAE_torch


def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


# parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
args.add_argument('--horizon', default=1, type=int, help='How far in future to predict')

# data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--switch_x_y', default=config['data']['switch_x_y'], type=eval)


args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
args.add_argument('--grid_km', default=config['data']['grid_km'], type=int)
args.add_argument('--load_numpy_data_train', default=config['data']['load_numpy_data_train'], type=eval)
args.add_argument('--load_numpy_data_validation', default=config['data']['load_numpy_data_validation'], type=eval)
args.add_argument('--load_numpy_data_test', default=config['data']['load_numpy_data_test'], type=eval)

args.add_argument('--nearest_stations', default=config['data']['nearest_stations'], type=int)
args.add_argument('--station_setting', default=config['data']['station_setting'], type=int)

# model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
# train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)

args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')
# test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
# log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args = args.parse_args(args=[])

# %%

init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

# %%

horizons = config.get("data", "horizons")
horizons_value_list = json.loads(horizons)
current_time = datetime.now().strftime('%Y%m%d%H%M%S')

for horizon in horizons_value_list:
    args.horizon=horizon
    print("********* Horizon =", args.horizon,"*********")
    # init model
    model = Network(args)
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)


    # load dataset
    train_loader, val_loader, test_loader, scaler, test_combinations, distance_dict, Unknown_staion = get_KoreaRail_dataloader(args,
                                                                                                                               normalizer=args.normalizer,
                                                                                                                               tod=args.tod, dow=False,
                                                                                                                               weather=False, single=True)
    print("*****-----Dataset preparation****-----")
    # init loss function, optimizer
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse' or args.loss_func == 'rmse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError


    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)




    current_dir = file_dir
    log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time, "t+"+str(args.horizon))
    args.log_dir = log_dir
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                      args, lr_scheduler=lr_scheduler)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':


        path_model = "../experiments/Koreaair/t+{}/best_model_t+{}.pth".format(str(args.horizon),str(args.horizon))
        print("Load saved model from:", path_model)
        model.load_state_dict(torch.load(path_model))


        if args.station_setting==1:
            trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
        else:
            trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)

        y_true = np.load("np_data/Koreaair_test_true_t+" + str(args.horizon) + ".npy")
        y_pred = np.load("np_data/Koreaair_test_pred_t+" + str(args.horizon) + ".npy")

        test_combinations, distance_dict, Known_staion, Unknown_staion, Faulty_station=get_combinations(args)




        if args.station_setting == 1:
            y_pred_reshaped = y_pred.reshape(-1, int(y_pred.shape[0] / len(test_combinations)), y_pred.shape[1], y_pred.shape[2], y_pred.shape[3])
        elif args.station_setting == 2:
            y_pred_reshaped = y_pred.reshape(-1, int(y_pred.shape[0] / len(test_combinations)), y_pred.shape[1], y_pred.shape[2], y_pred.shape[3])
        elif args.station_setting == 3:
            y_pred_reshaped = y_pred.reshape(-1, int(y_pred.shape[0] / len(test_combinations)), y_pred.shape[1], y_pred.shape[2], y_pred.shape[3])
        else:
            raise ValueError




        def get_y_true(reshape_ytrue, sensor_id, hours_pred):
            return reshape_ytrue[:, 0, :, hours_pred, sensor_id, :]


        def rmse(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())

        Nearest_stations = args.nearest_stations
        if args.station_setting==1:
            y_true_re = y_true.reshape(len(Known_staion), Nearest_stations, -1, y_true.shape[1], y_true.shape[2], y_true.shape[3])

        elif args.station_setting==2:
            y_true_re = y_true.reshape(len(Unknown_staion), Nearest_stations, -1, y_true.shape[1], y_true.shape[2], y_true.shape[3])
        elif args.station_setting == 3:
            y_true_re = y_true.reshape(len(Faulty_station), Nearest_stations, -1, y_true.shape[1], y_true.shape[2], y_true.shape[3])
        else:
            raise ValueError



        s_setting=args.station_setting
        HOUR = 0
        DISTANCE_pred = mfp(distance_dict, test_combinations, y_pred_reshaped, s_setting, sensor_ID=0)
        CO_pred = mfp(distance_dict, test_combinations, y_pred_reshaped, s_setting, sensor_ID=1)
        NO_pred = mfp(distance_dict, test_combinations, y_pred_reshaped, s_setting, sensor_ID=2)
        O3_pred = mfp(distance_dict, test_combinations, y_pred_reshaped, s_setting, sensor_ID=3)
        pm10_pred = mfp(distance_dict, test_combinations, y_pred_reshaped, s_setting, sensor_ID=4)
        pm25_pred = mfp(distance_dict, test_combinations, y_pred_reshaped, s_setting, sensor_ID=5)

        CO_true = get_y_true(y_true_re, sensor_id=1, hours_pred=HOUR)
        NO_true = get_y_true(y_true_re, sensor_id=2, hours_pred=HOUR)
        O3_true = get_y_true(y_true_re, sensor_id=3, hours_pred=HOUR)
        pm10_true = get_y_true(y_true_re, sensor_id=4, hours_pred=HOUR)
        pm25_true = get_y_true(y_true_re, sensor_id=5, hours_pred=HOUR)
        if args.station_setting==1:
            y_train = np.load("np_data/y_tra_t+" + str(args.horizon) + ".npy")
            y_train_re = y_train.reshape(len(Known_staion), Nearest_stations, -1, y_train.shape[1], y_train.shape[2], y_train.shape[3])
            CO_train = get_y_true(y_train_re, sensor_id=1, hours_pred=HOUR)
            NO_train = get_y_true(y_train_re, sensor_id=2, hours_pred=HOUR)
            O3_train = get_y_true(y_train_re, sensor_id=3, hours_pred=HOUR)
            pm10_train = get_y_true(y_train_re, sensor_id=4, hours_pred=HOUR)
            pm25_train = get_y_true(y_train_re, sensor_id=5, hours_pred=HOUR)

        if args.station_setting == 1:
            MAEall,RMSEall = RMSE_MAE(CO_pred, NO_pred, O3_pred, pm10_pred, pm25_pred, CO_true, NO_true, O3_true, pm10_true,
                                pm25_true,s_setting)

            MASEall=mase_metric(CO_pred,NO_pred,O3_pred,pm10_pred,pm25_pred,CO_true,NO_true,O3_true,pm10_true,pm25_true,CO_train,NO_train,O3_train,pm10_train,pm25_train,s_setting)

            sio.savemat("./matlab_data/MASE_t+" + str(args.horizon) + ".mat", {"MASE": MASEall})
        elif args.station_setting == 2:
            MAEall, RMSEall = RMSE_MAE(CO_pred, NO_pred, O3_pred, pm10_pred, pm25_pred, CO_true, NO_true, O3_true,
                                        pm10_true,
                                        pm25_true, s_setting)

        elif args.station_setting == 3:
            MAEall, RMSEall  = imput_perf(CO_pred, NO_pred, O3_pred, pm10_pred, pm25_pred, CO_true, NO_true, O3_true,
                                        pm10_true,
                                        pm25_true, s_setting)
        else:
            raise ValueError



        # Saving Matlab files
        sio.savemat("./matlab_data/RMSE_t+" + str(args.horizon) + ".mat", {"RMSE": RMSEall})
        sio.savemat("./matlab_data/MAE_t+" + str(args.horizon) + ".mat", {"MAE": MAEall})
        sio.savemat("./matlab_data/GNNPM25_t+" + str(args.horizon) + ".mat", {"predpm25": pm25_pred})
        sio.savemat("./matlab_data/GNNPM10_t+" + str(args.horizon) + ".mat", {"predpm10": pm10_pred})
        sio.savemat("./matlab_data/GNNCO_t+" + str(args.horizon) + ".mat", {"predco": CO_pred / 100})
        sio.savemat("./matlab_data/GNNNO_t+" + str(args.horizon) + ".mat", {"predno": NO_pred / 1000})
        sio.savemat("./matlab_data/GNNO3_t+" + str(args.horizon) + ".mat", {"predo3": O3_pred / 1000})
        sio.savemat("./matlab_data/GNNPM25t_t+" + str(args.horizon) + ".mat", {"testpm25": pm25_true})
        sio.savemat("./matlab_data/GNNPM10t_t+" + str(args.horizon) + ".mat", {"testpm10": pm10_true})
        sio.savemat("./matlab_data/GNNCOt_t+" + str(args.horizon) + ".mat", {"testco": CO_true / 100})
        sio.savemat("./matlab_data/GNNNOt_t+" + str(args.horizon) + ".mat", {"testno": NO_true / 1000})
        sio.savemat("./matlab_data/GNNO3t_t+" + str(args.horizon) + ".mat", {"testo3": O3_true / 1000})
    else:
        raise ValueError
    trainer.logger.handlers.clear()
    print("done")






