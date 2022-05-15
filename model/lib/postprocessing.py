import numpy as np
from sklearn.metrics import mean_absolute_error as mae


Known_staion = ['S1','S2','S4','S5','S6',
                    'S7','S8','S9','S11','S12',
                    'S13','S14','S16','S17',
                    'S18','S21','S22','S23']
Unknown_staion = ['S3','S10','S15','S19','S20']
Faulty_station = ['S7', 'S11','S17']
def make_final_prediction(distance_dict,test_combinations, y_pred_reshaped, s_setting, sensor_ID):
    weighted_prediciton_all = []
    weighted_prediciton_dist_all = []

    if s_setting == 1:
        stations = Known_staion
    elif s_setting ==2:
        stations=Unknown_staion
    elif s_setting ==3:
        stations=Faulty_station
    else:
        raise ValueError
    for station in stations:
        weighted_prediciton_stations = []
        weighted_prediciton_dist = 0
        for i, combi in enumerate(y_pred_reshaped):
            if test_combinations[i][0] == station:
                current_distance = distance_dict[test_combinations[i][0] + "-" + test_combinations[i][1]]
                if current_distance == 0:
                    current_distance = 1
                sensor_divided = combi[:, 0, sensor_ID, :] / (current_distance)
                weighted_prediciton_stations.append(sensor_divided)
                weighted_prediciton_dist += 1 / (current_distance)
            else:
                continue

        weighted_prediciton_all.append(np.array(weighted_prediciton_stations))
        weighted_prediciton_dist_all.append(weighted_prediciton_dist)

    final_pred_all = []
    for i, wps in enumerate(weighted_prediciton_all):

        final_pred = []
        for ite in range(0, wps.shape[1]):
            final_pred.append(np.sum(wps[:, ite, :]) / weighted_prediciton_dist_all[i])
        final_pred_all.append(np.array(final_pred))
    return np.expand_dims(np.array(final_pred_all), axis=2)


def mase(y_true, y_pred, y_train):
    e_t = y_true - y_pred
    scale = mae(y_train[1:], y_train[:-1])
    mean = np.mean(np.abs(e_t / scale))
    return mean

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
CO2_norm_factors=100
O3_norm_factors=1000
NO_norm_factors=1000


def RMSE_MAE(CO_pred,NO_pred,O3_pred,pm10_pred,pm25_pred,CO_true,NO_true,O3_true,pm10_true,pm25_true,s_setting):
    MAE1 = np.empty([5])
    temp_mae = np.empty([CO_pred.shape[0], 5])
    RMSE1 = np.empty([5])
    temp_rmse = np.empty([CO_pred.shape[0], 5])
    if s_setting == 1:
        stations = Known_staion
    elif s_setting ==2:
        stations=Unknown_staion
    elif s_setting ==3:
        stations=Faulty_station
    else:
        raise ValueError
    print("-----****RMSE****-----")
    for st in range(0, len(stations)):
        print("Station:", stations[st])
        print("CO:", rmse(CO_pred[st, :, 0] / CO2_norm_factors, CO_true[st, :, 0] / CO2_norm_factors))
        print("NO:", rmse(NO_pred[st, :, 0] / NO_norm_factors, NO_true[st, :, 0] / NO_norm_factors))
        print("O3:", rmse(O3_pred[st, :, 0] / O3_norm_factors, O3_true[st, :, 0] / O3_norm_factors))
        print("PM10:", rmse(pm10_pred[st, :, 0], pm10_true[st, :, 0]))
        print("PM25:", rmse(pm25_pred[st, :, 0], pm25_true[st, :, 0]))
        print("------------------")
        RMSE1[0] = rmse(CO_pred[st, :, 0] / CO2_norm_factors, CO_true[st, :, 0] / CO2_norm_factors)
        RMSE1[1] = rmse(NO_pred[st, :, 0] / NO_norm_factors, NO_true[st, :, 0] / NO_norm_factors)
        RMSE1[2] = rmse(O3_pred[st, :, 0] / O3_norm_factors, O3_true[st, :, 0] / O3_norm_factors)
        RMSE1[3] = rmse(pm10_pred[st, :, 0], pm10_true[st, :, 0])
        RMSE1[4] = rmse(pm25_pred[st, :, 0], pm25_true[st, :, 0])
        temp_rmse[st, :] = RMSE1
    print("-----****Average RMSE****-----")
    print("CO:", rmse(CO_pred / CO2_norm_factors, CO_true / CO2_norm_factors))
    print("NO:", rmse(NO_pred / NO_norm_factors, NO_true / NO_norm_factors))
    print("O3:", rmse(O3_pred / O3_norm_factors, O3_true / O3_norm_factors))
    print("PM10:", rmse(pm10_pred, pm10_true))
    print("PM25:", rmse(pm25_pred, pm25_true))
    print("-----****MAE****-----")
    for st in range(0, len(stations)):
        print("Station:", stations[st])
        print("CO:", mae(CO_pred[st, :, 0] / CO2_norm_factors, CO_true[st, :, 0] / CO2_norm_factors))
        print("NO:", mae(NO_pred[st, :, 0] / NO_norm_factors, NO_true[st, :, 0] / NO_norm_factors))
        print("O3:", mae(O3_pred[st, :, 0] / O3_norm_factors, O3_true[st, :, 0] / O3_norm_factors))
        print("PM10:", mae(pm10_pred[st, :, 0], pm10_true[st, :, 0]))
        print("PM25:", mae(pm25_pred[st, :, 0], pm25_true[st, :, 0]))
        print("-------------")
        MAE1[0] = mae(CO_pred[st, :, 0] / CO2_norm_factors, CO_true[st, :, 0] / CO2_norm_factors)
        MAE1[1] = mae(NO_pred[st, :, 0] / NO_norm_factors, NO_true[st, :, 0] / NO_norm_factors)
        MAE1[2] = mae(O3_pred[st, :, 0] / O3_norm_factors, O3_true[st, :, 0] / O3_norm_factors)
        MAE1[3] = mae(pm10_pred[st, :, 0], pm10_true[st, :, 0])
        MAE1[4] = mae(pm25_pred[st, :, 0], pm25_true[st, :, 0])
        temp_mae[st, :] = MAE1
    print("-----****Average MAE****-----")
    print("CO:", mae(CO_pred[:, :, 0] / CO2_norm_factors, CO_true[:, :, 0] / CO2_norm_factors))
    print("NO:", mae(NO_pred[:, :, 0] / NO_norm_factors, NO_true[:, :, 0] / NO_norm_factors))
    print("O3:", mae(O3_pred[:, :, 0] / O3_norm_factors, O3_true[:, :, 0] / O3_norm_factors))
    print("PM10:", mae(pm10_pred[:, :, 0], pm10_true[:, :, 0]))
    print("PM25:", mae(pm25_pred[:, :, 0], pm25_true[:, :, 0]))
    print("----done------")
    return temp_mae,temp_rmse


def get_mase_sensor(CO_pred,NO_pred,O3_pred,pm10_pred,pm25_pred,CO_true,NO_true,O3_true,pm10_true,pm25_true,CO_train,NO_train,O3_train,pm10_train,pm25_train,s_setting):
    print("-----****MASE****-----")
    temp_mase = np.empty([CO_pred.shape[0], 5])
    MASE1 = np.empty([5])
    if s_setting == 1:
        stations = Known_staion
    elif s_setting ==2:
        stations=Unknown_staion
    elif s_setting ==3:
        stations=Faulty_station
    else:
        raise ValueError
    for st in range(0, len(stations)):
        print("Station:", stations[st])
        print("CO:", mase(CO_pred[st, :, 0] / CO2_norm_factors, CO_true[st, :, 0] / 100, CO_train[st, :, 0] / CO2_norm_factors))
        print("NO:", mase(NO_pred[st, :, 0] / NO_norm_factors, NO_true[st, :, 0] / NO_norm_factors, NO_train[st, :, 0] / NO_norm_factors))
        print("O3:", mase(O3_pred[st, :, 0] / O3_norm_factors, O3_true[st, :, 0] / O3_norm_factors, O3_train[st, :, 0] / O3_norm_factors))
        print("PM10:", mase(pm10_pred[st, :, 0], pm10_true[st, :, 0], pm10_train[st, :, 0]))
        print("PM25:", mase(pm25_pred[st, :, 0], pm25_true[st, :, 0], pm25_train[st, :, 0]))
        print("-------------")
        MASE1[0] = mase(CO_pred[st, :, 0] / CO2_norm_factors, CO_true[st, :, 0] / CO2_norm_factors, CO_train[st, :, 0] / CO2_norm_factors)
        MASE1[1] = mase(NO_pred[st, :, 0] / NO_norm_factors, NO_true[st, :, 0] / NO_norm_factors, NO_train[st, :, 0] / NO_norm_factors)
        MASE1[2] = mase(O3_pred[st, :, 0] / O3_norm_factors, O3_true[st, :, 0] / O3_norm_factors, O3_train[st, :, 0] / O3_norm_factors)
        MASE1[3] = mase(pm10_pred[st, :, 0], pm10_true[st, :, 0], pm10_train[st, :, 0])
        MASE1[4] = mase(pm25_pred[st, :, 0], pm25_true[st, :, 0], pm25_train[st, :, 0])
        temp_mase[st, :] = MASE1
    print("----done------")
    return temp_mase


def RMSE_MAE2(CO_pred,NO_pred,O3_pred,pm10_pred,pm25_pred,CO_true,NO_true,O3_true,pm10_true,pm25_true,s_setting):
    MAE1 = np.empty([5])
    temp_mae = np.empty([CO_pred.shape[0], 5])
    RMSE1 = np.empty([5])
    temp_rmse = np.empty([CO_pred.shape[0], 5])
    if s_setting == 1:
        stations = Known_staion
    elif s_setting ==2:
        stations=Unknown_staion
    elif s_setting ==3:
        stations=Faulty_station
    else:
        raise ValueError
    print("-----****MAE****-----")
    for st in range(0, len(stations)):
        if st==0:
            fs = [0,686,1022,1188,1356,180,516,1188,1356]
        elif st==1:
            fs = [0, 180, 348, 1188, 1524, 180, 348, 686, 1022]
        elif st == 2:
            fs = [0, 686, 854, 1188, 1524, 180, 516, 686, 854]
        CO_pred1 =CO_pred[st, fs[1]:fs[2],0]
        CO_pred1 =np.append(CO_pred1,CO_pred[st, fs[3]:fs[4],0],0)
        pm10_pred1 = pm10_pred[st, fs[1]:fs[2],0]
        pm10_pred1 = np.append(pm10_pred1, pm10_pred[st, fs[3]:fs[4],0])
        pm25_pred1= pm25_pred[st, fs[1]:fs[2],0]
        pm25_pred1 = np.append(pm25_pred1, pm25_pred[st, fs[3]:fs[4],0])
        NO_pred1 = NO_pred[st, fs[5]:fs[6],0]
        NO_pred1 = np.append(NO_pred1, NO_pred[st, fs[7]:fs[8],0])
        O3_pred1 = O3_pred[st, fs[5]:fs[6],0]
        O3_pred1= np.append(O3_pred1, O3_pred[st, fs[7]:fs[8],0])

        CO_true1 =CO_true[st, fs[1]:fs[2],0]
        CO_true1 = np.append(CO_true1,CO_true[st, fs[3]:fs[4],0])
        pm10_true1 = pm10_true[st, fs[1]:fs[2],0]
        pm10_true1 = np.append(pm10_true1, pm10_true[st, fs[3]:fs[4],0])
        pm25_true1 = pm25_true[st, fs[1]:fs[2],0]
        pm25_true1 = np.append(pm25_true1, pm25_true[st, fs[3]:fs[4],0])
        NO_true1 = NO_true[st, fs[5]:fs[6],0]
        NO_true1 = np.append(NO_true1, NO_true[st, fs[7]:fs[8],0])
        O3_true1 = O3_true[st, fs[5]:fs[6],0]
        O3_true1 = np.append(O3_true1, O3_true[st, fs[7]:fs[8],0])
        print("Station:", stations[st])
        print("CO:", mae(CO_pred1 / CO2_norm_factors, CO_true1 / CO2_norm_factors))
        print("NO:", mae(NO_pred1 / NO_norm_factors, NO_true1 / NO_norm_factors))
        print("O3:", mae(O3_pred1 / O3_norm_factors, O3_true1 / O3_norm_factors))
        print("PM10:", mae(pm10_pred1, pm10_true1))
        print("PM25:", mae(pm25_pred1, pm25_true1))
        print("-------------")
        MAE1[0] = mae(CO_pred1 / CO2_norm_factors, CO_true1 / CO2_norm_factors)
        MAE1[1] = mae(NO_pred1 / NO_norm_factors, NO_true1 / NO_norm_factors)
        MAE1[2] = mae(O3_pred1 / O3_norm_factors, O3_true1 / O3_norm_factors)
        MAE1[3] = mae(pm10_pred1, pm10_true1)
        MAE1[4] = mae(pm25_pred1, pm25_true1)
        temp_mae[st, :] = MAE1
        RMSE1[0] = rmse(CO_pred1 / CO2_norm_factors, CO_true1 / CO2_norm_factors)
        RMSE1[1] = rmse(NO_pred1 / NO_norm_factors, NO_true1 / NO_norm_factors)
        RMSE1[2] = rmse(O3_pred1 / O3_norm_factors, O3_true1 / O3_norm_factors)
        RMSE1[3] = rmse(pm10_pred1, pm10_true1)
        RMSE1[4] = rmse(pm25_pred1, pm25_true1)
        temp_rmse[st, :] = RMSE1
    print("----done------")
    return temp_mae, temp_rmse