import torch
import numpy as np
import torch.utils.data
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        # print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler



def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader
def read_distanace_matrix_excel(filename, sheet_name="Sheet1"):
    df = pd.read_excel(filename, sheet_name)
    df_dict = df.to_dict()
    temp_dict = {}
    for src_key, values in df_dict.items():
        if src_key == "Stations":
            continue
        for dst_key, dst_value in values.items():
            temp_dict[str(src_key) + "-" + "S" + str(dst_key + 1)] = round(dst_value, 2)
    return temp_dict

def find_closest (args,source, dist_dict,known_staion, Unknown_staion, Faulty_station, num=5):
    if args.station_setting==1:
        temp_dict = {key: val for key, val in dist_dict.items() if key.split("-")[0] == source and key.split("-")[1] in known_staion}
        sorted_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1]))  # sorting temp_dict based on the distance values.
        top_n = {key: sorted_dict[key] for key in list(sorted_dict)[0:num]}  # including index zero as it is the station itself for validation
    elif args.station_setting==2:
        temp_dict = {key: val for key, val in dist_dict.items() if key.split("-")[0] == source and key.split("-")[1] not in Unknown_staion}  # finding all values which start with source
        sorted_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1]))  # sorting temp_dict based on the distance values.
        top_n = {key: sorted_dict[key] for key in list(sorted_dict)[0:num]}  # excluding index zero as it is the station itself with minimun distance 0
    elif args.station_setting==3:
        temp_dict = {key: val for key, val in dist_dict.items() if key.split("-")[0] == source and key.split("-")[1] not in Unknown_staion and key.split("-")[1] not in Faulty_station}
        sorted_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1]))
        top_n = {key: sorted_dict[key] for key in list(sorted_dict)[0:num]}  # excluding index zero as it is the station itself for validation
    else:
        raise ValueError
    return top_n

def get_combinations(args):
    Known_staion = ['S1','S2','S4','S5','S6',
                    'S7','S8','S9','S11','S12',
                    'S13','S14','S16','S17',
                    'S18','S21','S22','S23']
    if args.station_setting==1:
        Faulty_station=[]
        Unknown_staion = []
    elif args.station_setting == 2:
        Faulty_station =[]
        Unknown_staion = ['S3','S10','S15','S19','S20']
    elif args.station_setting == 3:
        Faulty_station = ['S7', 'S11','S17'] # Faulty Sensors
        Unknown_staion = ['S3', 'S10', 'S15', 'S19', 'S20']
    else:
        raise ValueError
    dist_dict = read_distanace_matrix_excel("../data/koreaair/Station_distance.xlsx", sheet_name="sheet1")
    station_combination = []
    if args.mode =="train" and args.station_setting==1:
        for station1 in Known_staion:
            for station2 in Known_staion:
                station_combination.append([station1, station2])
    elif args.mode =="test" and args.station_setting==1:
        for station1 in Known_staion:
            for station2 in (find_closest(args,source=station1, dist_dict=dist_dict, known_staion=Known_staion, Unknown_staion=Unknown_staion, Faulty_station=Faulty_station,
                                          num=args.nearest_stations)):
                station_combination.append([station2.split('-')[0], station2.split('-')[1]])
    elif args.mode =="test" and args.station_setting==2:
        for station1 in Unknown_staion:
            for station2 in (find_closest(args,source=station1, dist_dict=dist_dict, known_staion=Known_staion, Unknown_staion=Unknown_staion, Faulty_station=Faulty_station,
                                          num=args.nearest_stations)):
                station_combination.append([station2.split('-')[0], station2.split('-')[1]])
    elif args.mode == "test" and args.station_setting==3:
        for station1 in Faulty_station:
            for station2 in (find_closest(args,source=station1, dist_dict=dist_dict, known_staion=Known_staion, Unknown_staion=Unknown_staion,Faulty_station=Faulty_station, num=args.nearest_stations)):
                station_combination.append([station2.split('-')[0], station2.split('-')[1]])
    else:
        print ("args.station_setting =1 can only be used in Training mode and 2,3 in test mode")
        raise ValueError
    return station_combination, dist_dict, Known_staion, Unknown_staion, Faulty_station

def MakeKoreaRailData(args, df, Training=True):
    def listoflist2np(listoflist):  # Also reshaping
        check = 0
        for l in listoflist:
            if check == 0:
                temp = np.array(l)
                check = 1
            else:
                temp = np.concatenate((temp, np.array(l)))
        temp = temp.reshape(temp.shape[0], temp.shape[1], temp.shape[2], 1)
        return temp

    def make_data(df, columns, station):
        if station[0] in columns: columns.remove(station[0])
        if station[1] in columns: columns.remove(station[1])
        newdf = df[columns]
        distance = [dist_dict[station[0] + "-" + station[1]]] * 8760 # samples in file
        newdf['Distance'] = distance
        newdf = newdf.reindex(sorted(newdf.columns), axis=1)
        checker = newdf.values
        newdf, scaler = normalize_dataset(newdf.values, 'None', True)
        return newdf, scaler,checker

    def Add_Window_Horizon_KoreaRail(X_data, Y_data, window=3, horizon=1, single=True):
        '''
        :param data: shape [B, ...]
        :param window:
        :param horizon:
        :return: X is [B, W, ...], Y is [B, H, ...]
        '''
        length = len(X_data)
        end_index = length - horizon - window + 1
        X = []  # windows
        Y = []  # horizon
        index = 0
        if single:
            while index < end_index:
                X.append(X_data[index:index + window])
                Y.append(Y_data[index + window + horizon - 1:index + window + horizon])
                index = index + 1
        else:
            while index < end_index:
                X.append(X_data[index:index + window])
                Y.append(Y_data[index + window:index + window + horizon])
                index = index + 1
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    station_combination, dist_dict, Known_staion, Unknown_staion, Faulty_station=get_combinations(args)
    if not station_combination:
        print("Test station_combination List is empty")
        raise ValueError

    ALL_X_train = []
    ALL_y_train = []
    ALL_X_test = []
    ALL_y_test = []
    for stations in station_combination:
        if args.switch_x_y == True:
            # switching X with y
            y_columns = [string for string in df.columns if stations[0] == string.split("-")[0]]
            X_columns = [string for string in df.columns if stations[1] == string.split("-")[0]]
        else:
            X_columns = [string for string in df.columns if stations[0] == string.split("-")[0]]
            y_columns = [string for string in df.columns if stations[1] == string.split("-")[0]]

        # Implement normalization if you want to use some normalization. Currently, I set the
        # normaliztion to None. So, Scaler will return the same data upon calling inverse transform
        # This is kind of a work around just to run the code without normaliztion.

        X, scaler,mxvalues = make_data(df, X_columns, stations)
        y, scaler,myvalues = make_data(df, y_columns, stations)
        if Training == True:
            X, y = Add_Window_Horizon_KoreaRail(X, y, args.lag, args.horizon, False)
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=32, shuffle=False)
            ALL_X_train.append(X_train)
            ALL_y_train.append(y_train)
        else:
            X, y = Add_Window_Horizon_KoreaRail(X, y, args.lag, args.horizon, False)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=32, shuffle=False)
            ALL_X_test.append(X_test)
            ALL_y_test.append(y_test)
    if Training == True:
        ALL_X_train = listoflist2np(ALL_X_train)
        ALL_y_train = listoflist2np(ALL_y_train)
        return ALL_X_train, ALL_y_train, scaler
    else:
        ALL_X_test = listoflist2np(ALL_X_test)
        ALL_y_test = listoflist2np(ALL_y_test)
        #
        if args.station_setting==3:
            return ALL_X_test, ALL_y_test, scaler, station_combination, dist_dict, Faulty_station
        else:
            return ALL_X_test, ALL_y_test, scaler, station_combination, dist_dict, Unknown_staion

def get_KoreaRail_dataloader(args, normalizer = 'None', tod=False, dow=False, weather=False, single=True):
    if not args.load_numpy_data_train or not args.load_numpy_data_test:
        df = pd.read_excel("../data/koreaair/GNN_dataf.xlsx")
        df = df.interpolate(method="spline", order=3, axis=0).ffill().bfill()
    train_dataloader=None
    val_dataloader = None
    test_dataloader = None
    if args.mode == "train":
        if args.load_numpy_data_train:
            print("Loading training data from saved numpy files")
            X_tra=np.load("np_data/X_tra_t+"+str(args.horizon)+".npy")
            y_tra=np.load("np_data/y_tra_t+"+str(args.horizon)+".npy")
            scaler=joblib.load("np_data/scaler_t+"+str(args.horizon)+".save")
        else:
            X_tra, y_tra, scaler = MakeKoreaRailData(args, df=df)
            np.save("np_data/X_tra_t+"+str(args.horizon)+".npy",X_tra)
            np.save("np_data/y_tra_t+"+str(args.horizon)+".npy",y_tra)
            joblib.dump(scaler,"np_data/scaler_t+"+str(args.horizon)+".save")
        train_dataloader = data_loader(X_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if args.load_numpy_data_test:
        print("Loading test data from saved numpy files")
        X_test = np.load("np_data/X_test_t+"+str(args.horizon)+".npy")
        y_test = np.load("np_data/y_test_t+"+str(args.horizon)+".npy")
        scaler = joblib.load("np_data/scaler_t+" + str(args.horizon) + ".save")
        test_combinations=None
        distance_dict=None
        Unknown_station=None
    else:
        X_test, y_test,scaler, test_combinations,distance_dict, Unknown_station = MakeKoreaRailData(args,df=df,Training=False)
        np.save("np_data/X_test_t+"+str(args.horizon)+".npy",X_test)
        np.save("np_data/y_test_t+"+str(args.horizon)+".npy",y_test)

    ##############get dataloader######################
    test_dataloader = data_loader(X_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler, test_combinations,distance_dict,Unknown_station

