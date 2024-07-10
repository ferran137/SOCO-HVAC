import pandas as pd
from datetime import datetime, timedelta
import holidays
import joblib
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import math
import pytz

# todo: Per fer funcionar els dos models s'han d'assumir 3 suposicions:
#  1.- Coneixem el número de persones dins la sala durant tot el temps predit
#  2.- La temperatura de la sala de màquines és una constant de 23ºC
#  3.- La temperatura a l'equilibri quan fancoil i diposit estan obert segueix una corba exponencial
#  (s'obté al fitxer proves_temp_dip.py)


def convert_to_spain_time(utc_time):
    utc_zone = pytz.utc
    spain_zone = pytz.timezone('Europe/Madrid')
    utc_time = utc_zone.localize(utc_time)
    return pd.to_datetime(utc_time.astimezone(spain_zone).strftime('%Y-%m-%d %H:%M:%S'))


def tendencia_lab(lab1, lab2, dip1, exte, consigna_lab, threshold):
    if lab1 > consigna_lab + threshold:
        return exte
    elif consigna_lab-threshold <= lab1 <= consigna_lab + threshold:
        pendent_recta = lab1-lab2
        if pendent_recta >= 0:
            return dip1
        elif pendent_recta < 0:
            return exte
    elif lab1 < consigna_lab - threshold:
        return dip1


def tend_exp(consigna_dip):
    """
    Funció que determina cap a on ha de tendir la temperatura del dipòsit quan fancoil esta obert

    :param consigna_dip: Consigna diposit
    :return: Temperatura on tendeix el diposit
    """
    a = 0.004765673343956606
    b = 0.19260695180765264
    c = -2.3067216941481656
    d = 0.38350692545927056

    f = a * np.exp(b*consigna_dip+c) + d
    return f


def tendencia_dip(dip1, dip2, exte, tendencia_labo, consigna_dip, threshold_sup, threshold_inf):
    if tendencia_labo == exte:
        if dip1 > consigna_dip+threshold_sup:
            return 23  # temperatura sala de maquines. asumim temp constant a 23 graus
        elif consigna_dip-threshold_inf <= dip1 <= consigna_dip+threshold_sup:
            pendent_diposit = dip1-dip2
            if pendent_diposit >= 0:
                return consigna_dip+threshold_sup
            elif pendent_diposit < 0:
                return consigna_dip-threshold_inf
        elif dip1 < consigna_dip-threshold_inf:
            return consigna_dip+threshold_sup

    elif tendencia_labo == dip1:
        return consigna_dip - tend_exp(consigna_dip)


def get_data_train(df_name):
    df = pd.read_excel(f"Dades/consum/{df_name}")

    """
    df = df.dropna().reset_index(drop=True)

    # Columna de les dates no té nom, li posem un
    df.columns.values[0] = "Date"
    # Li sumem una hora pq esta en hora utf
    df["Date"] = df["Date"].apply(convert_to_spain_time)
    # Definim dies de la setmana festius i caps de setmana com a 0, la resta de dies de la setmana com a 1,2,3,4,5
    unique_years = {timestamp.year for timestamp in df["Date"]}
    festius_cat = holidays.Spain(prov='CT', years=unique_years)
    dia_festiu = [1 if day in festius_cat else 0 for day in df["Date"]]
    days = [day.weekday() + 1 if fest == 0 else 0 for day, fest in zip(df["Date"], dia_festiu)]
    days = [0 if day > 5 else day for day in days]
    hour = [day.hour + day.minute / 60 for day in df["Date"]]
    df["Day"] = days
    df["Hour"] = hour

    # CONSIGNA LAB
    # per la consigna quan el sistema esta ON (switch=1) definim consigna establerta manualment,
    # quan esta OFF (switch=0) definim la consigna com a la temperatura exterior
    consigna_lab = []
    for hour, index_dumy in zip(df["Hour"], df.index):
        if df.loc[index_dumy, "Consigna Laboratori Hivern Balanced"] > 0:
            if hour < 5:
                consigna_lab.append(20)
            if 5 <= hour <= 17:
                consigna_lab.append(24)
            if hour > 17:
                consigna_lab.append(20)
        else:
            consigna_lab.append(df.loc[index_dumy, "geotermia temperatura_exterior"])
    df["consigna lab"] = consigna_lab

    switch_fancoil_arreglat = [1 if i > 0 else 0 for i in df["Shelly Plus 1pm Fancoil switch_0"]]
    df["Shelly Plus 1pm Fancoil switch_0"] = switch_fancoil_arreglat

    switch_dip = [1 if i > 0 else 0 for i in df["geotermia potncia_bomba_geotrmia"]]
    df["Switch bomba diposit"] = switch_dip

    # Numero Persones
    numero_persones = [math.ceil(i) for i in df["counter.numero_persones"]]
    numero_persones = [i if i <= 11 else 11 for i in numero_persones]
    df["counter.numero_persones"] = numero_persones
    """

    data_x_dip = []
    data_y_dip = []

    for line in range(2, len(df)):

        delta_t1 = (df.loc[line, "Date"] - df.loc[line - 1, "Date"]).total_seconds() / 60
        delta_t2 = (df.loc[line, "Date"] - df.loc[line - 2, "Date"]).total_seconds() / 60

        if delta_t1 == 5 and delta_t2 == 10:

            tendencia_temperatura_lab = tendencia_lab(df.loc[line - 1, "Shelly Plus HT Temperature"],
                                                      df.loc[line - 2, "Shelly Plus HT Temperature"],
                                                      df.loc[line-1, "geotermia temperatura_diposit"],
                                                      df.loc[line, "geotermia temperatura_exterior"],
                                                      df.loc[line, "consigna lab"],
                                                      0.2)

            temporary_data_x_dip = []
            temp_dip_anterior = df.loc[line - 1, "geotermia temperatura_diposit"]
            temporary_data_x_dip.append(temp_dip_anterior)
            temp_dip_anterior2 = df.loc[line - 2, "geotermia temperatura_diposit"]
            temporary_data_x_dip.append(temp_dip_anterior2)

            tendencia_diposit = tendencia_dip(df.loc[line - 1, "geotermia temperatura_diposit"],
                                              df.loc[line - 2, "geotermia temperatura_diposit"],
                                              df.loc[line, "geotermia temperatura_exterior"],
                                              tendencia_temperatura_lab,
                                              df.loc[line, "geotermia temperatura_consigna_diposit_hivern"],
                                              0.7,
                                              1)
            temporary_data_x_dip.append(tendencia_diposit)
            data_x_dip.append(temporary_data_x_dip)

            temporary_data_y_dip = df.loc[line, "geotermia temperatura_diposit"]
            data_y_dip.append(temporary_data_y_dip)

        else:
            continue

    return data_x_dip, data_y_dip


def model_svr(data_x, data_y, kernel, C, epsilon, degree, gamma):

    regressor = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, gamma=gamma, cache_size=2000)

    c_str = str(C).replace('.', '')
    epsilon_str = str(epsilon).replace('.', '')
    gamma_str = str(gamma).replace('.', '')
    if kernel != "poly":
        print(f"Training SVR kernel:{kernel}, C:{C}, epsilon:{epsilon}, gamma:{gamma}")
        name = "svr_dip_" + kernel + "_" + c_str + "_" + epsilon_str + "_" + gamma_str
    else:
        print(f"Training SVR kernel:{kernel}, degree:{degree}, C:{C}, epsilon:{epsilon}, gamma:{gamma}")
        name = "svr_dip_" + kernel + "_" + str(degree) + "_" + c_str + "_" + epsilon_str + "_" + gamma_str

    print(f"Rows to process: {len(data_x)}")
    start_time = datetime.now()
    regressor.fit(data_x, data_y)
    end_time = datetime.now()
    time_dif = (end_time - start_time).total_seconds() / 60
    print(f"Training done. {time_dif} minutes")
    joblib.dump(regressor, "Models/svr/" + name + ".joblib")


def test_svr(df_name, date_of_start, steps, kernel, C, epsilon, degree, gamma):
    df = pd.read_excel(f"Dades/consum/{df_name}")

    """
    df = df.dropna().reset_index(drop=True)

    # Columna de les dates no té nom, li posem un
    df.columns.values[0] = "Date"
    # Li sumem una hora pq esta en hora utf
    df["Date"] = df["Date"].apply(convert_to_spain_time)
    # Definim dies de la setmana festius i caps de setmana com a 0, la resta de dies de la setmana com a 1,2,3,4,5
    unique_years = {timestamp.year for timestamp in df["Date"]}
    festius_cat = holidays.Spain(prov='CT', years=unique_years)
    dia_festiu = [1 if day in festius_cat else 0 for day in df["Date"]]
    days = [day.weekday() + 1 if fest == 0 else 0 for day, fest in zip(df["Date"], dia_festiu)]
    days = [0 if day > 5 else day for day in days]
    hour = [day.hour + day.minute / 60 for day in df["Date"]]
    df["Day"] = days
    df["Hour"] = hour

    # CONSIGNA LAB
    # per la consigna quan el sistema esta ON (switch=1) definim consigna asignada, quan esta OFF (switch=0) definim
    # la consigna com a la temperatura exterior
    consigna_lab = []
    for hour, index_dumy in zip(df["Hour"], df.index):
        if df.loc[index_dumy, "Consigna Laboratori Hivern Balanced"] > 0:
            if hour < 5:
                consigna_lab.append(20)
            if 5 <= hour <= 17:
                consigna_lab.append(24)
            if hour > 17:
                consigna_lab.append(20)
        else:
            consigna_lab.append(df.loc[index_dumy, "geotermia temperatura_exterior"])

    df["consigna lab"] = consigna_lab

    switch_fancoil_arreglat = [1 if i > 0 else 0 for i in df["Shelly Plus 1pm Fancoil switch_0"]]
    df["Shelly Plus 1pm Fancoil switch_0"] = switch_fancoil_arreglat

    switch_dip = [1 if i > 0 else 0 for i in df["geotermia potncia_bomba_geotrmia"]]
    df["Switch bomba diposit"] = switch_dip

    # Numero Persones
    numero_persones = [math.ceil(i) for i in df["counter.numero_persones"]]
    numero_persones = [i if i <= 11 else 11 for i in numero_persones]
    df["counter.numero_persones"] = numero_persones
    """

    # x_test = df.loc[df["Date"] == date_of_start].copy()
    index = df.loc[df["Date"] == date_of_start].index[0]

    if index < 1:
        raise TypeError("Start date does not have a previous row to extract the previous Temperature")

    new_temp_dip_ant = df.loc[index - 1, "geotermia temperatura_diposit"]
    new_temp_dip_ant2 = df.loc[index - 2, "geotermia temperatura_diposit"]
    new_temp_lab_ant = df.loc[index - 1, "Shelly Plus HT Temperature"]
    new_temp_lab_ant2 = df.loc[index - 2, "Shelly Plus HT Temperature"]
    new_temp_exte = df.loc[index, "geotermia temperatura_exterior"]
    new_consigna_lab = df.loc[index, "consigna lab"]
    new_consigna_dip = df.loc[index, "geotermia temperatura_consigna_diposit_hivern"]

    new_tend_lab = tendencia_lab(new_temp_lab_ant,
                                 new_temp_lab_ant2,
                                 new_temp_dip_ant,
                                 new_temp_exte,
                                 new_consigna_lab,
                                 0.2)

    new_tend_dip = tendencia_dip(new_temp_dip_ant,
                                 new_temp_dip_ant2,
                                 new_temp_exte,
                                 new_tend_lab,
                                 new_consigna_dip,
                                 0.7,
                                 1)

    temporary_x_dip = [
        new_temp_dip_ant,
        new_temp_dip_ant2,
        new_tend_dip,
    ]

    c_str = str(C).replace('.', '')
    epsilon_str = str(epsilon).replace('.', '')
    gamma_str = str(gamma).replace('.', '')
    if kernel != "poly":
        name = "svr_dip_" + kernel + "_" + c_str + "_" + epsilon_str + "_" + gamma_str
    else:
        name = "svr_dip_" + kernel + "_" + str(degree) + "_" + c_str + "_" + epsilon_str + "_" + gamma_str
    model = joblib.load(f"Models/svr/{name}.joblib")

    prediction = model.predict(np.array(temporary_x_dip).reshape(1, -1))[0]

    y_predict_dip = [prediction]
    y_test_dip = [df.loc[index, "geotermia temperatura_diposit"]]
    consignes = [df.loc[index, "geotermia temperatura_consigna_diposit_hivern"]]

    x_test_date = [df.loc[index, "Date"]]

    for i in range(1, steps + 1):
        print(f"Step {i}")

        new_temp_dip_ant2 = new_temp_dip_ant
        new_temp_dip_ant = prediction

        new_temp_lab_ant = df.loc[index + i - 1, "Shelly Plus HT Temperature"]
        new_temp_lab_ant2 = df.loc[index + i - 2, "Shelly Plus HT Temperature"]
        new_temp_exte = df.loc[index + i, "geotermia temperatura_exterior"]
        new_consigna_lab = df.loc[index + i, "consigna lab"]
        new_consigna_dip = df.loc[index + i, "geotermia temperatura_consigna_diposit_hivern"]

        new_tend_lab = tendencia_lab(new_temp_lab_ant,
                                     new_temp_lab_ant2,
                                     new_temp_dip_ant,
                                     new_temp_exte,
                                     new_consigna_lab,
                                     0.2)

        new_tend_dip = tendencia_dip(new_temp_dip_ant,
                                     new_temp_dip_ant2,
                                     new_temp_exte,
                                     new_tend_lab,
                                     new_consigna_dip,
                                     0.8,
                                     1)

        temporary_x_dip = [
            new_temp_dip_ant,
            new_temp_dip_ant2,
            new_tend_dip
        ]

        prediction = model.predict(np.array(temporary_x_dip).reshape(1, -1))[0]
        y_predict_dip.append(prediction)

        y_test_dip.append(df.loc[index + i, "geotermia temperatura_diposit"])
        consignes.append(df.loc[index + i, "geotermia temperatura_consigna_diposit_hivern"])
        x_test_date.append(df.loc[index + i, "Date"])

    r2 = r2_score(y_test_dip, y_predict_dip)

    err_pre_mape = []
    true_mae = []

    for i in range(0, len(y_predict_dip)):

        error_mae = np.abs(y_test_dip[i] - y_predict_dip[i])
        true_mae.append(error_mae)

        err = np.abs((y_test_dip[i] - y_predict_dip[i]) / y_test_dip[i])
        err_pre_mape.append(err * 100)

    mae_final = np.mean(true_mae)
    mape = np.mean(err_pre_mape)

    print(f"R2 score: {r2}")
    print(f"MAPE score: {mape}")
    print(f"MAE score: {mae_final}")

    # x_test_date = df.loc[index: index+steps, "Date"]

    plt.plot(x_test_date, y_test_dip, label="Real data")
    plt.plot(x_test_date, y_predict_dip, label="Prediction")
    plt.plot(x_test_date, consignes, label="Instruction")

    plt.title(f"Temperature inertia tank forecast ($R^2$={round(r2, 2)}, MAPE% = {round(mape, 2)} %)")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Temperature (ºC)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plots/svr/{name}_steps_{steps}.jpg")
    plt.close()
    plt.show()

    return r2, mape


##################
# Prova senzilla #
##################

steps0 = 284
ker0 = "rbf"
C0 = 50
eps0 = 0.05
gamm0 = "auto"
pol0 = 1

# [50, 'rbf', 0.05, 'auto', 1.310607542989364, 0.9251119709683806]

ini_date = pd.to_datetime("2024-02-22T00:00:00") + pd.Timedelta(minutes=15)

test_svr("noves_alternades/dades_test_tractades.xlsx", ini_date, steps0, ker0, C0, eps0, pol0, gamm0)


##################


##################
"""
kernel_list = ["rbf"]  # ["rbf", "linear", "sigmoid"]
C_list = [0.1, 0.5, 1, 5, 10, 25, 50, 100]  # fer a casa els C=10,100. original [0.001, 0.01, 0.1, 1, 10, 100]
epsilon_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]  # original [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
gamma_list = ["auto", "scale"]  # original ["scale", "auto"]
poly_list = [1]

df_name0 = "noves_alternades/dades_train_tractades.xlsx"
date_start0 = "2024-02-18T00:00:00"

datax0, datay0 = get_data_train(df_name0)

r2_scores = []
for ker in kernel_list:
    for c in C_list:
        for eps in epsilon_list:
            if ker != "linear":
                for gamm in gamma_list:
                    for pol in poly_list:
                        model_svr(datax0, datay0, ker, c, eps, pol, gamm)
                        r20, mape0 = test_svr(df_name0, date_start0, 280, ker, c, eps, pol, gamm)
                        r2vec = [c, ker, eps, gamm, mape0, r20]
                        r2_scores.append(r2vec)
            elif ker == "linear":
                model_svr(datax0, datay0, ker, c, eps, 0, "auto")
                r20, mape0 = test_svr(df_name0, date_start0, 280, ker, c, eps, 0, "auto")
                r2vec = [c, ker, eps, "nogam", mape0, r20]
                r2_scores.append(r2vec)

vector_ordenat = sorted(r2_scores, key=lambda x: x[-1], reverse=True)

print("Vector ordenat per la component r2:", vector_ordenat)


# todo: Pressió estandard del dipòsit a 2 bars. Quan augmenta la temperatura l'aigua s'expandeix i la pressió augmenta
#   llavors s'ha de buidar aigua del diposit perquè disminueixi la pressió i en consequencia el diposit no es podrà
#   escalfar tant com abans  o directament no s'escalfarà.
#   a recordar: si s'escalfa molt el diposit la pressió augmenta i dels 250l s'ha de buidar.
#   Lo ideal seria tenir un regulador que si la pressió fos massa alta obris una aixeta per deixar sortir aigua i quan
#   la pressió fos massa baixa obrir una aixeta per deixar entrar aigua i aixi sempre tenir els mateixos litres d'aigua.
#   o simplement tenir un regulador.

"""
################################
"""
r2_scores = []

new_ker_lst = ['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf']


new_c_lst = [0.5, 25, 50, 0.5, 50, 1, 25, 0.5, 50, 100, 25, 10, 1, 10, 10, 100, 50, 25, 100, 100, 50, 10, 0.5, 5,
             100, 1, 25, 1, 10, 5, 5, 100, 0.5, 25, 50, 25, 10, 5, 5, 1, 0.1, 5, 0.5, 1, 0.1, 0.1, 5, 10, 1, 0.5,
             5, 1, 10, 50, 0.1, 50, 100, 25, 0.5, 100, 0.1, 0.1, 0.1, 50, 100, 0.5, 0.5, 0.1, 1, 0.1, 5, 25, 50,
             100, 10, 1, 5, 10, 25, 50, 100, 1]


new_eps_lst = [0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.05, 0.1, 0.1, 0.1,
               0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05, 0.1, 0.05, 0.01, 0.05, 0.5, 0.01, 0.01,
               0.01, 0.01, 0.01, 0.01, 0.1, 0.05, 0.01, 0.05, 0.05, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1,
               0.5, 1, 0.5, 0.5, 0.5, 0.01, 0.05, 5, 1, 0.5, 5, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 5]


new_gamma_lst = ['auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto',
                 'auto', 'auto', 'auto', 'scale', 'scale', 'scale', 'auto', 'scale', 'scale', 'scale', 'scale',
                 'scale', 'auto', 'scale', 'scale', 'scale', 'scale', 'auto', 'scale', 'scale', 'scale', 'scale',
                 'scale', 'scale', 'scale', 'auto', 'scale', 'scale', 'scale', 'auto', 'scale', 'auto', 'scale',
                 'scale', 'scale', 'scale', 'scale', 'scale', 'auto', 'auto', 'auto', 'auto', 'auto', 'scale',
                 'scale', 'auto', 'auto', 'auto', 'auto', 'auto', 'scale', 'scale', 'scale', 'scale', 'auto',
                 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'scale', 'scale', 'scale', 'scale',
                 'scale', 'scale', 'scale']


df_name0 = "noves_alternades/dades_test_tractades.xlsx"

df = pd.read_excel(f"Dades/consum/{df_name0}")


lst_of_dates = []
for row in df.index[2:len(df)-287]:
    if df.loc[row, "Hour"] == 0:
        lst_of_dates.append(df.loc[row, "Date"])


lists_of_lists = [[] for i in range(0, len(lst_of_dates))]

for date, lst in zip(lst_of_dates, lists_of_lists):
    for nK, nC, nE, nG in zip(new_ker_lst, new_c_lst, new_eps_lst, new_gamma_lst):
        r20, mape0 = test_svr(df_name0, date + pd.Timedelta(minutes=15), 284, nK, nC, nE, 0, nG)
        r2vec = [nC, nK, nE, nG, mape0, r20]
        lst.append(r2vec)


models_average = []
for model in range(0, len(new_ker_lst)):
    r2_sum = 0
    mape_sum = 0
    model_vector = lists_of_lists[0][model][:-2]
    for listt in lists_of_lists:
        r2_sum += listt[model][-1]
        mape_sum += listt[model][-2]
    r2_average = r2_sum/len(lists_of_lists)
    mape_average = mape_sum/len(lists_of_lists)
    model_vector.append(mape_average)
    model_vector.append(r2_average)
    models_average.append(model_vector)

sorted_models_by_r2 = sorted(models_average, key=lambda v: v[-1], reverse=True)
sorted_models_by_mape = sorted(models_average, key=lambda v: v[-2], reverse=False)

print(f"Sorted models by average r2: {sorted_models_by_r2}")
print(f"Sorted models by average mape: {sorted_models_by_mape}")
"""

# Sorted models by average r2: [[50, 'rbf', 0.05, 'auto', 1.310607542989364, 0.9251119709683806], [10, 'rbf', 0.05, 'auto', 1.436501409253156, 0.9154112066729818], [10, 'rbf', 0.1, 'auto', 1.5463890796866933, 0.902643379049884], [5, 'rbf', 0.1, 'auto', 1.4590946521943355, 0.9025462827560322], [100, 'rbf', 0.05, 'auto', 1.3674940106155216, 0.9020159438993627], [50, 'rbf', 0.1, 'auto', 1.4341554534187315, 0.8961777488312549], [50, 'rbf', 0.01, 'auto', 1.4591118722925267, 0.8928708900799865], [25, 'rbf', 0.01, 'auto', 1.4877234405937327, 0.8793607352537901], [25, 'rbf', 0.05, 'auto', 1.516609541379635, 0.8689533538825032], [10, 'rbf', 0.01, 'auto', 1.684352836115948, 0.8443134905958255], [5, 'rbf', 0.01, 'auto', 1.6791789316422538, 0.829479511479191], [0.5, 'rbf', 0.1, 'auto', 1.8316369999966096, 0.8195196103662727], [0.5, 'rbf', 0.01, 'auto', 1.8882371737211385, 0.8120418101889355], [0.5, 'rbf', 0.05, 'auto', 1.9838027495317676, 0.8118519535997308], [25, 'rbf', 0.1, 'auto', 1.836902932564719, 0.8117242876323276], [1, 'rbf', 0.1, 'auto', 1.8683660235112725, 0.8049199858556308], [1, 'rbf', 0.01, 'auto', 1.9606516978626107, 0.8041155008706264], [5, 'rbf', 0.05, 'auto', 1.92786819460104, 0.7972297465624342], [100, 'rbf', 0.1, 'auto', 1.9842063948979174, 0.7797513737601794], [1, 'rbf', 0.05, 'auto', 2.037762666908334, 0.7741286151417891], [0.1, 'rbf', 0.01, 'scale', 2.2308058449375037, 0.7727432401199554], [5, 'rbf', 0.05, 'scale', 2.136701934996852, 0.772687270019009], [0.1, 'rbf', 0.05, 'scale', 2.2218072945717946, 0.7722424231453195], [10, 'rbf', 0.05, 'scale', 2.191023759834978, 0.7716221125416851], [0.5, 'rbf', 0.01, 'scale', 2.2043042246928195, 0.7679581514337538], [0.1, 'rbf', 0.1, 'scale', 2.3792411108171097, 0.7658485048284266], [0.5, 'rbf', 0.05, 'scale', 2.2004824201846938, 0.7622854562824448], [1, 'rbf', 0.01, 'scale', 2.2101525955807206, 0.7573472170347302], [1, 'rbf', 0.05, 'scale', 2.1621459380208456, 0.757173422008151], [25, 'rbf', 0.5, 'scale', 2.7093710385199397, 0.7538320778712546], [25, 'rbf', 0.1, 'scale', 2.311335976750945, 0.7514809929898263], [10, 'rbf', 0.1, 'scale', 2.3039006042941157, 0.7497414459303026], [5, 'rbf', 0.1, 'scale', 2.3061030051786044, 0.7489257742045164], [100, 'rbf', 0.01, 'auto', 2.007838373985238, 0.7429203811839384], [0.5, 'rbf', 0.1, 'scale', 2.4498737964184496, 0.739071467199297], [1, 'rbf', 0.1, 'scale', 2.417852238212821, 0.72959754525649], [10, 'rbf', 0.01, 'scale', 2.5104322614307666, 0.7294441769597857], [5, 'rbf', 0.01, 'scale', 2.5594913452245462, 0.7285328993667227], [100, 'rbf', 0.01, 'scale', 2.524605956106972, 0.7262986080835485], [25, 'rbf', 0.01, 'scale', 2.537197492215328, 0.7237432380298041], [25, 'rbf', 0.05, 'scale', 2.610091971172901, 0.7213566527495264], [50, 'rbf', 0.01, 'scale', 2.5638903849352306, 0.7189539929767229], [100, 'rbf', 0.05, 'scale', 2.7478355577026448, 0.707693364269233], [50, 'rbf', 0.05, 'scale', 2.7717168879379837, 0.7033681804687255], [1, 'rbf', 0.5, 'scale', 3.3581216684414676, 0.6767000922787232], [0.5, 'rbf', 0.5, 'scale', 3.451532656590829, 0.661669554328386], [100, 'rbf', 0.1, 'scale', 3.125700339386199, 0.6438357299275543], [50, 'rbf', 0.1, 'scale', 3.0927619360519096, 0.635074031355771], [50, 'rbf', 0.5, 'scale', 3.9083183516773836, 0.5298848224293381], [1, 'rbf', 0.5, 'auto', 3.923638936905585, 0.5179755431497718], [10, 'rbf', 0.5, 'auto', 4.03181475538412, 0.50742767417742], [5, 'rbf', 0.5, 'auto', 4.082076363268186, 0.5009426830478085], [50, 'rbf', 0.5, 'auto', 4.176304671344799, 0.4711364174826274], [25, 'rbf', 0.5, 'auto', 4.238744232745222, 0.46245035058653816], [5, 'rbf', 0.5, 'scale', 4.458576360196621, 0.46120660841219696], [10, 'rbf', 0.5, 'scale', 4.471235960359309, 0.4610203310825698], [100, 'rbf', 1, 'scale', 3.839035779037397, 0.4588729661576444], [0.5, 'rbf', 0.5, 'auto', 4.152825440157526, 0.4449924707428912], [100, 'rbf', 0.5, 'auto', 4.403924557300476, 0.4310187311028117], [100, 'rbf', 0.5, 'scale', 4.3024364717692425, 0.4306497642949002], [50, 'rbf', 1, 'scale', 4.095785337252618, 0.41295700506930555], [0.5, 'rbf', 5, 'scale', 4.952531750257093, 0.28206579939709625], [0.1, 'rbf', 5, 'scale', 4.639628837423169, 0.2653720604100346], [0.5, 'rbf', 1, 'auto', 4.6419318418922995, 0.22908066581956188], [5, 'rbf', 1, 'auto', 5.19857175436516, 0.2281059323304282], [1, 'rbf', 1, 'auto', 5.161295317405424, 0.21987687335832895], [25, 'rbf', 1, 'auto', 5.261111384185037, 0.2085646801208092], [50, 'rbf', 1, 'auto', 5.267954862657914, 0.20629684721236946], [100, 'rbf', 1, 'auto', 5.270537924596603, 0.20533485726526202], [0.1, 'rbf', 0.5, 'auto', 4.649647985713183, 0.2029563192569946], [0.1, 'rbf', 0.1, 'auto', 4.800414117745671, 0.2009527661809923], [10, 'rbf', 1, 'auto', 5.296971527767505, 0.19697210883322164], [1, 'rbf', 5, 'scale', 5.709145750764738, 0.1841031102824596], [0.1, 'rbf', 1, 'auto', 4.823212431532317, 0.11628899870292327], [0.1, 'rbf', 0.01, 'auto', 5.621850968167374, 0.09557113270164055], [0.1, 'rbf', 0.05, 'auto', 5.639373821089775, 0.0746010163932554], [1, 'rbf', 10, 'scale', 6.383046155514219, -0.12927089607885076], [5, 'rbf', 10, 'scale', 6.411794652420265, -0.13679778209349755], [10, 'rbf', 10, 'scale', 6.411794652420265, -0.13679778209349755], [25, 'rbf', 10, 'scale', 6.411794652420265, -0.13679778209349755], [50, 'rbf', 10, 'scale', 6.411794652420265, -0.13679778209349755], [100, 'rbf', 10, 'scale', 6.411794652420265, -0.13679778209349755]]