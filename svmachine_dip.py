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
