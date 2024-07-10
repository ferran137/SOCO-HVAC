import pandas as pd
from datetime import datetime
import holidays
import joblib
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import math
from sklearn.preprocessing import StandardScaler
import pytz


def convert_to_spain_time(utc_time):
    utc_zone = pytz.utc
    spain_zone = pytz.timezone('Europe/Madrid')
    utc_time = utc_zone.localize(utc_time)
    return pd.to_datetime(utc_time.astimezone(spain_zone).strftime('%Y-%m-%d %H:%M:%S'))


def tendencia_lab(lab1, lab2, dip1, exte, consigna, threshold):
    if lab1 > consigna + threshold:
        return exte
    elif consigna-threshold <= lab1 <= consigna + threshold:
        pendent_recta = lab1-lab2
        if pendent_recta >= 0:
            return dip1
        elif pendent_recta < 0:
            return exte
    elif lab1 < consigna - threshold:
        return dip1


def get_data_train(df_name):

    df = pd.read_excel(f"Dades/consum/{df_name}")
    """
    df = df.dropna().reset_index(drop=True)

    # Columna de les dates no té nom, li posem un
    df.columns.values[0] = "Date"
    # Li sumem una hora pq esta en hora utf
    # new_date_actualized = [date + timedelta(hours=1) for date in df["Date"]]
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

    # df = norm_std_data_set(df)
    """
    data_x_lab = []
    data_y_lab = []

    for line in range(2, len(df)):

        delta_t1 = (df.loc[line, "Date"] - df.loc[line - 1, "Date"]).total_seconds() / 60
        delta_t2 = (df.loc[line, "Date"] - df.loc[line - 2, "Date"]).total_seconds() / 60

        if delta_t1 == 5 and delta_t2 == 10:
            # Per la temperatura lab farem servir els inputs:
            #   - Temperatura lab (t-1)
            #   - Tendencia lab
            #   - Dia setmana
            temporary_data_x_lab = []
            temp_lab_anterior = df.loc[line - 1, "Shelly Plus HT Temperature"]
            temporary_data_x_lab.append(temp_lab_anterior)
            temp_lab_anterior2 = df.loc[line - 2, "Shelly Plus HT Temperature"]
            temporary_data_x_lab.append(temp_lab_anterior2)
            tendencia_temperatura_lab = tendencia_lab(temp_lab_anterior, temp_lab_anterior2,
                                                      df.loc[line-1, "geotermia temperatura_diposit"],
                                                      df.loc[line, "geotermia temperatura_exterior"],
                                                      df.loc[line, "consigna lab"], 0.2)
            temporary_data_x_lab.append(tendencia_temperatura_lab)

            temp_exte = df.loc[line, "geotermia temperatura_exterior"]
            temporary_data_x_lab.append(temp_exte)

            num_persones = df.loc[line, "smooth_pers"]  # df.loc[line, "counter.numero_persones"]
            temporary_data_x_lab.append(num_persones)

            data_x_lab.append(temporary_data_x_lab)

            temporary_data_y_lab = df.loc[line, "Shelly Plus HT Temperature"]
            data_y_lab.append(temporary_data_y_lab)

        else:
            continue

    return data_x_lab, data_y_lab


def model_svr(data_x, data_y, kernel, C, epsilon, degree, gamma):

    regressor = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, gamma=gamma, cache_size=2000)
    # cache size: default 200, passar a 500 o 1000

    c_str = str(C).replace('.', '')
    epsilon_str = str(epsilon).replace('.', '')
    gamma_str = str(gamma).replace('.', '')
    if kernel != "poly":
        print(f"Training SVR kernel:{kernel}, C:{C}, epsilon:{epsilon}, gamma:{gamma}")
        name = "svr_lab_" + kernel + "_" + c_str + "_" + epsilon_str + "_" + gamma_str
    else:
        print(f"Training SVR kernel:{kernel}, degree:{degree}, C:{C}, epsilon:{epsilon}, gamma:{gamma}")
        name = "svr_lab_" + kernel + "_" + str(degree) + "_" + c_str + "_" + epsilon_str + "_" + gamma_str

    print(f"Rows to process: {len(data_x)}")
    start_time = datetime.now()
    regressor.fit(data_x, data_y)
    end_time = datetime.now()
    time_dif = (end_time - start_time).total_seconds() / 60
    print(f"Training done. {time_dif} minutes")
    joblib.dump(regressor, "Models/svr/" + name + ".joblib")


def test_svr(df_name, date_of_start, steps, kernel, C, epsilon, degree, gamma):
    df = pd.read_excel(f"Dades/consum/{df_name}")

    # x_test = df.loc[df["Date"] == date_of_start].copy()
    index = df.loc[df["Date"] == date_of_start].index[0]

    if index < 1:
        raise TypeError("Start date does not have a previous row to extract the previous Temperature")

    new_temp_lab_ant = df.loc[index - 1, "Shelly Plus HT Temperature"]
    new_temp_lab_ant2 = df.loc[index - 2, "Shelly Plus HT Temperature"]
    new_temp_dip_ant = df.loc[index - 1, "geotermia temperatura_diposit"]
    new_consigna_lab = df.loc[index, "consigna lab"]
    new_temp_exte = df.loc[index, "geotermia temperatura_exterior"]
    new_tend_lab = tendencia_lab(new_temp_lab_ant, new_temp_lab_ant2, new_temp_dip_ant, new_temp_exte,
                                 new_consigna_lab, 0.2)
    new_num_persones = df.loc[index, "smooth_pers"]  # df.loc[index, "counter.numero_persones"]
    #new_day = df.loc[index, "Day"]

    # Per la temperatura lab farem servir els inputs:
    #   - Temperatura lab (t-1)
    #   - Tendencia lab
    # Per la temperatura diposit farem servir:
    #   - Temperatura diposit (t-1)
    #   - Tendencia diposit
    temporary_x_lab = [
        new_temp_lab_ant,
        new_temp_lab_ant2,
        new_tend_lab,
        new_temp_exte,
        new_num_persones
    ]

    c_str = str(C).replace('.', '')
    epsilon_str = str(epsilon).replace('.', '')
    gamma_str = str(gamma).replace('.', '')
    if kernel != "poly":
        name = "svr_lab_" + kernel + "_" + c_str + "_" + epsilon_str + "_" + gamma_str
    else:
        name = "svr_lab_" + kernel + "_" + str(degree) + "_" + c_str + "_" + epsilon_str + "_" + gamma_str
    model = joblib.load(f"Models/svr/{name}.joblib")

    prediction_lab = model.predict(np.array(temporary_x_lab).reshape(1, -1))[0]

    y_predict_lab = [prediction_lab]
    y_test_lab = [df.loc[index, "Shelly Plus HT Temperature"]]
    consignes = [df.loc[index, "consigna lab"]]
    numero_persones_lab = [new_num_persones]

    x_test_date = [df.loc[index, "Date"]]

    for i in range(1, steps + 1):
        print(f"Step {i}")

        new_temp_lab_ant2 = new_temp_lab_ant
        new_temp_lab_ant = prediction_lab
        new_temp_dip_ant = df.loc[index + i - 1, "geotermia temperatura_diposit"]
        new_consigna_lab = df.loc[index + i, "consigna lab"]
        new_temp_exte = df.loc[index + i, "geotermia temperatura_exterior"]
        new_tend_lab = tendencia_lab(new_temp_lab_ant, new_temp_lab_ant2, new_temp_dip_ant, new_temp_exte,
                                     new_consigna_lab, 0.2)
        new_num_persones = df.loc[index + i, "smooth_pers"]  # df.loc[index + i, "counter.numero_persones"]

        temporary_x_lab = [
            new_temp_lab_ant,
            new_temp_lab_ant2,
            new_tend_lab,
            new_temp_exte,
            new_num_persones
        ]

        prediction_lab = model.predict(np.array(temporary_x_lab).reshape(1, -1))[0]
        y_predict_lab.append(prediction_lab)

        y_test_lab.append(df.loc[index + i, "Shelly Plus HT Temperature"])
        consignes.append(df.loc[index + i, "consigna lab"])
        x_test_date.append(df.loc[index + i, "Date"])
        numero_persones_lab.append(new_num_persones)

    r2 = r2_score(y_test_lab, y_predict_lab)

    err_pre_mape = []
    true_mae = []

    for i in range(0, len(y_predict_lab)):

        error_mae = np.abs(y_test_lab[i] - y_predict_lab[i])
        true_mae.append(error_mae)

        err = np.abs((y_test_lab[i] - y_predict_lab[i]) / y_test_lab[i])
        err_pre_mape.append(err * 100)

    mae_final = np.mean(true_mae)
    mape = np.mean(err_pre_mape)

    print(f"R2 score: {r2}")
    print(f"MAPE score: {mape}")
    print(f"MAE score: {mae_final}")

    # x_test_date = df.loc[index: index+steps, "Date"]

    plt.plot(x_test_date, y_test_lab, label="Real data")
    plt.plot(x_test_date, y_predict_lab, label="Prediction")
    plt.plot(x_test_date, consignes, label="Instruction")

    plt.title(f"Temperature lab forecast ($R^2$={round(r2, 2)}, MAPE% = {round(mape, 2)} %)")
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
C0 = 1
eps0 = 0.05
gamm0 = "scale"
pol0 = 1

# [1, 'rbf', 0.05, 'scale', 1.2010106060927566, 0.5359420608879646]

ini_date = pd.to_datetime("2024-03-20T00:00:00") + pd.Timedelta(minutes=15)

test_svr("noves_alternades/dades_test_tractades.xlsx", ini_date, steps0, ker0, C0, eps0, pol0, gamm0)

#################


#######################################################################

"""
kernel_list = ["rbf"]  # ["rbf", "linear", "sigmoid"]
C_list = [0.1, 0.5, 1, 5, 10, 25, 50, 100]
epsilon_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
gamma_list = ["auto", "scale"]
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
                        r20, mape0 = test_svr(df_name0, date_start0, 288, ker, c, eps, pol, gamm)
                        r2vec = [c, ker, eps, gamm, mape0, r20]
                        r2_scores.append(r2vec)

            elif ker == "linear":
                model_svr(datax0, datay0, ker, c, eps, 0, "auto")
                r20, mape0 = test_svr(df_name0, date_start0, 288, ker, c, eps, 0, "auto")
                r2vec = [c, ker, eps, "nogam", mape0, r20]
                r2_scores.append(r2vec)

vector_ordenat = sorted(r2_scores, key=lambda x: x[-1], reverse=True)

print("Vector ordenat per la component r2:", vector_ordenat)

"""
########################################################################
"""
r2_scores = []

new_ker_lst = ['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf',
               'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf']


new_c_lst = [5, 10, 50, 5, 25, 10, 1, 0.5, 1, 25, 1, 0.1, 100, 50, 0.5, 0.5, 0.1, 100, 1, 0.1, 0.1, 5, 50, 100, 25,
             10, 0.5, 5, 0.1, 0.1, 5, 0.1, 10, 0.5, 0.5, 5, 1, 0.5, 5, 10, 25, 50, 100, 25, 50, 100, 0.5, 1, 1, 25,
             0.5, 0.1, 1, 10, 0.5, 1, 5, 10, 25, 50, 100, 0.1, 25, 50, 100, 10, 10, 25, 50, 5]

new_eps_lst = [0.01, 0.01, 0.01, 0.05, 0.01, 0.05, 0.05, 0.05, 0.01, 0.05, 0.01, 0.1, 0.05, 0.05, 0.01, 0.1, 0.01,
               0.01, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01, 0.5, 0.1, 0.5, 0.1, 0.1, 0.5, 0.5,
               0.1, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 1, 0.1, 1, 1, 0.5, 0.1, 1, 1, 1, 1, 1, 1, 1, 1,
               0.5, 0.5, 0.5, 1, 0.5, 0.1, 0.1, 1]

new_gamma_lst = ['auto', 'auto', 'auto', 'scale', 'auto', 'scale', 'scale', 'scale', 'auto', 'scale', 'scale',
                 'scale', 'scale', 'scale', 'auto', 'scale', 'scale', 'auto', 'scale', 'scale', 'auto', 'scale',
                 'scale', 'scale', 'scale', 'scale', 'scale', 'auto', 'auto', 'auto', 'scale', 'scale', 'auto', 'auto',
                 'auto', 'scale', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'auto', 'scale', 'scale', 'scale',
                 'scale', 'auto', 'scale', 'auto', 'scale', 'auto', 'scale', 'scale', 'auto', 'auto', 'auto', 'auto',
                 'auto', 'auto', 'auto', 'scale', 'scale', 'scale', 'scale', 'scale', 'scale', 'scale', 'auto', 'scale']

#new_ker_lst = ['rbf', 'rbf', 'rbf', 'rbf', 'rbf']

#new_c_lst = [0.1, 1, 0.5, 5, 5]

#new_eps_lst = [0.5, 0.01, 0.01, 0.01, 0.05]

#new_gamma_lst = ["scale", "auto", "auto", "scale", "auto"]

df_name0 = "noves_alternades/dades_test_tractades.xlsx"
df = pd.read_excel(f"Dades/consum/{df_name0}")

lst_of_dates = []
for row in df.index[2:len(df)-287]:
    if df.loc[row, "Hour"] == 0:
        lst_of_dates.append(df.loc[row, "Date"])

print(ls_of_dates)
lists_of_lists = [[] for i in range(0, len(lst_of_dates))]

for date, lst in zip(lst_of_dates, lists_of_lists):
    for nK, nC, nE, nG in zip(new_ker_lst, new_c_lst, new_eps_lst, new_gamma_lst):
        r2, mape = test_svr(df_name0, date + pd.Timedelta(minutes=15), 284, nK, nC, nE, 0, nG)
        r2vec = [nC, nK, nE, nG, mape, r2]
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

# Sorted models by average r2: [[1, 'rbf', 0.05, 'scale', 1.2010106060927566, 0.5359420608879646], [10, 'rbf', 0.1, 'scale', 1.3718270650135278, 0.5224551011317449], [5, 'rbf', 0.05, 'scale', 1.2112204283889207, 0.519049164634023], [5, 'rbf', 0.1, 'scale', 1.4064758411237392, 0.5122900265619145], [25, 'rbf', 0.1, 'scale', 1.3777045381348554, 0.49329015690094363], [0.5, 'rbf', 0.05, 'scale', 1.280018757880995, 0.48238995684942954], [1, 'rbf', 0.1, 'scale', 1.4057083126625638, 0.48201353144050446], [10, 'rbf', 0.05, 'scale', 1.3127716727516083, 0.44247972165610205], [0.5, 'rbf', 0.1, 'scale', 1.5220198531376272, 0.42899641461223653], [25, 'rbf', 0.05, 'scale', 1.4036209864223386, 0.3403881008383255], [50, 'rbf', 0.05, 'scale', 1.4091914524372688, 0.3351257858861937], [100, 'rbf', 0.05, 'scale', 1.4328298362497414, 0.29690918900261326], [10, 'rbf', 0.01, 'scale', 1.5204045592007307, 0.26218619934023757], [5, 'rbf', 0.01, 'scale', 1.5342035187957435, 0.24942155204826885], [0.1, 'rbf', 0.01, 'auto', 1.3990936932497071, 0.24557185098924203], [50, 'rbf', 0.01, 'scale', 1.482522718138996, 0.2421227836111104], [25, 'rbf', 0.01, 'scale', 1.5056342230710722, 0.24112832390101496], [0.5, 'rbf', 0.01, 'scale', 1.659563520723152, 0.23888766515196225], [100, 'rbf', 0.01, 'scale', 1.478080894220719, 0.22810865925036766], [0.1, 'rbf', 0.1, 'scale', 1.6364249857024542, 0.2027676460753401], [0.1, 'rbf', 0.01, 'scale', 1.655898423719459, 0.1919658702165147], [1, 'rbf', 0.01, 'scale', 1.7231795170032875, 0.1709467896739188], [0.1, 'rbf', 0.05, 'scale', 1.6597687558180862, 0.1598361915973502], [0.5, 'rbf', 0.1, 'auto', 1.7838929338588965, 0.13756482642840634], [1, 'rbf', 0.1, 'auto', 1.7248441772147807, 0.1340851810759925], [0.5, 'rbf', 0.01, 'auto', 1.493784834194702, 0.12147432199678061], [1, 'rbf', 0.01, 'auto', 1.525890042871, 0.053638853424898825], [25, 'rbf', 0.1, 'auto', 1.8697797270472443, 0.004844966823232566], [5, 'rbf', 0.1, 'auto', 1.9039818590614637, -0.007772353581294089], [0.1, 'rbf', 0.05, 'auto', 1.6608658497243804, -0.01059609265680308], [10, 'rbf', 0.1, 'auto', 1.893559111134228, -0.01820507091738923], [50, 'rbf', 0.1, 'auto', 1.9306953569280847, -0.03823772423709419], [0.5, 'rbf', 0.05, 'auto', 1.820467373575801, -0.19285803631506204], [5, 'rbf', 0.01, 'auto', 1.6557075953761422, -0.19831687744243992], [10, 'rbf', 0.01, 'auto', 1.6924794669915908, -0.24902194534327973], [25, 'rbf', 0.01, 'auto', 1.7317974327964254, -0.34374622395747745], [50, 'rbf', 0.01, 'auto', 1.758331922669008, -0.3670701893679711], [0.1, 'rbf', 0.5, 'scale', 2.3313537606124712, -0.49880424689347136], [0.1, 'rbf', 0.5, 'auto', 2.306839130134913, -0.5571569742666036], [100, 'rbf', 0.01, 'auto', 1.9098173375821585, -0.5999423367914276], [0.5, 'rbf', 0.5, 'auto', 2.4364824600420834, -0.7372794513511053], [5, 'rbf', 0.5, 'auto', 2.442458539668341, -0.7574124599674896], [10, 'rbf', 0.5, 'auto', 2.442458539668341, -0.7574124599674896], [25, 'rbf', 0.5, 'auto', 2.442458539668341, -0.7574124599674896], [50, 'rbf', 0.5, 'auto', 2.442458539668341, -0.7574124599674896], [100, 'rbf', 0.5, 'auto', 2.442458539668341, -0.7574124599674896], [1, 'rbf', 0.5, 'auto', 2.4458456218811295, -0.7640878729567009], [25, 'rbf', 1, 'scale', 2.617020363807586, -0.7756346887039124], [50, 'rbf', 1, 'scale', 2.617020363807586, -0.7756346887039124], [100, 'rbf', 1, 'scale', 2.617020363807586, -0.7756346887039124], [0.5, 'rbf', 0.5, 'scale', 2.5673079239834564, -0.7963899174377188], [1, 'rbf', 0.5, 'scale', 2.6131371023465277, -0.8379589535541718], [5, 'rbf', 0.5, 'scale', 2.6554565750175447, -0.8430486729344899], [10, 'rbf', 1, 'scale', 2.7081859224519462, -0.9079122501680936], [1, 'rbf', 1, 'scale', 2.7476876804970116, -0.9943674397338681], [0.5, 'rbf', 1, 'scale', 2.7697850228118743, -1.0437117181612625], [0.1, 'rbf', 1, 'scale', 2.760034474046669, -1.0449556515828053], [5, 'rbf', 1, 'scale', 2.8140263285055833, -1.060914206232436], [10, 'rbf', 0.5, 'scale', 2.8422348481751483, -1.1008045532841444], [0.1, 'rbf', 1, 'auto', 2.8622968806680618, -1.107712849830333], [0.5, 'rbf', 1, 'auto', 2.8843722802869225, -1.1443914539802065], [1, 'rbf', 1, 'auto', 2.892482414838166, -1.155871628384522], [5, 'rbf', 1, 'auto', 2.894195891340753, -1.158343763801458], [10, 'rbf', 1, 'auto', 2.894195891340753, -1.158343763801458], [25, 'rbf', 1, 'auto', 2.894195891340753, -1.158343763801458], [50, 'rbf', 1, 'auto', 2.894195891340753, -1.158343763801458], [100, 'rbf', 1, 'auto', 2.894195891340753, -1.158343763801458], [25, 'rbf', 0.5, 'scale', 2.9196980658229945, -1.2595346016551523], [50, 'rbf', 0.5, 'scale', 2.94557264641408, -1.3010500234523015], [100, 'rbf', 0.5, 'scale', 2.94557264641408, -1.3010500234523015]]