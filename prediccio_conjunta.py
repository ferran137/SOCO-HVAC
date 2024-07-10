import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import holidays
from sklearn.metrics import r2_score
import pytz


# Dades pel fer el test #

df_test = pd.read_excel(f"Dades/consum/noves_alternades/dades_test_tractades.xlsx")

data_inici = pd.to_datetime("2024-02-06T00:00:00") + pd.Timedelta(minutes=15)

index = df_test.loc[df_test["Date"] == data_inici].index[0]
######

######################################
# DADES NECESSÀRIES PER LA PREDICCIÓ #
######################################

# Les dades de dades_mimo.xlsx van del 2024-01-21 23:00:00 - 2024-05-01 21:55:00

steps = 284  # passos en el temps. 1 pas són 5 minuts (288 són 24h)

temp_lab_ant = df_test.loc[index-1, "Shelly Plus HT Temperature"]  # temperatura laboratori a temps t-1
temp_lab_ant2 = df_test.loc[index-2, "Shelly Plus HT Temperature"]  # temperatura laboratori a temps t-2

temp_dip_ant = df_test.loc[index-1, "geotermia temperatura_diposit"]  # temperatura diposit temps t-1
temp_dip_ant2 = df_test.loc[index-2, "geotermia temperatura_diposit"]  # temperatura diposit temps t-2

hora_inici_pred = 00
minut_inici_pred = 15

temperatura_exterior = df_test.loc[index:index+steps, "geotermia temperatura_exterior"].to_list()
consignes_laboratori = df_test.loc[index:index+steps, "consigna lab"].to_list()
consignes_diposit = df_test.loc[index:index+steps, "geotermia temperatura_consigna_diposit_hivern"].to_list()
dies_laborables_festius = [1]

################################
# MODELS DEL LAB I DEL DIPOSIT #
################################

# LABORATORI
# [1, 'rbf', 0.05, 'scale', 1.2010106060927566, 0.5359420608879646]
directori_model_lab = "Models/svr/"
C_lab = 1
kernel_lab = "rbf"
epsilon_lab = 0.05
gamma_lab = "scale"

c_str_lab = str(C_lab).replace('.', '')
epsilon_str_lab = str(epsilon_lab).replace('.', '')
gamma_str_lab = str(gamma_lab).replace('.', '')
name_lab = "svr_lab_" + kernel_lab + "_" + c_str_lab + "_" + epsilon_str_lab + "_" + gamma_str_lab
model_lab = joblib.load(f"{directori_model_lab + name_lab}.joblib")

# DIPOSIT
# [50, 'rbf', 0.05, 'auto', 1.310607542989364, 0.9251119709683806]
directori_model_dip = "Models/svr/"
C_dip = 50
kernel_dip = "rbf"
epsilon_dip = 0.05
gamma_dip = "auto"

c_str_dip = str(C_dip).replace('.', '')
epsilon_str_dip = str(epsilon_dip).replace('.', '')
gamma_str_dip = str(gamma_dip).replace('.', '')
name_dip = "svr_dip_" + kernel_dip + "_" + c_str_dip + "_" + epsilon_str_dip + "_" + gamma_str_dip
model_dip = joblib.load(f"{directori_model_dip + name_dip}.joblib")


##########################################
#  FUNCIONS NECESSARIES PER LA PREDICCIÓ #
##########################################

def tendencia_lab(lab1, lab2, dip1, exte, consigna_lab, threshold):
    """
    Funció que determina cap a on tendeix la temperatura del laboratori i si el fancoil ha d'estar obert o no.

    :param lab1: Temperatura laboratori temps t-1
    :param lab2: Temperatura laboratori temps t-2
    :param dip1: Temperatura diposit temps t-1
    :param exte: Temperatura exterior temps t
    :param consigna_lab: Consigna laboratori temps t
    :param threshold: Threshold superior i inferior. Per defecte és 0.2ºC
    :return: Vector [Temp. cap a on tendeix el laboratori, 0 si fancoil obert 1 si fancoil tancat]
    """
    if lab1 > consigna_lab + threshold:
        return [exte, 0]
    elif consigna_lab-threshold <= lab1 <= consigna_lab + threshold:
        pendent_recta = lab1-lab2
        if pendent_recta >= 0:
            return [dip1, 1]
        elif pendent_recta < 0:
            return [exte, 0]
    elif lab1 < consigna_lab - threshold:
        return [dip1, 1]


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
    """
    Funció que determina cap a on tendeix la temperatura del diposit i si gasta energia o no.

    :param dip1: Temperatura diposit temps t-1
    :param dip2: Temperatura diposit temps t-2
    :param exte: Temperatura exterior temps t
    :param tendencia_labo: Temperatura tendencia laboratori
    :param consigna_dip: Consigna diposit temps t
    :param threshold_sup: Threshold superior. Per defecte és 0.7ºC
    :param threshold_inf: Threshold inferior. Per defecte és 1ºC
    :return: Vector [Temp. cap a on tendeix el diposit, 0 si esta gastant 1 si no]
    """
    if tendencia_labo == exte:
        if dip1 > consigna_dip+threshold_sup:
            return [23, 0]  # temperatura sala de maquines. asumim temp constant a 23 graus
        elif consigna_dip-threshold_inf <= dip1 <= consigna_dip+threshold_sup:
            pendent_diposit = dip1-dip2
            if pendent_diposit >= 0:
                return [consigna_dip+threshold_sup, 1]
            elif pendent_diposit < 0:
                return [23, 0]
        elif dip1 < consigna_dip-threshold_inf:
            return [consigna_dip+threshold_sup, 1]

    elif tendencia_labo == dip1:
        if dip1 > consigna_dip+threshold_sup:
            return [consigna_dip-tend_exp(consigna_dip), 0]
        elif consigna_dip-threshold_inf <= dip1 <= consigna_dip+threshold_sup:
            pendent_diposit = dip1-dip2
            if pendent_diposit >= 0:
                return [consigna_dip-tend_exp(consigna_dip), 1]
            elif pendent_diposit < 0:
                return [consigna_dip-tend_exp(consigna_dip), 0]
        elif dip1 < consigna_dip-threshold_inf:
            return [consigna_dip-tend_exp(consigna_dip), 1]


def consum_total_fancoil(llista_on_off):
    """
    Càlcul del consum total del fancoil en kWh.

    :param llista_on_off: Llista on-off del fancoil en el periode predit
    :return: Consum total fancoil en kWh
    """
    potencia_fancoil = 0.2  # kW
    interval_5_min = 1/12  # 5 min = 1/12 hores (per tenir el consum en kWh)
    consum_5_minutal = potencia_fancoil*interval_5_min
    consum_total = 0
    for item in llista_on_off:
        consum_total = consum_total + item*consum_5_minutal
    return consum_total


def consum_total_diposit(llista_on_off):
    """
    Càlcul del consum total del diposit en kWh.

    (Valor del consum 5 minutal segur que es pot ajustar millor ajustant el consum la temperatura
    del diposit amb alguna funció)

    :param llista_on_off: Llista on-off del diposit en el periode predit
    :return: Consum total diposit en kWh
    """
    potencia_diposit = 2  # kW
    interval_5_min = 1/12  # 5 min = 1/12 hores (per tenir el consum en kWh)
    consum_5_minutal = potencia_diposit*interval_5_min
    consum_total = 0
    for item in llista_on_off:
        consum_total = consum_total + item*consum_5_minutal
    return consum_total


def distribucio_persones(pas, hora_inici, minut_inici):
    """

    :param minut_inici:
    :param pas: Pas en el futur des de l'hora d'inici. (Cada pas són 5 minuts)
    :param hora_inici: Hora en que es comença la predicció
    param minut_inici: Minut en que es comença la predicció. (Cal que sigui un múltiple de 5)
    :return: Numero de persones aproximat dins la sala
    """
    mu = 146.80148912324546
    stddev = 32.452878655270204
    a = 6.725450861215734
    ajust_hora_inici = (hora_inici + minut_inici/60)*(60/5)  # Passem unitats hora a step

    if ajust_hora_inici + pas < 288:
        val = a * np.exp(-(((pas + ajust_hora_inici) - mu) / stddev) ** 2 / 2)
        return val

    else:
        ajust_dia_nou = (ajust_hora_inici + pas)//288
        val = a * np.exp(-(((pas + ajust_hora_inici - ajust_dia_nou*288) - mu) / stddev) ** 2 / 2)
        return val


def llista_numero_persones(llista_festivitat, hora, minut, passos):
    """
    Llista aproximació numero de persones del lab
    :param llista_festivitat: Llista indicant si els dies a predir són laborables (1) o festiu / cap de setmana (0).
    :param hora: Hora d'inici de la predicció
    :param minut: Minut d'inici de la predicció. Ha de ser multiple de 5
    :param passos: Numero de passos de la predicció
    :return: Llista de persones a cada instant
    """

    if len(llista_festivitat) == 1:
        num_pers = []
        ajust_hora_inici = int((hora + minut / 60) * (60 / 5))

        for pas in range(ajust_hora_inici, ajust_hora_inici+passos):
            num_pers.append(llista_festivitat[0]*distribucio_persones(pas, hora, minut))
        return num_pers

    elif len(llista_festivitat) > 1:
        num_pers = []
        ajust_hora_inici = int((hora + minut / 60) * (60 / 5))
        bandera = 0
        for dia in llista_festivitat:
            bandera += 1
            if bandera == 1:
                for pas in range(ajust_hora_inici, 288):
                    num_pers.append(dia*distribucio_persones(pas, hora, minut))

            elif 1 < bandera < len(llista_festivitat):
                for pas in range(0,288):
                    num_pers.append(dia*distribucio_persones(pas,hora,minut))

            elif bandera == len(llista_festivitat):
                passos_restants = passos - 288*(len(llista_festivitat) - 1) + ajust_hora_inici
                for pas in range(0, passos_restants):
                    num_pers.append(dia*distribucio_persones(pas,hora,minut))
        return num_pers


##########################
# SIMULACIÓ TEMPERATURES #
##########################

numero_persones_laboratori = llista_numero_persones(dies_laborables_festius, hora_inici_pred, minut_inici_pred, steps)

new_temp_lab_ant = temp_lab_ant
new_temp_lab_ant2 = temp_lab_ant2
new_temp_dip_ant = temp_dip_ant
new_temp_dip_ant2 = temp_dip_ant2
new_num_pers = numero_persones_laboratori[0]
new_temp_exte = temperatura_exterior[0]
new_consigna_lab = consignes_laboratori[0]
new_consigna_dip = consignes_diposit[0]

new_tend_lab = tendencia_lab(new_temp_lab_ant, new_temp_lab_ant2, new_temp_dip_ant, new_temp_exte, new_consigna_lab,
                             0.2)
new_tend_dip = tendencia_dip(new_temp_dip_ant, new_temp_dip_ant2, new_temp_exte, new_tend_lab[0], new_consigna_dip,
                             0.7, 1)
temporary_lab = [
    new_temp_lab_ant,
    new_temp_lab_ant2,
    new_tend_lab[0],
    new_temp_exte,
    new_num_pers
]

temporary_dip = [
    new_temp_dip_ant,
    new_temp_dip_ant2,
    new_tend_dip[0]
]

prediccio_lab = model_lab.predict(np.array(temporary_lab).reshape(1, -1))[0]
prediccio_dip = model_dip.predict(np.array(temporary_dip).reshape(1, -1))[0]
print("Step 1")

pred_lab = [prediccio_lab]
pred_dip = [prediccio_dip]
fancoil_on_off = [new_tend_lab[1]]
diposit_on_off = [new_tend_dip[1]]
y_test_lab = [df_test.loc[index, "Shelly Plus HT Temperature"]]  # PEL TEST
y_test_dip = [df_test.loc[index, "geotermia temperatura_diposit"]]  # PEL TEST

for step in range(1, steps):
    print(f"Step {1 + step}")

    new_temp_lab_ant2 = new_temp_lab_ant
    new_temp_lab_ant = prediccio_lab
    new_temp_dip_ant2 = new_temp_dip_ant
    new_temp_dip_ant = prediccio_dip
    new_num_pers = numero_persones_laboratori[step]
    new_temp_exte = temperatura_exterior[step]
    new_consigna_lab = consignes_laboratori[step]
    new_consigna_dip = consignes_diposit[step]

    new_tend_lab = tendencia_lab(new_temp_lab_ant, new_temp_lab_ant2, new_temp_dip_ant, new_temp_exte, new_consigna_lab,
                                 0.2)
    new_tend_dip = tendencia_dip(new_temp_dip_ant, new_temp_dip_ant2, new_temp_exte, new_tend_lab[0], new_consigna_dip,
                                 0.7, 1)
    temporary_lab = [new_temp_lab_ant, new_temp_lab_ant2, new_tend_lab[0], new_temp_exte, new_num_pers]
    temporary_dip = [new_temp_dip_ant, new_temp_dip_ant2, new_tend_dip[0]]

    prediccio_lab = model_lab.predict(np.array(temporary_lab).reshape(1, -1))[0]
    prediccio_dip = model_dip.predict(np.array(temporary_dip).reshape(1, -1))[0]

    pred_lab.append(prediccio_lab)
    pred_dip.append(prediccio_dip)
    fancoil_on_off.append(new_tend_lab[1])
    diposit_on_off.append(new_tend_dip[1])
    y_test_lab.append(df_test.loc[index + step, "Shelly Plus HT Temperature"])
    y_test_dip.append(df_test.loc[index + step, "geotermia temperatura_diposit"])


#####################
# EXCLUSIU PEL TEST #
#####################

def r2_mape(y_test, y_predict):
    r2 = r2_score(y_test, y_predict)
    err_pre_mape = []
    true_mae = []

    for i in range(0, len(y_predict)):
        error_mae = np.abs(y_test[i] - y_predict[i])
        true_mae.append(error_mae)

        err = np.abs((y_test[i] - y_predict[i]) / y_test[i])
        err_pre_mape.append(err * 100)

    mae_final = np.mean(true_mae)
    mape = np.mean(err_pre_mape)
    return [r2, mape, mae_final]


######################################################
# GRÀFIQUES DE TEMPERATURA I CÀLCULS DE CONSUM TOTAL #
######################################################

x_steps = np.linspace(0, steps, steps)
y_fill_lab = [min(pred_lab+consignes_laboratori)+i*0 for i in range(len(x_steps))]
y_fill_dip = [min(pred_dip+consignes_diposit)+i*0 for i in range(len(x_steps))]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_steps, y_test_lab, label="Real data")
plt.plot(x_steps, pred_lab, label="Prediction")
plt.plot(x_steps, consignes_laboratori[1:], label="Instruction")
plt.fill_between(x=x_steps, y1=pred_lab, y2=y_fill_lab, where=(np.array(fancoil_on_off) > 0),
                 color='skyblue', alpha=0.5, label="Fancoil ON/OFF")
plt.title(f"Total fancoil consumption {round(consum_total_fancoil(fancoil_on_off),2)} kWh. "
          f"Score R2 = {round(r2_mape(pred_lab, y_test_lab)[0], 2)}")
plt.xlabel('Steps (1 step = 5 min)')
plt.ylabel('Temperatura (ºC)')
plt.legend(loc="best")

plt.subplot(1, 2, 2)
plt.plot(x_steps, y_test_dip, label="Real data")
plt.plot(x_steps, pred_dip, label="Prediction")
plt.plot(x_steps, consignes_diposit[1:], label="Instruction")
plt.fill_between(x=x_steps, y1=pred_dip, y2=y_fill_dip, where=(np.array(diposit_on_off) > 0),
                 color='skyblue', alpha=0.5, label="Inertia tank ON/OFF")
plt.title(f"Total inertia tank consumption {round(consum_total_diposit(diposit_on_off),2)} kWh."
          f" Score R2 = {round(r2_mape(pred_dip, y_test_dip)[0],2)}")
plt.xlabel('Steps (1 step = 5 min)')
plt.ylabel('Temperatura (ºC)')
plt.legend(loc="best")
plt.suptitle(f"Forecasting HVAC system {round(steps*(5/60),2)}h ({steps} steps)")
plt.tight_layout()
plt.savefig(f"Plots/Pred_conjunta_{steps}_steps.jpg")
plt.show()