import pandas as pd
import math
from datetime import timedelta
import holidays
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_excel(f"Dades/consum/dades_mimo.xlsx")
df = df.dropna().reset_index(drop=True)

# Columna de les dates no té nom, li posem un
df.columns.values[0] = "Date"
# Li sumem una hora pq esta en hora utf
new_date_actualized = [date + timedelta(hours=1) for date in df["Date"]]
df["Date"] = new_date_actualized
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
df["numero persones"] = numero_persones


import numpy as np
import statistics

x_data = []
y_data = []
consignes_diposit = set(df["geotermia temperatura_consigna_diposit_hivern"])
info_list = []
for consigna in consignes_diposit:
    diff_temp = []
    all_Temp_consignes = df[(df["geotermia temperatura_consigna_diposit_hivern"] == consigna) &
                            (df["Shelly Plus 1pm Fancoil switch_0"] == 1)].reset_index(drop=True)

    diff_list = []
    for index in range(0, len(all_Temp_consignes)):
        diff = (all_Temp_consignes.loc[index, "geotermia temperatura_consigna_diposit_hivern"] -
                all_Temp_consignes.loc[index, "geotermia temperatura_diposit"])

        if diff > 0:
            diff_list.append(diff)

    if len(diff_list) == 0:
        continue
    info_list.append([consigna, np.mean(diff_list), np.median(diff_list)])

    print(f"\nTEMPERATURA {consigna} (data len: {len(diff_list)})")
    print(f"Mitjana {np.mean(diff_list)}")
    print(f"Mediana {np.median(diff_list)}")
    print(f"Moda {statistics.mode(diff_list)}")

# Deixem fora als valors de consignes que tenen poques temperatures, segurament és degut a algun error de lectura o algo
# perquè no hi ha cap consigna que sigui 39.7 o 48 per exemple

# Given data points
x_data = np.array([25, 33, 40, 42, 45, 48, 55])
y_data = np.array([0.7, 0.75, 1.15, 1.85, 2.7, 5.8, 19.26])


def func(x, a, b, d,g):
    return a * np.exp(b * x+g) + d

popt, pcov = curve_fit(func, x_data, y_data, p0=(0.1, 0.31, 0,0.001))  # Initial guess for parameters: a=1, b=1, d=24.2
a_opt, b_opt, d_opt, g_opt= popt

print(f"\nf(x) = A*e^(bx+g)+d")
print(f"A={a_opt}")
print(f"b={b_opt}")
print(f"d={d_opt}")
print(f"g={g_opt}")


xx = np.linspace(min(x_data), max(x_data), 1000)
yy_fit = func(xx, a_opt, b_opt, d_opt, g_opt)

plt.figure(figsize=(8, 6), dpi=1200)
plt.scatter(x_data, y_data, label='Data points', color='blue')
plt.plot(xx, yy_fit, label='Fitted exponential curve', color='red')
plt.xlabel('Instruction ($\degree$C)')
plt.ylabel('Temperature difference ($\degree$C)')
plt.legend()
plt.savefig(f"Plots/plots_overleaf/equilibrium_tank.jpg")
plt.close()
plt.show()

# Aproximació de la temperatura a la que ha de tendir el diposit quan tant el diposit com el fancoil estan oberts.
# Ho aproxim-ho amb una exponencial. (executar el programa ja es veu el plot guapo)
# Per seleccionar les dade el que he fet és separar les temperatures per consignes, mirar les temperatures del
# laboratori que queden per sota de la consigna quan els dos estan oberts. Fer la mitjana i agafar aquests punts per
# construir una exponencial perquè sembla ser la corva que s'hi adapta millor
# best fitting curve. f(x) = A*e^(bx) . A=0.01, b=0.129