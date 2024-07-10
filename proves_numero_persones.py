import pandas as pd
import math
from datetime import timedelta
import holidays
import numpy as np
from scipy.optimize import curve_fit
import pytz


def convert_to_spain_time(utc_time):
    utc_zone = pytz.utc
    spain_zone = pytz.timezone('Europe/Madrid')
    utc_time = utc_zone.localize(utc_time)
    return pd.to_datetime(utc_time.astimezone(spain_zone).strftime('%Y-%m-%d %H:%M:%S'))


df = pd.read_excel(f"Dades/consum/dades_tractades.xlsx")
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
# Numero Persones
numero_persones = [math.ceil(i) for i in df["counter.numero_persones"]]
numero_persones = [i if i <= 9 else 9 for i in numero_persones]
df["counter.numero_persones"] = numero_persones

df = df[df["Day"] != 0].reset_index(drop=True)

print(df)
def gaussian(x, mu, stddev, a):
    return a * np.exp(-((x - mu) / stddev) ** 2 / 2)


how_many_days = math.ceil(len(df)/288)
date_of_start = pd.to_datetime("2024-01-27T00:00:00")
date_of_end = pd.to_datetime("2024-05-01T00:00:00")

llista_mu = []
llista_stddev = []
llista_a = []

for i in range(0, how_many_days):

    date_of_start = date_of_start + timedelta(days=1)
    if len(df.loc[df["Date"] == date_of_start]) == 0:
        continue
    else:
        index_start = df.loc[df["Date"] == date_of_start].index[0]
        if index_start + 288 > len(df):
            break

    date_of_end = date_of_start + timedelta(days=1)
    if len(df.loc[df["Date"] == date_of_end]) == 0:
        continue
    else:
        index_end = df.loc[df["Date"] == date_of_end].index[0]

    if index_end-index_start < 288:
        print(index_end - index_start)
        continue
    else:
        print(index_end - index_start)

    points_x = []
    points_y = []
    flag = 0
    for item in df[index_start:index_end]["counter.numero_persones"]:
        flag += 1
        if item > 0:
            points_y.append(item)
            points_x.append(flag)

    if len(points_y) < 10:
        continue
    else:

        try:
            initial_guess = [150, 40, 8]
            popt, pcov = curve_fit(gaussian, points_x, points_y, initial_guess)

            if np.all(np.diag(pcov) != 0):
                llista_mu.append(popt[0])
                llista_stddev.append(popt[1])
                llista_a.append(popt[2])

            else:
                print("optimal Parameters not found. skipping iteration")
        except RuntimeError as e:
            print("RuntimeError", e)
            print("optimal paramenters not found. skipping iteration")

print(llista_mu)
print(llista_stddev)
print(llista_a)

print(f"Mean mu {np.median(llista_mu)}")  
print(f"Mean stddev {np.median(llista_stddev)}") 
print(f"Mean a {np.median(llista_a)}") 


# Aproximació del numero de persones en u dia. Aproximació a una gausiana.
