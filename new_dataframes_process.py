import pandas as pd
import pytz
import math
import holidays


def convert_to_spain_time(utc_time):
    utc_zone = pytz.utc
    spain_zone = pytz.timezone('Europe/Madrid')
    utc_time = utc_zone.localize(utc_time)
    return pd.to_datetime(utc_time.astimezone(spain_zone).strftime('%Y-%m-%d %H:%M:%S'))


df_name0 = "dades_tractades_ultra.xlsx"

df = pd.read_excel(f"Dades/consum/noves_alternades/{df_name0}")
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


# Define the dates to filter
dates_to_select = ['2024-01-29',
                   '2024-02-06', '2024-02-10', '2024-02-15', '2024-02-17', '2024-02-22', '2024-02-27',
                   '2024-03-01', '2024-03-06', '2024-03-11', '2024-03-20', '2024-03-22', '2024-03-24',
                   '2024-04-02', '2024-04-06', '2024-04-14', '2024-04-16', '2024-04-20', '2024-04-28']
dates_to_select = pd.to_datetime(dates_to_select).date

# Filter the DataFrame based on the selected dates
filtered_df = df[df['Date'].dt.date.isin(dates_to_select)].reset_index(drop=True)
print(filtered_df)

print(len(filtered_df)/len(df)*100)

# Save the filtered DataFrame to a new file
filtered_df.to_excel('Dades/consum/noves_alternades/dades_test_tractades.xlsx', index=False)

# Get the remaining rows
remaining_df = df[~df['Date'].dt.date.isin(dates_to_select)].reset_index(drop=True)
print(remaining_df)
print(len(remaining_df)/len(df)*100)


# Save the remaining DataFrame to a new file
remaining_df.to_excel('Dades/consum/noves_alternades/dades_train_tractades.xlsx', index=False)

