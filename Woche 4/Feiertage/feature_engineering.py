import pandas as pd
import os

# Pfad zu den Dateien relativ zum aktuellen Verzeichnis
input_path = os.path.join('Woche 4', 'Feiertage', 'zusammengefuehrte_daten.csv')
output_path = os.path.join('Woche 4', 'Feiertage', 'erweiterte_daten.csv')

# Überprüfen, ob die Datei existiert
if not os.path.isfile(input_path):
    raise FileNotFoundError(f'Die Datei {input_path} wurde nicht gefunden. Bitte überprüfen Sie den Pfad.')

# Importiere den zusammengeführten Datensatz
merged_data = pd.read_csv(input_path)

# Feature Engineering: Beispielsweise gleitender Durchschnitt
merged_data['Umsatz_ma_7'] = merged_data['Umsatz'].rolling(window=7).mean()

# Weitere Features
merged_data['Wochentag'] = pd.to_datetime(merged_data['Datum']).dt.dayofweek
merged_data['Monat'] = pd.to_datetime(merged_data['Datum']).dt.month
merged_data['Tag_des_Jahres'] = pd.to_datetime(merged_data['Datum']).dt.dayofyear

# Zeitabhängige Features
merged_data['Umsatz_Lag_1'] = merged_data['Umsatz'].shift(1)
merged_data['Umsatz_Lag_7'] = merged_data['Umsatz'].shift(7)

# Fehlende Werte nach Lagging auffüllen
merged_data.fillna(0, inplace=True)

# Ausgabe der ersten Zeilen des erweiterten Datensatzes
print(merged_data.head())

# Speichere den erweiterten Datensatz als CSV
merged_data.to_csv(output_path, index=False)