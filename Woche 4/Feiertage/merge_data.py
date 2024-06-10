import pandas as pd
import os

# Pfade zu den Dateien relativ zum aktuellen Verzeichnis
umsatzdaten_path = os.path.join('Woche 4', 'Feiertage', 'umsatzdaten_gekuerzt.csv')
wetterdaten_path = os.path.join('Woche 4', 'Feiertage', 'wetter.csv')
feiertagsdaten_path = os.path.join('Woche 4', 'Feiertage', 'feiertage.csv')
output_path = os.path.join('Woche 4', 'Feiertage', 'zusammengefuehrte_daten.csv')

# Überprüfen, ob die Dateien existieren
if not os.path.isfile(umsatzdaten_path):
    raise FileNotFoundError(f'Die Datei {umsatzdaten_path} wurde nicht gefunden. Bitte überprüfen Sie den Pfad.')

if not os.path.isfile(wetterdaten_path):
    raise FileNotFoundError(f'Die Datei {wetterdaten_path} wurde nicht gefunden. Bitte überprüfen Sie den Pfad.')

if not os.path.isfile(feiertagsdaten_path):
    raise FileNotFoundError(f'Die Datei {feiertagsdaten_path} wurde nicht gefunden. Bitte überprüfen Sie den Pfad.')

# Importiere die Daten
umsatzdaten = pd.read_csv(umsatzdaten_path)
wetterdaten = pd.read_csv(wetterdaten_path)
feiertagsdaten = pd.read_csv(feiertagsdaten_path)

# Konvertiere Datumsspalten in Datetime-Format
umsatzdaten['Datum'] = pd.to_datetime(umsatzdaten['Datum'])
wetterdaten['Datum'] = pd.to_datetime(wetterdaten['Datum'])
feiertagsdaten['Datum'] = pd.to_datetime(feiertagsdaten['Datum'])

# Merge der Umsatz- und Wetterdaten
merged_data = pd.merge(umsatzdaten, wetterdaten, on='Datum', how='left')

# Merge mit Feiertagsdaten
merged_data = pd.merge(merged_data, feiertagsdaten, on='Datum', how='left')

# Feiertagsinformationen in binäre Variable umwandeln
merged_data['Feiertag'] = merged_data['Feiertag'].apply(lambda x: 1 if pd.notna(x) else 0)

# Fehlende Werte in Wetterdaten auffüllen
merged_data.fillna(method='ffill', inplace=True)

# Ausgabe der ersten Zeilen des zusammengeführten Datensatzes
print(merged_data.head())

# Speichere den zusammengeführten Datensatz als CSV
merged_data.to_csv(output_path, index=False)