import pandas as pd
import matplotlib.pyplot as plt
import os

# Pfade zu den Dateien relativ zum aktuellen Verzeichnis
umsatzdaten_path = os.path.join('Woche 3', 'Deskriptive Statistiken', 'umsatzdaten_gekuerzt.csv')
wetterdaten_path = os.path.join('Woche 3', 'Deskriptive Statistiken', 'wetter.csv')
output_plot_path = os.path.join('Woche 3', 'Deskriptive Statistiken', 'umsatz_ueber_die_zeit.png')
output_corr_path = os.path.join('Woche 3', 'Deskriptive Statistiken', 'korrelation_umsatz_wetter.png')
output_feature_path = os.path.join('Woche 3', 'Deskriptive Statistiken', 'feature_engineering.png')

# Überprüfen, ob die Dateien existieren
if not os.path.isfile(umsatzdaten_path):
    raise FileNotFoundError(f'Die Datei {umsatzdaten_path} wurde nicht gefunden. Bitte überprüfen Sie den Pfad.')

if not os.path.isfile(wetterdaten_path):
    raise FileNotFoundError(f'Die Datei {wetterdaten_path} wurde nicht gefunden. Bitte überprüfen Sie den Pfad.')

# Importiere die Daten
umsatzdaten = pd.read_csv(umsatzdaten_path)
wetterdaten = pd.read_csv(wetterdaten_path)

# Zeige die ersten Zeilen der Daten an
print(umsatzdaten.head())
print(wetterdaten.head())

# Konvertiere Datumsspalten in Datetime-Format
umsatzdaten['Datum'] = pd.to_datetime(umsatzdaten['Datum'])
wetterdaten['Datum'] = pd.to_datetime(wetterdaten['Datum'])

# Merge der beiden Datensätze auf das Datum
merged_data = pd.merge(umsatzdaten, wetterdaten, on='Datum')

# Explorative Datenanalyse

# Umsatz über die Zeit plotten
plt.figure(figsize=(12, 6))
plt.plot(merged_data['Datum'], merged_data['Umsatz'], label='Umsatz')
plt.xlabel('Datum')
plt.ylabel('Umsatz')
plt.title('Umsatz über die Zeit')
plt.legend()

# Speichere den Plot als PNG-Datei
plt.savefig(output_plot_path)
plt.close()

# Korrelation zwischen Umsatz und Wetterbedingungen
correlation_matrix = merged_data.corr()
print(correlation_matrix['Umsatz'])

# Korrelation Plot speichern
plt.figure(figsize=(12, 6))
plt.matshow(correlation_matrix)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.colorbar()
plt.title('Korrelation zwischen Umsatz und Wetterbedingungen', pad=20)
plt.savefig(output_corr_path)
plt.close()

# Feature Engineering: Beispielsweise gleitender Durchschnitt
merged_data['Umsatz_ma_7'] = merged_data['Umsatz'].rolling(window=7).mean()

# Feature Engineering Plot speichern
plt.figure(figsize=(12, 6))
plt.plot(merged_data['Datum'], merged_data['Umsatz'], label='Umsatz')
plt.plot(merged_data['Datum'], merged_data['Umsatz_ma_7'], label='7-Tage gleitender Durchschnitt')
plt.xlabel('Datum')
plt.ylabel('Umsatz')
plt.title('Umsatz und 7-Tage gleitender Durchschnitt')
plt.legend()
plt.savefig(output_feature_path)
plt.close()