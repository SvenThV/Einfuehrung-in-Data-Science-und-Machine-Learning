import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Pfad zur Datei relativ zum aktuellen Verzeichnis
file_path = os.path.join('Woche 2 - Umsatzdaten', 'umsatzdaten_gekuerzt.csv')

# Überprüfen, ob die Datei existiert
if not os.path.isfile(file_path):
    raise FileNotFoundError(f'Die Datei {file_path} wurde nicht gefunden. Bitte überprüfen Sie den Pfad.')

# Daten einlesen
df = pd.read_csv(file_path)

# Schritt 1: Wochentag aus Datum berechnen
df['Datum'] = pd.to_datetime(df['Datum'])
df['Wochentag'] = df['Datum'].dt.day_name()

# Schritt 2: Durchschnittliche Umsätze je Wochentag berechnen
df['Wochentag'] = pd.Categorical(df['Wochentag'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
umsatz_durchschnitt = df.groupby('Wochentag')['Umsatz'].mean()

# Schritt 3: Konfidenzintervalle berechnen
conf_intervals = df.groupby('Wochentag')['Umsatz'].agg(lambda x: stats.sem(x) * stats.t.ppf((1 + 0.95) / 2., len(x) - 1))

# Schritt 4: NaN-Werte in Konfidenzintervallen behandeln
conf_intervals = conf_intervals.fillna(0)

# Schritt 5: Balkendiagramm mit Konfidenzintervallen erstellen
plt.figure(figsize=(10, 6))
plt.bar(umsatz_durchschnitt.index, umsatz_durchschnitt.values, yerr=conf_intervals.values, capsize=5)
plt.xlabel('Wochentag')
plt.ylabel('Durchschnittlicher Umsatz')
plt.title('Durchschnittlicher Umsatz je Wochentag mit Konfidenzintervallen')
plt.xticks(rotation=45)
plt.tight_layout()

# Speichern des Diagramms als PNG-Datei
plt.savefig('umsatzdiagramm.png')
plt.show()