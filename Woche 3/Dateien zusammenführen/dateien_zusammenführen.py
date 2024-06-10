import pandas as pd
import os

# Pfade zu den Dateien relativ zum aktuellen Verzeichnis
umsatzdaten_path = os.path.join('Woche 3', 'Dateien zusammenführen', 'umsatzdaten_gekuerzt.csv')
kiwo_path = os.path.join('Woche 3', 'Dateien zusammenführen', 'kiwo.csv')
wetter_path = os.path.join('Woche 3', 'Dateien zusammenführen', 'wetter.csv')
output_path = os.path.join('Woche 3', 'Dateien zusammenführen', 'zusammengefuehrte_daten.csv')

# Lade die CSV-Dateien in DataFrames
umsatzdaten = pd.read_csv(umsatzdaten_path)
kiwo = pd.read_csv(kiwo_path)
wetter = pd.read_csv(wetter_path)

# Konvertiere die Datumsspalten in das Datumsformat
umsatzdaten["Datum"] = pd.to_datetime(umsatzdaten["Datum"])
kiwo["Datum"] = pd.to_datetime(kiwo["Datum"])
wetter["Datum"] = pd.to_datetime(wetter["Datum"])

# Führe die DataFrames anhand des Datums zusammen
merged_data = pd.merge(umsatzdaten, kiwo, on="Datum", how="left")
merged_data = pd.merge(merged_data, wetter, on="Datum", how="left")

# Speichere das zusammengeführte DataFrame als CSV-Datei
merged_data.to_csv(output_path, index=False)