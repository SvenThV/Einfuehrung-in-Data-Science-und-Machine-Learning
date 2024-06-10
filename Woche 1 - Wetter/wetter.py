import pandas as pd
from scipy import stats

# Importiere den Datensatz
df = pd.read_csv("wetter.csv")

# Berechne die Gesamtdurchschnittstemperatur
gesamt_durchschnittstemperatur = df["Temperatur"].mean()
print(f"Die Gesamtdurchschnittstemperatur beträgt: {gesamt_durchschnittstemperatur:.2f} °C")

# Filtern der Daten für den Monat Juli und Berechnen der Durchschnittstemperatur
df_juli = df[df["Datum"].str.contains("-07-")]
durchschnittstemperatur_juli = df_juli["Temperatur"].mean()
print(f"Die Durchschnittstemperatur für den Monat Juli beträgt: {durchschnittstemperatur_juli:.2f} °C")

# Filtern der Daten für den Monat Mai und Berechnen der Durchschnittstemperatur
df_mai = df[df["Datum"].str.contains("-05-")]
durchschnittstemperatur_mai = df_mai["Temperatur"].mean()
print(f"Die Durchschnittstemperatur für den Monat Mai beträgt: {durchschnittstemperatur_mai:.2f} °C")

# Vergleiche, ob die Monate Juli und Mai sich in ihrer Durchschnittstemperatur signifikant unterscheiden
t_stat, p_value = stats.ttest_ind(df_juli["Temperatur"], df_mai["Temperatur"])

print(f"t-Wert: {t_stat:.2f}, p-Wert: {p_value:.5f}")

if p_value < 0.05:
    print("Die Durchschnittstemperaturen für den Juli und Mai unterscheiden sich signifikant.")
else:
    print("Die Durchschnittstemperaturen für den Juli und Mai unterscheiden sich nicht signifikant.")
