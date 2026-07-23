import pandas as pd
import os

cartella_stagioni = "stagioni"
tutti_i_df = []

# Colonne che vogliamo tenere: risultati + quote
colonne_da_tenere = [
    "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "B365H", "B365D", "B365A",   # Bet365
    "PSH", "PSD", "PSA"          # Pinnacle (le più accurate)
]

colonne_alternative = {
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "Bb1X2": None,  # lo ignoriamo ma potrebbe esserci
}

for file in sorted(os.listdir(cartella_stagioni)):
    if file.endswith(".txt"):
        percorso = os.path.join(cartella_stagioni, file)
        print(f"Leggendo {file}...")
        
        try:
            df = pd.read_csv(percorso, encoding="latin1", on_bad_lines="skip")
        except:
            try:
                df = pd.read_csv(percorso, encoding="cp1252", on_bad_lines="skip")
            except:
                df = pd.read_csv(percorso, encoding="latin1", on_bad_lines="skip", engine="python")
        
        # Rinomina colonne se necessario
        if "Home" in df.columns and "HomeTeam" not in df.columns:
            df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
        
        # Trova le colonne presenti
        colonne_presenti = [c for c in colonne_da_tenere if c in df.columns]
        
        if "HomeTeam" not in colonne_presenti or "AwayTeam" not in colonne_presenti:
            print(f"  ⚠️ {file}: colonne essenziali mancanti, lo salto.")
            continue
        
        # Prendi solo le colonne che ci sono
        df = df[colonne_presenti]

        # Tiene traccia della stagione di provenienza (serve per il backtesting per stagione)
        df["Stagione"] = file.replace(".txt", "")

        # Converti i numeri
        for col in ["FTHG", "FTAG"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Converti le quote in numeri
        for col in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["FTHG", "FTAG"])
        tutti_i_df.append(df)
        
        # Messaggio informativo sulle quote
        ha_quote = "B365H" in colonne_presenti or "PSH" in colonne_presenti
        print(f"  ✅ {len(df)} partite" + (" (con quote)" if ha_quote else " (senza quote)"))

# Unisci tutto
df_finale = pd.concat(tutti_i_df, ignore_index=True)

# Per le partite senza quote, lascia NaN (il modello gestirà l'assenza)
df_finale.to_csv("serie_a.csv", index=False)

print(f"\n✅ Creato serie_a.csv con {len(df_finale):,} partite.")
print(f"Colonne: {df_finale.columns.tolist()}")
print(f"Partite con quote Bet365: {df_finale['B365H'].notna().sum() if 'B365H' in df_finale.columns else 0}")
print(f"Partite con quote Pinnacle: {df_finale['PSH'].notna().sum() if 'PSH' in df_finale.columns else 0}")