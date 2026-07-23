import pandas as pd
import os

cartella_stagioni = "stagioni"
tutti_i_df = []

# Colonne che vogliamo tenere sempre: risultati, quote dei due bookmaker storici,
# statistiche di partita (utili per feature future: tiri, corner, cartellini).
colonne_da_tenere = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
    "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",   # Bet365
    "PSH", "PSD", "PSA",         # Pinnacle (le più accurate)
]

# Quota media di mercato: preferiamo il consenso su più bookmaker quando disponibile
# (AvgH/D/A dal 2019 circa, BbAvH/D/A per gli anni precedenti su football-data.co.uk),
# altrimenti ripieghiamo su Bet365 da solo. Serve per il modello a pesare meglio "quote".
COLONNE_QUOTA_MEDIA = {
    "H": ["AvgH", "BbAvH", "B365H"],
    "D": ["AvgD", "BbAvD", "B365D"],
    "A": ["AvgA", "BbAvA", "B365A"],
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

        # Quota media di mercato: cascata di priorità calcolata sulle colonne grezze,
        # PRIMA di scartare le colonne non nell'elenco fisso.
        for esito, candidate in COLONNE_QUOTA_MEDIA.items():
            colonna_scelta = next((c for c in candidate if c in df.columns), None)
            df[f"OddsAvg{esito}"] = pd.to_numeric(df[colonna_scelta], errors="coerce") if colonna_scelta else pd.NA

        # Trova le colonne presenti tra quelle richieste + le quote di consenso appena create
        colonne_presenti = [c for c in colonne_da_tenere if c in df.columns] + [f"OddsAvg{e}" for e in COLONNE_QUOTA_MEDIA]

        if "HomeTeam" not in colonne_presenti or "AwayTeam" not in colonne_presenti:
            print(f"  ⚠️ {file}: colonne essenziali mancanti, lo salto.")
            continue

        # Prendi solo le colonne che ci sono
        df = df[colonne_presenti]

        # Tiene traccia della stagione di provenienza (serve per il backtesting per stagione)
        df["Stagione"] = file.replace(".txt", "")

        # Converti i numeri
        for col in ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Converti le quote in numeri
        for col in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Converti la data reale (formato dd/mm/yy o dd/mm/yyyy a seconda della stagione)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        df = df.dropna(subset=["FTHG", "FTAG"])
        tutti_i_df.append(df)

        # Messaggio informativo sulle quote
        ha_quote = "B365H" in colonne_presenti or "PSH" in colonne_presenti
        print(f"  ✅ {len(df)} partite" + (" (con quote)" if ha_quote else " (senza quote)"))

# Unisci tutto e riordina per data reale (non fidarsi solo dell'ordine dei file)
df_finale = pd.concat(tutti_i_df, ignore_index=True)
if "Date" in df_finale.columns:
    df_finale = df_finale.sort_values("Date", kind="stable").reset_index(drop=True)

# Per le partite senza quote, lascia NaN (il modello gestirà l'assenza)
df_finale.to_csv("serie_a.csv", index=False)

print(f"\n✅ Creato serie_a.csv con {len(df_finale):,} partite.")
print(f"Colonne: {df_finale.columns.tolist()}")
print(f"Partite con quote Bet365: {df_finale['B365H'].notna().sum() if 'B365H' in df_finale.columns else 0}")
print(f"Partite con quote Pinnacle: {df_finale['PSH'].notna().sum() if 'PSH' in df_finale.columns else 0}")
print(f"Partite con quota media di consenso: {df_finale['OddsAvgH'].notna().sum()}")
if "Date" in df_finale.columns:
    print(f"Partite con data valida: {df_finale['Date'].notna().sum()} / {len(df_finale)}")
