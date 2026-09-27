"""
Unisce i file grezzi di nba_stagioni/*.csv (una riga per squadra per partita,
formato nba_api/LeagueGameLog) in nba_storico.csv: una riga per partita, con
casa e trasferta separati in colonne diverse. Stessa idea di unisci_dati.py per
il calcio, adattata al fatto che qui non esiste un formato "gia' orientato":
la colonna MATCHUP di nba_api distingue casa (" vs. ") da trasferta (" @ "),
ed e' l'unico modo per accoppiare le due righe di ogni GAME_ID, dato che
stats.nba.com non fornisce un campo home/away esplicito in questo endpoint.
"""
import glob
import os

import pandas as pd

CARTELLA = "nba_stagioni"

# Statistiche di box score da portare a valle: base per feature future
# (efficienza tiro, rimbalzi, palle perse) come HS/AS/HC/AC lo sono per il calcio.
COLONNE_BOX = [
    "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
]


def carica_grezzi():
    file_stagioni = sorted(glob.glob(os.path.join(CARTELLA, "*.csv")))
    if not file_stagioni:
        raise FileNotFoundError(f"Nessun file in {CARTELLA}/: esegui prima scarica_nba.py")
    return pd.concat([pd.read_csv(f) for f in file_stagioni], ignore_index=True)


def accoppia_partite(df):
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["CASA"] = df["MATCHUP"].str.contains(" vs. ", regex=False)

    # Scarta le partite non accoppiabili in esattamente due righe (dati grezzi
    # incompleti, All-Star Game, partite annullate): senza un contatore == 2
    # non c'e' garanzia che esistano sia la riga casa sia quella trasferta.
    conteggio = df.groupby("GAME_ID")["GAME_ID"].transform("count")
    df = df[conteggio == 2]

    casa = df[df["CASA"]].set_index("GAME_ID")
    trasferta = df[~df["CASA"]].set_index("GAME_ID")
    comuni = casa.index.intersection(trasferta.index)
    casa, trasferta = casa.loc[comuni], trasferta.loc[comuni]

    partite = pd.DataFrame({
        "GameID": comuni,
        "Date": casa["GAME_DATE"].values,
        "Stagione": casa["SEASON_ID"].values,
        "HomeTeam": casa["TEAM_ABBREVIATION"].values,
        "AwayTeam": trasferta["TEAM_ABBREVIATION"].values,
    })
    for col in COLONNE_BOX:
        partite[f"{col}_Home"] = casa[col].values
        partite[f"{col}_Away"] = trasferta[col].values

    partite["Winner"] = (partite["PTS_Home"] > partite["PTS_Away"]).map({True: "H", False: "A"})
    return partite.sort_values("Date", kind="stable").reset_index(drop=True)


def main():
    grezzi = carica_grezzi()
    partite = accoppia_partite(grezzi)
    partite = partite.drop_duplicates(subset=["GameID"])
    partite.to_csv("nba_storico.csv", index=False)
    print(
        f"Creato nba_storico.csv con {len(partite):,} partite "
        f"({partite['Date'].min().date()} - {partite['Date'].max().date()})"
    )


if __name__ == "__main__":
    main()
