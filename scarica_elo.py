"""
Scarica lo storico dei rating Elo per ogni squadra presente in serie_a.csv da
clubelo.com (API pubblica e gratuita: http://api.clubelo.com/{Squadra}) e lo
salva in elo_storico.csv, cosi' l'app non deve fare chiamate esterne a ogni
avvio (piu' robusto per il deploy, riproducibile per il backtesting).

Rilancialo di tanto in tanto per aggiornare i rating piu' recenti.
"""
import time
import urllib.parse

import pandas as pd
import requests

# Alcuni nomi di squadra usati da football-data.co.uk non coincidono con quelli
# di ClubElo: mappatura manuale per i casi noti.
MAPPATURA_NOMI = {
    "Spal": "SPAL",
}

def scarica_storico_squadra(squadra, tentativi=3):
    nome_clubelo = MAPPATURA_NOMI.get(squadra, squadra)
    url = f"http://api.clubelo.com/{urllib.parse.quote(nome_clubelo)}"
    for tentativo in range(tentativi):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.text.strip():
                righe = [riga for riga in r.text.splitlines() if riga.strip()]
                if len(righe) > 1:
                    from io import StringIO
                    df = pd.read_csv(StringIO(r.text))
                    if len(df) > 0:
                        return df
        except requests.RequestException:
            pass
        time.sleep(2)
    return None

def main():
    df_partite = pd.read_csv("serie_a.csv")
    squadre = sorted(set(df_partite["HomeTeam"]) | set(df_partite["AwayTeam"]))

    tutti_i_df = []
    non_trovate = []
    for i, squadra in enumerate(squadre):
        print(f"[{i+1}/{len(squadre)}] {squadra}...", end=" ")
        df = scarica_storico_squadra(squadra)
        if df is not None:
            df["Squadra"] = squadra
            tutti_i_df.append(df)
            print(f"OK ({len(df)} periodi)")
        else:
            non_trovate.append(squadra)
            print("NON TROVATA")
        time.sleep(1.5)  # rispetta il rate limit del servizio gratuito

    if tutti_i_df:
        elo_finale = pd.concat(tutti_i_df, ignore_index=True)
        elo_finale = elo_finale[["Squadra", "Elo", "From", "To"]]
        elo_finale.to_csv("elo_storico.csv", index=False)
        print(f"\nSalvato elo_storico.csv con {len(elo_finale):,} periodi per {len(tutti_i_df)} squadre.")

    if non_trovate:
        print(f"\nSquadre non trovate su ClubElo ({len(non_trovate)}): {non_trovate}")
        print("Per queste il modello userà un fallback (Elo medio di lega).")

if __name__ == "__main__":
    main()
