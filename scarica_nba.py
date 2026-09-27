"""
Scarica lo storico delle partite NBA da stats.nba.com tramite nba_api (endpoint
LeagueGameLog), stessa logica di scarica_altre_leghe.py: un file grezzo per
stagione, cartella gitignored perche' rigenerabile.

A differenza di football-data.co.uk, nba_api restituisce una riga per SQUADRA
per partita (non una riga per partita con casa/trasferta gia' separati):
l'accoppiamento in partite viene fatto da unisci_dati_nba.py, non qui.

Nessuna quota di scommessa: stats.nba.com espone solo box score ufficiali.
Se in futuro servono le quote (moneyline/spread/totale), vanno prese da una
fonte diversa e unite su (Date, HomeTeam, AwayTeam).

Richiede il pacchetto nba_api (non ancora in requirements.txt):
    pip install nba_api
"""
import os
import time

STAGIONI = [f"{anno}-{str(anno + 1)[2:]}" for anno in range(1996, 2025)]
# 1996-97 e' la prima stagione con copertura affidabile di tutte le statistiche
# di squadra su stats.nba.com; prima di allora mancano colonne (es. FG3_PCT non
# esisteva ancora nella lega, tre punti introdotto nel 1979 ma i dati NBA online
# peggiorano molto andando indietro).
CARTELLA_BASE = "nba_stagioni"
PAUSA_SECONDI = 1.0  # cortesia verso stats.nba.com, stessa logica di scarica_elo.py


def scarica_stagione(stagione, tipo_stagione="Regular Season"):
    """Una riga per squadra per partita. player_or_team_abbreviation='T' chiede
    le statistiche di squadra (non dei singoli giocatori)."""
    from nba_api.stats.endpoints import leaguegamelog

    gamelog = leaguegamelog.LeagueGameLog(
        season=stagione,
        season_type_all_star=tipo_stagione,
        player_or_team_abbreviation="T",
        timeout=30,
    )
    return gamelog.get_data_frames()[0]


def main():
    os.makedirs(CARTELLA_BASE, exist_ok=True)
    scaricate, saltate = 0, 0
    for stagione in STAGIONI:
        percorso = os.path.join(CARTELLA_BASE, f"{stagione}.csv")
        if os.path.exists(percorso):
            saltate += 1
            continue
        try:
            df = scarica_stagione(stagione)
            if len(df) > 0:
                df.to_csv(percorso, index=False)
                scaricate += 1
                print(f"  {stagione}: OK ({len(df)} righe squadra-partita)")
            else:
                print(f"  {stagione}: nessun dato, salto")
        except Exception as e:
            print(f"  {stagione}: errore ({e}), salto")
        time.sleep(PAUSA_SECONDI)
    print(f"\nTotale: {scaricate} scaricate, {saltate} gia' presenti")


if __name__ == "__main__":
    main()
