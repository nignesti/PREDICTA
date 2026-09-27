"""
Pulisce il CSV di quote NBA (spread/totale/moneyline, storico 2010-oggi) e lo
allinea al formato di nba_storico.csv (colonne Date, HomeTeam, AwayTeam con
abbreviazioni nba_api) cosi' che i due si possano unire su quella chiave.

La fonte delle quote usa abbreviazioni squadra proprietarie (es. "gs", "ny",
"utah") diverse da quelle di nba_api (es. "GSW", "NYK", "UTA"): serve una
mappa esplicita, non una semplice .upper().
"""
import pandas as pd

MAPPA_SQUADRE = {
    "atl": "ATL", "bkn": "BKN", "bos": "BOS", "cha": "CHA", "chi": "CHI",
    "cle": "CLE", "dal": "DAL", "den": "DEN", "det": "DET", "gs": "GSW",
    "hou": "HOU", "ind": "IND", "lac": "LAC", "lal": "LAL", "mem": "MEM",
    "mia": "MIA", "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHX", "por": "POR",
    "sa": "SAS", "sac": "SAC", "tor": "TOR", "utah": "UTA", "wsh": "WAS",
}

COLONNE_QUOTA = [
    "regular", "playoffs", "whos_favored", "spread", "total",
    "moneyline_away", "moneyline_home", "h2_spread", "h2_total",
]


def carica_quote(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["season"]).copy()

    squadre_ignote = (set(df["home"]) | set(df["away"])) - set(MAPPA_SQUADRE)
    if squadre_ignote:
        raise ValueError(f"Codici squadra non mappati: {sorted(squadre_ignote)}")

    df["Date"] = pd.to_datetime(df["date"])
    df["HomeTeam"] = df["home"].map(MAPPA_SQUADRE)
    df["AwayTeam"] = df["away"].map(MAPPA_SQUADRE)

    quote = df[["Date", "HomeTeam", "AwayTeam"] + COLONNE_QUOTA].copy()
    return quote.sort_values("Date", kind="stable").reset_index(drop=True)


def unisci_con_storico(quote, path_storico="nba_storico.csv"):
    """Left join di nba_storico.csv con le quote su (Date, HomeTeam, AwayTeam):
    le partite senza quota (es. fuori dal periodo 2010-oggi coperto dalla
    fonte) restano nel dataset con le colonne quota vuote."""
    storico = pd.read_csv(path_storico, parse_dates=["Date"])
    unito = storico.merge(quote, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    return unito


def main():
    quote = carica_quote("nba_2010-2026.csv")
    quote.to_csv("nba_quote_storico.csv", index=False)
    print(
        f"Creato nba_quote_storico.csv con {len(quote):,} partite "
        f"({quote['Date'].min().date()} - {quote['Date'].max().date()})"
    )


if __name__ == "__main__":
    main()
