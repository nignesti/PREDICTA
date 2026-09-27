"""
Raccolta programmata delle quote NBA (eseguita da GitHub Actions,
.github/workflows/quote_nba.yml): una chiamata a The Odds API, snapshot
salvato in quote_live/ e committato nel repo, cosi' lo storico delle quote
sopravvive ai riavvii di Streamlit Cloud ed e' leggibile da clv_nba.py e
dalla pagina "Quote live e valore".

Chiave da variabile d'ambiente ODDS_API_KEY (secret del repository).
Mercati da QUOTE_MERCATI (default h2h,spreads,totals: 3 crediti a chiamata).
Scarica solo le partite che iniziano entro QUOTE_ORIZZONTE_ORE (default 36):
servono le linee vicine alla palla a due, quelle con cui si misura il CLV,
non le linee aperte settimane prima. Prima controlla /events (gratuito): se
nell'orizzonte non c'e' nessuna partita non chiama /odds e non spende crediti.
"""
import os
import sys

import pandas as pd

import predicta.nba.quote.quote_live_nba as ql


def main():
    mercati = tuple(m for m in os.environ.get("QUOTE_MERCATI", ",".join(ql.MERCATI)).split(",") if m)
    chiave = os.environ.get("ODDS_API_KEY")
    fino_a = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=float(os.environ.get("QUOTE_ORIZZONTE_ORE", 36)))
    try:
        if not chiave:
            raise ql.ErroreQuoteLive("Chiave API mancante (secret ODDS_API_KEY)")
        n = ql.partite_in_programma(chiave, fino_a)
        if n == 0:
            print(f"Nessuna partita entro {fino_a:%d/%m %H:%M} UTC: nessun credito speso")
            return 0
        eventi, crediti = ql.scarica_quote(chiave, mercati=mercati, fino_a=fino_a)
    except ql.ErroreQuoteLive as errore:
        print(f"Errore: {errore}")
        return 1
    print(f"{len(eventi)} partite, crediti usati da questa chiamata: {crediti['ultima']}, "
          f"rimanenti: {crediti['rimanenti']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
