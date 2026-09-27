"""
Raccolta programmata delle quote Serie A (eseguita da GitHub Actions,
.github/workflows/quote_serie_a.yml): una chiamata a The Odds API, snapshot
salvato in quote_live/ e committato nel repo, cosi' lo storico delle quote
sopravvive ai riavvii di Streamlit Cloud ed e' leggibile da clv_serie_a.py e
dalla pagina "Valore Serie A".

Chiave da variabile d'ambiente ODDS_API_KEY (secret del repository, condiviso
con l'NBA). Mercati da QUOTE_MERCATI (default h2h,totals: 2 crediti a
chiamata). Scarica solo le partite che iniziano entro QUOTE_ORIZZONTE_ORE
(default 36). Prima controlla /events (gratuito): se nell'orizzonte non c'e'
nessuna partita non chiama /odds e non spende crediti.
"""
import os
import sys

import pandas as pd

import predicta.serie_a.quote.quote_live_serie_a as ql


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
