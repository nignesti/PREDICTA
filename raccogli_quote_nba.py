"""
Raccolta programmata delle quote NBA (eseguita da GitHub Actions,
.github/workflows/quote_nba.yml): una chiamata a The Odds API, snapshot
salvato in quote_live/ e committato nel repo, cosi' lo storico delle quote
sopravvive ai riavvii di Streamlit Cloud ed e' leggibile da clv_nba.py e
dalla pagina "Quote live e valore".

Chiave da variabile d'ambiente ODDS_API_KEY (secret del repository).
Mercati da QUOTE_MERCATI (default h2h,spreads,totals: 3 crediti a chiamata).
Senza partite in programma non salva nulla e non consuma crediti.
"""
import os
import sys

import quote_live_nba as ql


def main():
    mercati = tuple(m for m in os.environ.get("QUOTE_MERCATI", ",".join(ql.MERCATI)).split(",") if m)
    try:
        eventi, crediti = ql.scarica_quote(os.environ.get("ODDS_API_KEY"), mercati=mercati)
    except ql.ErroreQuoteLive as errore:
        print(f"Errore: {errore}")
        return 1
    print(f"{len(eventi)} partite, crediti usati da questa chiamata: {crediti['ultima']}, "
          f"rimanenti: {crediti['rimanenti']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
