"""
Avviso "quote superate": aggiornamenti sugli infortuni NBA arrivati DOPO lo
scarico delle quote mostrate in pagina.

Perche' (valida_fonte_quote.py, docs/ROADMAP.md): la quota di chiusura non
lascia informazione residua, ma la quota di apertura e' misurabilmente peggiore.
L'unica leva e' il tempismo. Le notizie non correggono la quota (il mercato le
prezza in minuti): dicono solo quando la quota che si sta guardando e'
precedente alla notizia, quindi da riscaricare.

Fonte: endpoint JSON pubblico di ESPN (non documentato, gratuito, niente
chiave). Ogni voce ha la data dell'ultimo aggiornamento, che e' cio' che serve
per il confronto con l'istante delle quote. Se ESPN cambia formato o non
risponde, ErroreInfortuni: la pagina lo mostra e continua senza avviso.
"""
import pandas as pd

from predicta.nba.quote.quote_live_nba import SQUADRE

URL_INFORTUNI = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
# Una notizia uscita pochi minuti prima dello scarico puo' non essere ancora nel
# prezzo: la si segnala comunque. Manopola, non misurata.
MARGINE_MINUTI = 10
COLONNE = ["squadra", "giocatore", "stato", "aggiornato", "nota"]


class ErroreInfortuni(Exception):
    """ESPN non risponde o ha cambiato formato."""


def infortuni_in_tabella(dati):
    """Risposta ESPN -> una riga per giocatore, squadra come abbreviazione
    dello storico (stessa mappa di quote_live_nba)."""
    try:
        righe = [{
            "squadra": SQUADRE.get(squadra["displayName"], squadra["displayName"]),
            "giocatore": voce["athlete"]["displayName"],
            "stato": voce.get("status", ""),
            "aggiornato": pd.Timestamp(voce["date"]).tz_convert("UTC"),
            "nota": voce.get("shortComment", ""),
        } for squadra in dati["injuries"] for voce in squadra.get("injuries", [])]
    except (KeyError, TypeError, ValueError) as errore:
        raise ErroreInfortuni(f"Formato ESPN inatteso: {errore!r}") from errore
    return pd.DataFrame(righe, columns=COLONNE)


def scarica_infortuni():
    import requests

    try:
        risposta = requests.get(URL_INFORTUNI, timeout=20)
    except requests.RequestException as errore:
        raise ErroreInfortuni(f"ESPN non raggiungibile: {errore}") from errore
    if not risposta.ok:
        raise ErroreInfortuni(f"ESPN: HTTP {risposta.status_code}")
    return infortuni_in_tabella(risposta.json())


def notizie_dopo_quote(infortuni, partite, istante_quote, margine_minuti=MARGINE_MINUTI):
    """Aggiornamenti sulle squadre di 'partite' (colonne Casa, Trasferta,
    inizio) usciti dopo istante_quote - margine e prima dell'inizio della
    partita. Una riga per notizia, con la partita a cui si riferisce."""
    da = pd.Timestamp(istante_quote).tz_convert("UTC") - pd.Timedelta(minutes=margine_minuti)
    squadre = pd.concat([
        partite[["Casa", "Trasferta", "inizio"]].assign(squadra=partite["Casa"]),
        partite[["Casa", "Trasferta", "inizio"]].assign(squadra=partite["Trasferta"]),
    ])
    unite = infortuni.merge(squadre, on="squadra")
    unite = unite[(unite["aggiornato"] >= da) & (unite["aggiornato"] < unite["inizio"])]
    return unite.sort_values("aggiornato", ascending=False).reset_index(drop=True)
