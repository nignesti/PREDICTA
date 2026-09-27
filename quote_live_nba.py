"""
Quote NBA in tempo reale da The Odds API (v4) e ricerca di valore contro un
bookmaker "sharp".

Perche': valida_nba.py mostra che nessun modello costruito sui soli risultati
batte le quote di chiusura. L'unico vantaggio documentato per chi scommette
senza informazioni privilegiate e' il confronto fra bookmaker: la probabilita'
"vera" si prende dal book piu' efficiente (Pinnacle, margine basso e limiti
alti: le sue linee si muovono sui soldi informati), ripulita dal margine con
Shin come nel resto del progetto, e si cercano i book che pagano una quota
piu' alta di quella equa.

Crediti: ogni chiamata a /odds costa (numero di mercati) x (numero di regioni).
Pinnacle sta nella regione "eu", quindi il default (eu, 3 mercati) costa 3
crediti. Le chiamate senza partite in programma non costano nulla (documentato
da The Odds API). Nessuna chiamata parte da sola: la pagina la fa solo al clic,
con cache condivisa di 15 minuti (pages/2_NBA_Valore.py).

La chiave API si legge da st.secrets["ODDS_API_KEY"] o dalla variabile
d'ambiente ODDS_API_KEY: mai nel codice (vedi .streamlit/secrets.toml, ignorato
da git).
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from modello import probabilita_shin

URL_ODDS = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
REGIONI = "eu"
MERCATI = ("h2h", "spreads", "totals")
BOOK_SHARP = "pinnacle"
CARTELLA_SNAPSHOT = "quote_live"  # uno snapshot JSON per chiamata: base per misurare il CLV in futuro

# Nomi completi di The Odds API -> abbreviazioni usate nello storico
# (unisci_quote_nba.MAPPA_SQUADRE).
SQUADRE = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


class ErroreQuoteLive(Exception):
    """Chiave mancante, crediti esauriti o risposta non valida dall'API."""


def chiave_api():
    try:
        import streamlit as st
        chiave = st.secrets.get("ODDS_API_KEY")
    except Exception:
        chiave = None
    return chiave or os.environ.get("ODDS_API_KEY")


def scarica_quote(chiave, mercati=MERCATI, regioni=REGIONI, salva_snapshot=True):
    """Una sola chiamata a /odds. Restituisce (eventi_json, crediti) dove
    crediti = {"rimanenti", "usati", "ultima"} dagli header di risposta."""
    import requests

    if not chiave:
        raise ErroreQuoteLive("Chiave API mancante: imposta ODDS_API_KEY in .streamlit/secrets.toml")
    risposta = requests.get(URL_ODDS, timeout=20, params={
        "apiKey": chiave, "regions": regioni, "markets": ",".join(mercati),
        "oddsFormat": "decimal", "dateFormat": "iso",
    })
    if risposta.status_code == 401:
        raise ErroreQuoteLive("Chiave API non valida o crediti esauriti (HTTP 401)")
    if risposta.status_code == 429:
        raise ErroreQuoteLive("Troppe richieste o quota mensile esaurita (HTTP 429)")
    if not risposta.ok:
        raise ErroreQuoteLive(f"Errore The Odds API: HTTP {risposta.status_code} {risposta.text[:200]}")

    crediti = {
        "rimanenti": risposta.headers.get("x-requests-remaining"),
        "usati": risposta.headers.get("x-requests-used"),
        "ultima": risposta.headers.get("x-requests-last"),
    }
    eventi = risposta.json()
    if salva_snapshot and eventi:
        os.makedirs(CARTELLA_SNAPSHOT, exist_ok=True)
        istante = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with open(os.path.join(CARTELLA_SNAPSHOT, f"nba_{istante}.json"), "w") as f:
            json.dump(eventi, f)
    return eventi, crediti


def quote_in_tabella(eventi):
    """Appiattisce la risposta in una riga per (partita, book, mercato, esito).
    Gli esiti di moneyline e spread sono espressi come "casa"/"trasferta",
    quelli del totale come "over"/"under"; 'linea' e' lo spread della squadra
    dell'esito (negativo se favorita) o il totale."""
    righe = []
    for ev in eventi:
        casa, trasferta = ev["home_team"], ev["away_team"]
        lato = {casa: "casa", trasferta: "trasferta", "Over": "over", "Under": "under"}
        for book in ev.get("bookmakers", []):
            for mercato in book.get("markets", []):
                for esito in mercato.get("outcomes", []):
                    if esito["name"] not in lato:
                        continue
                    righe.append({
                        "id": ev["id"],
                        "inizio": pd.Timestamp(ev["commence_time"]),
                        "Casa": SQUADRE.get(casa, casa),
                        "Trasferta": SQUADRE.get(trasferta, trasferta),
                        "book": book["key"],
                        "nome_book": book.get("title", book["key"]),
                        "mercato": mercato["key"],
                        "esito": lato[esito["name"]],
                        "linea": esito.get("point"),
                        "quota": float(esito["price"]),
                    })
    colonne = ["id", "inizio", "Casa", "Trasferta", "book", "nome_book", "mercato", "esito", "linea", "quota"]
    return pd.DataFrame(righe, columns=colonne)


ESITI = {"h2h": ("casa", "trasferta"), "spreads": ("casa", "trasferta"), "totals": ("over", "under")}


def _coppie(tabella):
    """Una riga per (partita, book, mercato, linea) con le due quote affiancate.
    La linea di riferimento e' quella del primo esito (casa per lo spread,
    identica per over/under); per lo spread la trasferta ha la linea opposta."""
    t = tabella.copy()
    primo = t.apply(lambda r: r["esito"] == ESITI[r["mercato"]][0], axis=1)
    t["linea_rif"] = np.where(primo | (t["mercato"] != "spreads"), t["linea"], -t["linea"])
    t["lato"] = np.where(primo, "q1", "q2")
    chiave = ["id", "inizio", "Casa", "Trasferta", "book", "nome_book", "mercato", "linea_rif"]
    t["linea_rif"] = t["linea_rif"].fillna(0.0)
    coppie = t.pivot_table(index=chiave, columns="lato", values="quota", aggfunc="first").reset_index()
    return coppie.dropna(subset=["q1", "q2"]) if {"q1", "q2"} <= set(coppie.columns) else coppie.iloc[0:0]


def probabilita_eque(tabella, book_sharp=BOOK_SHARP):
    """Probabilita' senza margine (Shin) del primo esito di ogni mercato,
    secondo il book sharp. Se il book sharp non quota una partita/mercato, la
    moneyline ripiega sulla mediana delle probabilita' senza margine di tutti
    i book (per spread e totale no: le linee dei book differiscono e una
    mediana fra linee diverse non ha senso). Colonna 'fonte' per trasparenza."""
    coppie = _coppie(tabella)
    if coppie.empty:
        return pd.DataFrame(columns=["id", "mercato", "linea_rif", "p1", "fonte"])
    coppie["p1"] = [probabilita_shin([a, b])[0] for a, b in zip(coppie["q1"], coppie["q2"])]

    sharp = coppie[coppie["book"] == book_sharp][["id", "mercato", "linea_rif", "p1"]].assign(fonte=book_sharp)
    ml = coppie[coppie["mercato"] == "h2h"]
    consenso = ml.groupby(["id", "mercato", "linea_rif"], as_index=False)["p1"].median().assign(fonte="mediana book")
    senza_sharp = ~consenso.set_index(["id", "mercato"]).index.isin(sharp.set_index(["id", "mercato"]).index)
    return pd.concat([sharp, consenso[senza_sharp]], ignore_index=True)


def trova_valore(tabella, soglia_ev=0.0, book_sharp=BOOK_SHARP, frazione_kelly=0.25):
    """Ogni quota di ogni book con valore atteso p_equa * quota - 1 sopra
    soglia_ev, dove p_equa viene da probabilita_eque sulla STESSA linea (uno
    spread -3.5 non si confronta con un -4). Esclude il book sharp stesso.
    'Kelly' e' la frazione di bankroll suggerita (Kelly frazionario: il Kelly
    pieno presuppone probabilita' esatte, qui sono stime)."""
    eque = probabilita_eque(tabella, book_sharp)
    coppie = _coppie(tabella)
    if eque.empty or coppie.empty:
        return pd.DataFrame()
    unite = coppie.merge(eque, on=["id", "mercato", "linea_rif"])
    unite = unite[unite["book"] != book_sharp]

    righe = []
    for r in unite.itertuples(index=False):
        primo, secondo = ESITI[r.mercato]
        for esito, quota, p in ((primo, r.q1, r.p1), (secondo, r.q2, 1 - r.p1)):
            ev = p * quota - 1
            if ev > soglia_ev:
                righe.append({
                    "inizio": r.inizio, "Casa": r.Casa, "Trasferta": r.Trasferta,
                    "mercato": r.mercato, "esito": esito,
                    "linea": _linea_esito(r.mercato, esito, r.linea_rif),
                    "book": r.nome_book, "quota": quota, "quota_equa": 1 / p,
                    "p_equa": p, "EV": ev, "fonte": r.fonte,
                    "Kelly": frazione_kelly * ev / (quota - 1),
                })
    if not righe:
        return pd.DataFrame()
    return pd.DataFrame(righe).sort_values("EV", ascending=False).reset_index(drop=True)


def _linea_esito(mercato, esito, linea_rif):
    if mercato == "h2h":
        return None
    if mercato == "spreads" and esito == "trasferta":
        return -linea_rif
    return linea_rif


def quote_sharp_per_schedina(tabella, book_sharp=BOOK_SHARP):
    """Una riga per partita nel formato della tabella di pages/2_NBA_Schedina.py
    (colonne COLONNE), con le quote del book sharp: la schedina ne ricava le
    probabilita' con Shin, quindi parte dalle probabilita' piu' affidabili
    disponibili. Mercati non quotati dal book sharp restano vuoti."""
    coppie = _coppie(tabella)
    coppie = coppie[coppie["book"] == book_sharp]
    righe = []
    for (id_, inizio, casa, trasferta), gruppo in coppie.groupby(["id", "inizio", "Casa", "Trasferta"], sort=False):
        riga = {"Casa": casa, "Trasferta": trasferta}
        for mercato, (c1, c2, col_linea) in {
            "h2h": ("Quota vittoria Casa", "Quota vittoria Trasferta", None),
            "spreads": ("Quota spread Casa", "Quota spread Trasferta", "Linea spread (Casa)"),
            "totals": ("Quota Over", "Quota Under", "Linea totale"),
        }.items():
            m = gruppo[gruppo["mercato"] == mercato]
            if not m.empty:
                riga[c1], riga[c2] = float(m["q1"].iloc[0]), float(m["q2"].iloc[0])
                if col_linea:
                    riga[col_linea] = float(m["linea_rif"].iloc[0])
        righe.append((inizio, riga))
    return pd.DataFrame([r for _, r in sorted(righe, key=lambda x: x[0])])
