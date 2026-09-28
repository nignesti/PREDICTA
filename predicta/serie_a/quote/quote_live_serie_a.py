"""
Quote Serie A in tempo reale da The Odds API (v4) e ricerca di valore contro un
bookmaker "sharp".

Stessa logica di predicta/nba/quote/quote_live_nba.py (vedi quel modulo per il
perche'): la probabilita' "vera" si prende dal book piu' efficiente (Pinnacle),
ripulita dal margine con Shin, e si cercano i book che pagano una quota piu'
alta di quella equa.

Differenza dal basket: l'1X2 ha tre esiti (casa/pareggio/trasferta), non due,
quindi le funzioni di questo modulo lavorano su terne di probabilita' invece
che su una sola p1 (l'altra dedotta per complemento a 1). Il mercato "totals"
(Over/Under) resta a due esiti come nel basket.

Crediti: ogni chiamata a /odds costa (numero di mercati) x (numero di regioni).
Pinnacle sta nella regione "eu". Le chiamate senza partite in programma non
costano nulla. Nessuna chiamata parte da sola: la pagina la fa solo al clic,
con cache condivisa di 15 minuti (pages/3_Serie_A_Valore.py).

La chiave API si legge da st.secrets["ODDS_API_KEY"] o dalla variabile
d'ambiente ODDS_API_KEY (stesso secret usato per l'NBA): mai nel codice.

Nota: sport key e mapping SQUADRE verificati il 27/9/2026 contro /events
(chiamata gratuita) con le 20 squadre della stagione 2026/27 in calendario.
Una squadra non mappata (es. neopromossa non ancora vista) resta con il nome
originale di The Odds API, vedi quote_in_tabella: ricontrollare a ogni cambio
stagione.
"""
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from modello import probabilita_shin

URL_ODDS = "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/odds"
URL_EVENTI = "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/events"  # gratuito, niente quote
REGIONI = "eu"
MERCATI = ("h2h", "totals")
BOOK_SHARP = "pinnacle"
CARTELLA_SNAPSHOT = "quote_live"  # stessa cartella dell'NBA, prefisso file diverso (serie_a_*)

# Nomi di The Odds API -> nomi usati in serie_a.csv/pronostico.py. Le 20
# squadre della stagione 2026/27 (verificate contro /events il 27/9/2026); le
# altre restano invariate (vedi nota sopra).
SQUADRE = {
    "Atalanta BC": "Atalanta", "Bologna": "Bologna", "Cagliari": "Cagliari", "Como": "Como",
    "Fiorentina": "Fiorentina", "Frosinone": "Frosinone", "Genoa": "Genoa",
    "Inter Milan": "Inter", "Juventus": "Juventus", "Lazio": "Lazio", "Lecce": "Lecce",
    "AC Milan": "Milan", "Monza": "Monza", "Napoli": "Napoli", "Parma": "Parma",
    "AS Roma": "Roma", "Sassuolo": "Sassuolo", "Torino": "Torino", "Udinese": "Udinese",
    "Venezia": "Venezia",
}

# Esiti attesi per mercato, nell'ordine in cui probabilita_shin li riceve.
ESITI = {"h2h": ("casa", "pareggio", "trasferta"), "totals": ("over", "under")}
MIN_BOOK_CONSENSO = 3  # sotto questa soglia la mediana e' troppo esposta al singolo book fuori mercato


class ErroreQuoteLive(Exception):
    """Chiave mancante, crediti esauriti o risposta non valida dall'API."""


def chiave_api():
    try:
        import streamlit as st
        chiave = st.secrets.get("ODDS_API_KEY")
    except Exception:
        chiave = None
    return chiave or os.environ.get("ODDS_API_KEY")


def _iso(istante):
    return pd.Timestamp(istante).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def partite_in_programma(chiave, fino_a, da=None):
    """Numero di partite Serie A che iniziano fra 'da' e 'fino_a', da /events:
    non consuma crediti, quindi serve a decidere se vale la pena spendere
    crediti su /odds."""
    import requests

    parametri = {"apiKey": chiave, "commenceTimeTo": _iso(fino_a)}
    if da is not None:
        parametri["commenceTimeFrom"] = _iso(da)
    risposta = requests.get(URL_EVENTI, timeout=20, params=parametri)
    if not risposta.ok:
        raise ErroreQuoteLive(f"Errore The Odds API /events: HTTP {risposta.status_code} {risposta.text[:200]}")
    return len(risposta.json())


def scarica_quote(chiave, mercati=MERCATI, regioni=REGIONI, salva_snapshot=True, fino_a=None, da=None):
    """Una sola chiamata a /odds. Restituisce (eventi_json, crediti) dove
    crediti = {"rimanenti", "usati", "ultima"} dagli header di risposta."""
    import requests

    if not chiave:
        raise ErroreQuoteLive("Chiave API mancante: imposta ODDS_API_KEY in .streamlit/secrets.toml")
    parametri = {
        "apiKey": chiave, "regions": regioni, "markets": ",".join(mercati),
        "oddsFormat": "decimal", "dateFormat": "iso",
    }
    if fino_a is not None:
        parametri["commenceTimeTo"] = _iso(fino_a)
    if da is not None:
        parametri["commenceTimeFrom"] = _iso(da)
    risposta = requests.get(URL_ODDS, timeout=20, params=parametri)
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
        salva_snapshot_quote(quote_in_tabella(eventi))
    return eventi, crediti


def salva_snapshot_quote(tabella, istante=None, cartella=CARTELLA_SNAPSHOT):
    """Salva la tabella piatta (quote_in_tabella) con l'istante di raccolta:
    csv.gz come per l'NBA, prefisso 'serie_a_' cosi' i due sport convivono
    nella stessa cartella senza mescolarsi (vedi carica_snapshot)."""
    istante = pd.Timestamp(istante or datetime.now(timezone.utc)).tz_convert("UTC")
    os.makedirs(cartella, exist_ok=True)
    percorso = os.path.join(cartella, f"serie_a_{istante:%Y%m%dT%H%M%SZ}.csv.gz")
    tabella.assign(istante=istante).to_csv(percorso, index=False)
    return percorso


def carica_snapshot(cartella=CARTELLA_SNAPSHOT):
    """Tutti gli snapshot Serie A salvati, concatenati (colonna 'istante')."""
    import glob

    file = sorted(glob.glob(os.path.join(cartella, "serie_a_*.csv.gz")))
    if not file:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in file], ignore_index=True)
    df["inizio"] = pd.to_datetime(df["inizio"], utc=True)
    df["istante"] = pd.to_datetime(df["istante"], utc=True)
    return df


def quote_in_tabella(eventi):
    """Appiattisce la risposta in una riga per (partita, book, mercato, esito).
    L'1X2 e' espresso come "casa"/"pareggio"/"trasferta" (The Odds API usa
    "Draw" per il pareggio), il totale come "over"/"under"; 'linea' e' il
    totale di gol per il mercato totals, vuota per l'1X2."""
    righe = []
    for ev in eventi:
        casa, trasferta = ev["home_team"], ev["away_team"]
        lato = {casa: "casa", trasferta: "trasferta", "Draw": "pareggio", "Over": "over", "Under": "under"}
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


def _coppie_h2h(tabella):
    """Una riga per (partita, book) con le tre quote 1X2 affiancate."""
    t = tabella[tabella["mercato"] == "h2h"]
    chiave = ["id", "inizio", "Casa", "Trasferta", "book", "nome_book"]
    colonne_vuote = chiave + list(ESITI["h2h"])
    if t.empty:
        return pd.DataFrame(columns=colonne_vuote)
    largo = t.pivot_table(index=chiave, columns="esito", values="quota", aggfunc="first").reset_index()
    if not set(ESITI["h2h"]) <= set(largo.columns):
        return pd.DataFrame(columns=colonne_vuote)
    return largo.dropna(subset=list(ESITI["h2h"]))


def _coppie_totals(tabella):
    """Una riga per (partita, book, linea) con le quote over/under affiancate."""
    t = tabella[tabella["mercato"] == "totals"]
    chiave = ["id", "inizio", "Casa", "Trasferta", "book", "nome_book", "linea"]
    colonne_vuote = ["id", "inizio", "Casa", "Trasferta", "book", "nome_book", "linea_rif", "over", "under"]
    if t.empty:
        return pd.DataFrame(columns=colonne_vuote)
    largo = t.pivot_table(index=chiave, columns="esito", values="quota", aggfunc="first").reset_index()
    largo = largo.rename(columns={"linea": "linea_rif"})
    if not {"over", "under"} <= set(largo.columns):
        return pd.DataFrame(columns=colonne_vuote)
    return largo.dropna(subset=["over", "under"])


def _quote_mercato(tabella, mercato):
    """_coppie_h2h/_coppie_totals con una colonna 'linea_rif' e 'mercato'
    uniformi, cosi' probabilita_eque e trova_valore possono trattarli allo
    stesso modo."""
    largo = _coppie_h2h(tabella) if mercato == "h2h" else _coppie_totals(tabella)
    if largo.empty:
        return largo
    if "linea_rif" not in largo.columns:
        largo = largo.assign(linea_rif=0.0)
    else:
        largo = largo.assign(linea_rif=largo["linea_rif"].fillna(0.0))
    return largo.assign(mercato=mercato)


def probabilita_eque(tabella, book_sharp=BOOK_SHARP, min_book=MIN_BOOK_CONSENSO):
    """Probabilita' senza margine (Shin) di ogni esito di ogni mercato, come
    terna (h2h) o coppia (totals) nella colonna 'probabilita', nello stesso
    ordine di ESITI[mercato].

    Stessa logica di quote_live_nba.probabilita_eque (vedi quel modulo): fonte
    preferita il book sharp, altrimenti consenso dei book sulla linea piu'
    quotata se ne quotano almeno 'min_book'. Il consenso qui e' la media
    elementwise delle probabilita' Shin dei book, rinormalizzata a somma 1
    (con due soli esiti coincide con la mediana usata per l'NBA solo quando
    c'e' un unico book fuori norma; con tre esiti la mediana per componente
    non garantirebbe somma 1)."""
    colonne = ["id", "mercato", "linea_rif", "probabilita", "fonte", "n_book"]
    pezzi = []
    for mercato, esiti in ESITI.items():
        largo = _quote_mercato(tabella, mercato)
        if largo.empty:
            continue
        largo = largo.copy()
        largo["probabilita"] = [
            tuple(probabilita_shin([r[e] for e in esiti])) for r in largo[list(esiti)].to_dict("records")
        ]

        sharp = largo[largo["book"] == book_sharp][["id", "mercato", "linea_rif", "probabilita"]].assign(
            fonte=book_sharp, n_book=1)

        def _consenso(gruppo):
            valori = np.array(gruppo.tolist())
            media = valori.mean(axis=0)
            return tuple(media / media.sum())

        per_linea = largo.groupby(["id", "mercato", "linea_rif"], as_index=False).agg(
            probabilita=("probabilita", _consenso), n_book=("book", "nunique"))
        # linea modale: a parita' di book, la prima in ordine di linea (deterministico)
        per_linea = per_linea.sort_values(["id", "n_book", "linea_rif"], ascending=[True, False, True], kind="stable")
        consenso = per_linea.drop_duplicates(["id"])
        consenso = consenso[consenso["n_book"] >= min_book].assign(fonte="mediana book")

        senza_sharp = ~consenso["id"].isin(sharp["id"])
        pezzi.append(pd.concat([sharp, consenso[senza_sharp]], ignore_index=True)[colonne])
    if not pezzi:
        return pd.DataFrame(columns=colonne)
    return pd.concat(pezzi, ignore_index=True)


def trova_valore(tabella, soglia_ev=0.0, book_sharp=BOOK_SHARP, frazione_kelly=0.25):
    """Ogni quota di ogni book con valore atteso p_equa * quota - 1 sopra
    soglia_ev, confrontata sulla STESSA linea (per il totale). Esclude il book
    sharp stesso. 'Kelly' e' la frazione di bankroll suggerita (Kelly
    frazionario: il Kelly pieno presuppone probabilita' esatte, qui sono
    stime)."""
    eque = probabilita_eque(tabella, book_sharp)
    if eque.empty:
        return pd.DataFrame()

    righe = []
    for mercato, esiti in ESITI.items():
        largo = _quote_mercato(tabella, mercato)
        if largo.empty:
            continue
        largo = largo[largo["book"] != book_sharp]
        e = eque[eque["mercato"] == mercato]
        unite = largo.merge(e, on=["id", "mercato", "linea_rif"])
        for r in unite.itertuples(index=False):
            for i, esito in enumerate(esiti):
                quota = getattr(r, esito)
                p = r.probabilita[i]
                ev = p * quota - 1
                if ev > soglia_ev:
                    righe.append({
                        "id": r.id, "inizio": r.inizio, "Casa": r.Casa, "Trasferta": r.Trasferta,
                        "mercato": mercato, "esito": esito,
                        "linea": None if mercato == "h2h" else r.linea_rif,
                        "book": r.nome_book, "quota": quota, "quota_equa": 1 / p,
                        "p_equa": p, "EV": ev, "fonte": r.fonte,
                        "Kelly": frazione_kelly * ev / (quota - 1),
                    })
    if not righe:
        return pd.DataFrame()
    return pd.DataFrame(righe).sort_values("EV", ascending=False).reset_index(drop=True)


def quote_sharp_per_schedina(tabella, book_sharp=BOOK_SHARP, linea_totale=2.5):
    """Una riga per partita nel formato della tabella di pages/1_Serie_A_Schedina.py
    (colonne COLONNE), con le quote 1X2 e Over/Under 2.5 del book sharp: la
    schedina ne ricava le probabilita' con Shin, quindi parte dalle
    probabilita' piu' affidabili disponibili. Partite non quotate dal book
    sharp sul totale restano senza Over/Under."""
    h2h = _coppie_h2h(tabella)
    h2h = h2h[h2h["book"] == book_sharp]
    totali = _coppie_totals(tabella)
    totali = totali[(totali["book"] == book_sharp) & (totali["linea_rif"] == linea_totale)]

    righe = []
    for r in h2h.itertuples(index=False):
        riga = {
            "Casa": r.Casa, "Trasferta": r.Trasferta,
            "Quota 1": float(r.casa), "Quota X": float(r.pareggio), "Quota 2": float(r.trasferta),
        }
        t = totali[totali["id"] == r.id]
        if not t.empty:
            riga["Quota Over 2.5"] = float(t["over"].iloc[0])
            riga["Quota Under 2.5"] = float(t["under"].iloc[0])
        righe.append((r.inizio, riga))
    return pd.DataFrame([riga for _, riga in sorted(righe, key=lambda x: x[0])])
