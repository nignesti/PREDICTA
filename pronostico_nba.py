"""
Logica di previsione NBA per la singola partita: rating Elo (modello_nba.py)
per il moneyline, proiezione lineare del margine per lo spread, media dei
punti totali recenti per il totale. Separato dall'interfaccia
(pages/2_NBA.py) per lo stesso motivo di pronostico.py per il calcio: i test
possono importare il modello senza far girare la UI.

Le quote storiche vengono da nba_quote_storico.csv (nba_2010-2026.csv via
unisci_quote_nba.py), sempre presente nel repo: e' la stessa fonte dati usata
per il backtest (valida_nba.py), quindi lo storico Elo qui e quello validato
sono lo stesso. Se esiste anche nba_storico.csv (box score reali via
nba_api, scarica_nba.py + unisci_dati_nba.py) viene unito alle quote per
Stagione/PTS/Winner piu' precisi; altrimenti bastano le sole quote, che li
forniscono comunque (vedi unisci_quote_nba.carica_quote).
"""
import numpy as np
import pandas as pd
import streamlit as st

import modello_nba as mn
import valida_nba

PARTITE_FINESTRA_TOTALE = 10  # finestra per il ritmo di gioco recente (punti totali):
                              # piu' ampia della "forma" nel calcio (3) perche' qui non
                              # deve catturare la forza relativa (ci pensa gia' l'Elo),
                              # solo un riferimento di quanti punti si segnano in media.


class DatiNBANonDisponibili(Exception):
    """Sollevata quando manca anche nba_quote_storico.csv (es. repo
    incompleto): la pagina la intercetta e mostra le istruzioni per
    ricrearlo (unisci_quote_nba.py, a partire da nba_2010-2026.csv)."""


@st.cache_data
def load_data():
    try:
        df = valida_nba.carica_storico()
    except FileNotFoundError as errore:
        raise DatiNBANonDisponibili(
            f"Storico NBA non disponibile ({errore}). Ricrea nba_quote_storico.csv con:\n\n"
            f"`python unisci_quote_nba.py`\n\n"
            f"(richiede nba_2010-2026.csv nel repo)."
        )
    return df


@st.cache_resource
def prepara_modello():
    """Calcola una sola volta per sessione l'intero storico Elo (walk-forward,
    nessun lookahead per costruzione) e le calibrazioni che ne dipendono:
    costoso da rifare a ogni interazione con gli slider."""
    df = load_data()
    df_elo, rating_finali = mn.calcola_elo_storico(df)
    modello_moneyline = mn.calibra_probabilita(df_elo["EloDiff"], df_elo["Winner"] == "H")
    modello_margine, sigma_margine = mn.calibra_margine(
        df_elo["EloDiff"], df_elo["PTS_Home"] - df_elo["PTS_Away"])
    sigma_totale = float((df_elo["PTS_Home"] + df_elo["PTS_Away"]).std())
    return df_elo, rating_finali, modello_moneyline, modello_margine, sigma_margine, sigma_totale


def squadre_disponibili():
    df = load_data()
    return sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))


def _media_punti_totali(df_elo, squadra, ultime_n=PARTITE_FINESTRA_TOTALE):
    """Media dei punti totali (Casa+Trasferta) delle ultime `ultime_n` partite
    giocate da `squadra`, in casa o in trasferta indifferentemente: e' un
    riferimento di ritmo di gioco, non di forza, quindi non serve separare i
    due contesti come per il margine/moneyline."""
    casa = df_elo[df_elo["HomeTeam"] == squadra]
    trasferta = df_elo[df_elo["AwayTeam"] == squadra]
    partite = pd.concat([casa, trasferta]).sort_values("Date", kind="stable").tail(ultime_n)
    if partite.empty:
        return None
    return float((partite["PTS_Home"] + partite["PTS_Away"]).mean())


def stima_probabilita_nba(squadra_casa, squadra_trasferta):
    """Stime pure-modello per moneyline (probabilita' vittoria casa), spread
    (margine atteso + deviazione standard) e totale (punti attesi + deviazione
    standard). Restituisce None se una delle due squadre non e' nello storico."""
    df_elo, rating, modello_moneyline, modello_margine, sigma_margine, sigma_totale = prepara_modello()

    if squadra_casa not in rating or squadra_trasferta not in rating:
        return None

    elo_casa, elo_trasferta = rating[squadra_casa], rating[squadra_trasferta]
    elo_diff = elo_casa + mn.VANTAGGIO_CASA - elo_trasferta

    p_casa_modello = float(mn.probabilita_calibrata(modello_moneyline, [elo_diff])[0])
    margine_previsto = float(mn.margine_atteso(modello_margine, [elo_diff])[0])

    riferimenti_totale = [
        m for m in (_media_punti_totali(df_elo, squadra_casa), _media_punti_totali(df_elo, squadra_trasferta))
        if m is not None
    ]
    totale_previsto = (float(np.mean(riferimenti_totale)) if riferimenti_totale
                       else float((df_elo["PTS_Home"] + df_elo["PTS_Away"]).mean()))

    return {
        "elo_casa": elo_casa,
        "elo_trasferta": elo_trasferta,
        "p_casa_modello": p_casa_modello,
        "margine_atteso": margine_previsto,
        "sigma_margine": sigma_margine,
        "totale_atteso": totale_previsto,
        "sigma_totale": sigma_totale,
    }
