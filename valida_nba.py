"""
Valida il modello Elo NBA (modello_nba.py) contro due baseline, con lo stesso
protocollo usato per il calcio (protocollo.py): Brier con bootstrap appaiato
come criterio primario, accuratezza con McNemar come descrittiva.

1. "Vince sempre la squadra di casa": baseline piu' semplice che usi comunque
   un'informazione reale (in NBA la squadra di casa vince ~57-60% delle
   partite).
2. Mercato (moneyline): ora disponibile grazie a nba_2010-2026.csv, quindi si
   puo' rifare per l'NBA la stessa domanda che il calcio ha gia' risposto
   (readme.md): il modello aggiunge qualcosa al mercato o e' solo rumore?

Split walk-forward: calibra la pendenza Elo->probabilita' sulle stagioni piu'
vecchie, valuta sulle ultime STAGIONI_TEST. Il rating stesso non ha bisogno di
split (e' gia' calcolato partita per partita senza guardare il futuro), solo
la calibrazione della regressione logistica lo richiede.
"""
import os

import numpy as np
import pandas as pd

import modello_nba
import protocollo
import unisci_quote_nba

STAGIONI_TEST = 3
COPERTURA_MINIMA_MERCATO = 0.9


def carica_storico():
    """Preferisce nba_storico.csv (box score reali via nba_api, se scaricato
    con scarica_nba.py + unisci_dati_nba.py) unito alle quote; se non esiste
    ancora usa il file derivato dalle sole quote (autosufficiente per
    l'Elo: ha gia' Stagione/PTS/Winner, vedi unisci_quote_nba.carica_quote)."""
    quote = unisci_quote_nba.carica_quote(unisci_quote_nba.FILE_GREZZO)
    if os.path.exists("nba_storico.csv"):
        df = unisci_quote_nba.unisci_con_storico(quote)
    else:
        df = quote
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def probabilita_mercato(moneyline_home, moneyline_away):
    """Converte moneyline americano in probabilita' implicita casa, rimuovendo
    il vig con normalizzazione proporzionale. Con due soli esiti (niente
    pareggio in NBA) la correzione di Shin usata per il calcio non aggiunge
    nulla: il favorite-longshot bias che corregge si manifesta a tre esiti."""
    def implicita(ml):
        ml = np.asarray(ml, dtype=float)
        negativo = ml < 0
        risultato = np.empty_like(ml)
        risultato[negativo] = -ml[negativo] / (-ml[negativo] + 100)
        risultato[~negativo] = 100.0 / (ml[~negativo] + 100)
        return risultato

    p_casa, p_trasferta = implicita(moneyline_home), implicita(moneyline_away)
    return p_casa / (p_casa + p_trasferta)


def stagioni_test_con_mercato(df_elo, n=STAGIONI_TEST, copertura_minima=COPERTURA_MINIMA_MERCATO):
    """Le ultime n stagioni con copertura moneyline sufficiente, non le ultime
    n a calendario: la fonte delle quote copre al 100% dal 2010 al 2022, poi
    scende al 50% nel 2023 e si azzera dal 2024 (probabile fine del feed) -
    prendere le ultime 3 a calendario metterebbe nel test proprio le stagioni
    senza mercato con cui confrontarsi."""
    copertura = df_elo.groupby("Stagione")["moneyline_home"].apply(lambda s: s.notna().mean())
    idonee = sorted(s for s in copertura.index if copertura[s] >= copertura_minima)
    scartate = sorted(set(copertura.index) - set(idonee))
    if scartate:
        print(f"Stagioni escluse per copertura moneyline < {copertura_minima:.0%}: {scartate}")
    return set(idonee[-n:])


def main():
    df = carica_storico()
    df_elo, rating_finali = modello_nba.calcola_elo_storico(df)

    stagioni_ordinate = sorted(df_elo["Stagione"].unique())
    stagioni_test = stagioni_test_con_mercato(df_elo)
    train = df_elo[~df_elo["Stagione"].isin(stagioni_test)]
    test = df_elo[df_elo["Stagione"].isin(stagioni_test)]
    print(f"Train: {len(train):,} partite ({len(stagioni_ordinate) - len(stagioni_test)} stagioni)")
    print(f"Test:  {len(test):,} partite ({len(stagioni_test)} stagioni: {sorted(stagioni_test)})")

    modello_calibrato = modello_nba.calibra_probabilita(
        train["EloDiff"], train["Winner"] == "H"
    )
    prob_elo = modello_nba.probabilita_calibrata(modello_calibrato, test["EloDiff"])

    tasso_casa_train = (train["Winner"] == "H").mean()
    prob_baseline = np.full(len(test), tasso_casa_train)

    esiti = (test["Winner"] == "H").to_numpy()
    esiti_da_confrontare = [
        protocollo.confronta_binario(
            "Elo calibrato", prob_elo, "Solo vantaggio campo", prob_baseline, esiti
        )
    ]

    con_quota = test["moneyline_home"].notna().to_numpy() & test["moneyline_away"].notna().to_numpy()
    if con_quota.any():
        test_quota = test[con_quota]
        prob_elo_quota = prob_elo[con_quota]
        prob_mercato = probabilita_mercato(test_quota["moneyline_home"], test_quota["moneyline_away"])
        esiti_quota = (test_quota["Winner"] == "H").to_numpy()
        esiti_da_confrontare.append(
            protocollo.confronta_binario(
                "Elo calibrato", prob_elo_quota, "Mercato (moneyline)", prob_mercato, esiti_quota
            )
        )
        print(f"({con_quota.sum():,}/{len(test):,} partite di test con moneyline disponibile)")

    print()
    protocollo.riepiloga(esiti_da_confrontare)


if __name__ == "__main__":
    main()
