"""
Valida il modello Elo NBA (modello_nba.py) contro un baseline ingenuo, con lo
stesso protocollo usato per il calcio (protocollo.py): Brier con bootstrap
appaiato come criterio primario, accuratezza con McNemar come descrittiva.
Nessun mercato di quote qui (nba_api non le fornisce): il confronto e' fra il
modello e "vince sempre la squadra di casa", il baseline piu' semplice che
usi comunque un'informazione reale (in NBA la squadra di casa vince ~57-60%
delle partite).

Split walk-forward: calibra la pendenza Elo->probabilita' sulle stagioni piu'
vecchie, valuta sulle ultime STAGIONI_TEST. Il rating stesso non ha bisogno di
split (e' gia' calcolato partita per partita senza guardare il futuro), solo
la calibrazione della regressione logistica lo richiede.
"""
import numpy as np
import pandas as pd

import modello_nba
import protocollo

STAGIONI_TEST = 3


def main():
    df = pd.read_csv("nba_storico.csv", parse_dates=["Date"])
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)

    df_elo, rating_finali = modello_nba.calcola_elo_storico(df)

    stagioni_ordinate = sorted(df_elo["Stagione"].unique())
    stagioni_test = set(stagioni_ordinate[-STAGIONI_TEST:])
    train = df_elo[~df_elo["Stagione"].isin(stagioni_test)]
    test = df_elo[df_elo["Stagione"].isin(stagioni_test)]
    print(f"Train: {len(train):,} partite ({len(stagioni_ordinate) - STAGIONI_TEST} stagioni)")
    print(f"Test:  {len(test):,} partite ({STAGIONI_TEST} stagioni: {sorted(stagioni_test)})")

    modello_calibrato = modello_nba.calibra_probabilita(
        train["EloDiff"], train["Winner"] == "H"
    )
    prob_elo = modello_nba.probabilita_calibrata(modello_calibrato, test["EloDiff"])

    tasso_casa_train = (train["Winner"] == "H").mean()
    prob_baseline = np.full(len(test), tasso_casa_train)

    esiti = (test["Winner"] == "H").to_numpy()
    esito = protocollo.confronta_binario(
        "Elo calibrato", prob_elo, "Solo vantaggio campo", prob_baseline, esiti
    )
    print()
    protocollo.riepiloga([esito])


if __name__ == "__main__":
    main()
