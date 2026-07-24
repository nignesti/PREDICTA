"""
Valida le quote di CHIUSURA (a ridosso del fischio d'inizio, colonne
OddsAvgCH/CD/CA) come sostituto delle quote di APERTURA (OddsAvgH/D/A, colonna
"quote" usata finora) nel blend statistico (Fase 2, punto 6 della ROADMAP),
con lo stesso protocollo a 3 stagioni indipendenti già usato per Elo, gradient
boosting e Shin: 2023, 2024, 2025 come test walk-forward, pesi fissi alla
configurazione già validata (forma=0.10, scontri diretti=0, quote=0.90),
metodo di conversione quote->probabilità fissato a Shin (già validato positivo).

Le quote di chiusura sono presenti nei file grezzi solo dal 2019 in poi (7
stagioni su 33): le 3 stagioni di test (2023/2024/2025) sono tutte coperte,
quindi il confronto è pulito. Come per Shin, non serve nessun addestramento:
cambia solo QUALE quota si usa come input al blend (di mercato più aggiornata
a ridosso del match, che incorpora più informazione, invece di quella fissata
all'apertura delle contrattazioni).
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
from sklearn.metrics import accuracy_score

import backtesting as bt
from modello import rps

PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO = 0.10, 0.0, 0.90, -0.10
EMIVITA_GIORNI, N_PARTITE_FORMA = 730, 3


def valuta_stagione(stagione_test, fonte_quote):
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2
    bt.modello_elo_casa, bt.modello_elo_trasferta = bt.calibra_regressione_elo(bt.train_df, bt.elo_df) if bt.ELO_DISPONIBILE else (None, None)

    componenti = bt.precompute_tutte(EMIVITA_GIORNI, N_PARTITE_FORMA, metodo_quote="shin", fonte_quote=fonte_quote)
    predizioni, reali, _, probabilita = bt.valuta_tutte(componenti, PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO)
    acc = accuracy_score(reali, predizioni)
    rps_medio = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita, reali)])
    quote_chiusura_presenti = sum(1 for c in componenti if c["quote_presenti"])
    return acc, rps_medio, quote_chiusura_presenti, len(componenti)


if __name__ == "__main__":
    print("Confronto fonte quote: apertura (OddsAvgH/D/A) vs chiusura (OddsAvgCH/CD/CA)")
    print(f"Pesi fissi: forma={PESO_FORMA}, scontri={PESO_SCONTRI}, quote={PESO_QUOTE}, rho={RHO}, metodo=Shin\n", flush=True)

    riepilogo = {}
    for fonte in ["apertura", "chiusura"]:
        print(f"=== Fonte quote: {fonte} ===", flush=True)
        accs, rpss = [], []
        for stagione in ["2025", "2024", "2023"]:
            acc, rps_medio, n_con_quote, n_tot = valuta_stagione(stagione, fonte)
            accs.append(acc); rpss.append(rps_medio)
            print(f"  {stagione}: acc={acc:.2%} rps={rps_medio:.4f} (quote presenti: {n_con_quote}/{n_tot})", flush=True)
        riepilogo[fonte] = (np.mean(accs), np.mean(rpss))
        print(f"  MEDIA: acc={riepilogo[fonte][0]:.2%} rps={riepilogo[fonte][1]:.4f}\n", flush=True)

    print("=== RIEPILOGO FINALE ===")
    for fonte, (acc, rps_medio) in riepilogo.items():
        print(f"  {fonte}: acc={acc:.2%} rps={rps_medio:.4f}")
