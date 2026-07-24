"""
Valida la correzione di Shin (1992/1993) per le quote (Fase 2, punto 2 della
ROADMAP) rispetto alla normalizzazione proporzionale attuale (1/quota poi
rinormalizzata), con lo stesso protocollo già usato per Elo e gradient
boosting: 3 stagioni di test indipendenti (2023, 2024, 2025), walk-forward
(training = tutte le stagioni precedenti), pesi fissi alla configurazione già
validata in Fase 1 (forma=0.10, scontri diretti=0, quote=0.90).

A differenza di Elo e gradient boosting, Shin non introduce nessun dato nuovo
né alcun addestramento: cambia solo COME si tolgono le probabilità vere dalle
quote del bookmaker già in uso. Se il segnale di mercato in ingresso è più
pulito, il beneficio dovrebbe propagarsi al blend finale (pesato 90% sulle
quote).
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


def valuta_stagione(stagione_test, metodo_quote):
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2
    bt.modello_elo_casa, bt.modello_elo_trasferta = bt.calibra_regressione_elo(bt.train_df, bt.elo_df)

    componenti = bt.precompute_tutte(EMIVITA_GIORNI, N_PARTITE_FORMA, metodo_quote=metodo_quote)
    predizioni, reali, _, probabilita = bt.valuta_tutte(componenti, PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO)
    acc = accuracy_score(reali, predizioni)
    rps_medio = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita, reali)])
    return acc, rps_medio


if __name__ == "__main__":
    print("Confronto metodo di conversione quote: proporzionale (1/quota) vs Shin (1992/1993)")
    print(f"Pesi fissi: forma={PESO_FORMA}, scontri={PESO_SCONTRI}, quote={PESO_QUOTE}, rho={RHO}\n", flush=True)

    riepilogo = {}
    for metodo in ["proporzionale", "shin"]:
        print(f"=== Metodo: {metodo} ===", flush=True)
        accs, rpss = [], []
        for stagione in ["2025", "2024", "2023"]:
            acc, rps_medio = valuta_stagione(stagione, metodo)
            accs.append(acc); rpss.append(rps_medio)
            print(f"  {stagione}: acc={acc:.2%} rps={rps_medio:.4f}", flush=True)
        riepilogo[metodo] = (np.mean(accs), np.mean(rpss))
        print(f"  MEDIA: acc={riepilogo[metodo][0]:.2%} rps={riepilogo[metodo][1]:.4f}\n", flush=True)

    print("=== RIEPILOGO FINALE ===")
    for metodo, (acc, rps_medio) in riepilogo.items():
        print(f"  {metodo}: acc={acc:.2%} rps={rps_medio:.4f}")
