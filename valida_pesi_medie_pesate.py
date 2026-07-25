"""
Grid search sui pesi del blend CON le medie di lega pesate nel tempo
(vedi valida_medie_lega.py).

Perche' serve rifare la ricerca. I pesi attuali (forma=0.10, quote=0.90) sono
stati trovati con una grid search fatta quando il modello statistico soffriva del
bias pro-casa descritto in valida_medie_lega.py: dava "1" nel 76.9% delle partite
contro il 72.2% dopo la correzione, e valeva 48.55% di accuratezza contro 49.96%.
Un peso quote cosi' alto e' anche il modo in cui la grid search compensava un
modello base distorto. Con il modello base migliorato l'ottimo puo' essersi
spostato, e tenere i vecchi pesi butterebbe via il guadagno.

Protocollo identico agli altri script di validazione: 3 stagioni di test
indipendenti (2023, 2024, 2025) walk-forward, Shin + quote di chiusura. Le
componenti sono precalcolate UNA VOLTA per stagione e riusate per tutte le
combinazioni di pesi (e' esattamente lo scopo della separazione
precompute_componente / valuta_componente), quindi la griglia costa poco.

Criterio di adozione, coerente con il resto del progetto: si adotta solo un
guadagno pulito o quasi su ENTRAMBE le metriche (accuratezza e RPS), e coerente
fra le stagioni. E' il criterio che ha fatto adottare Shin e le quote di chiusura
e scartare ensemble stacking e multi-campionato.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
from sklearn.metrics import accuracy_score

import backtesting as bt
from modello import rps

RHO = -0.10
EMIVITA_GIORNI, N_PARTITE_FORMA = 730, 3
STAGIONI = ["2025", "2024", "2023"]

# Griglia: peso forma e peso quote. peso_scontri resta 0 (validato negativo piu'
# volte); peso_storico e' il residuo 1 - forma - quote.
GRIGLIA_FORMA = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
GRIGLIA_QUOTE = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def prepara_stagione(stagione_test, medie_lega):
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2
    return bt.precompute_tutte(EMIVITA_GIORNI, N_PARTITE_FORMA, metodo_quote="shin",
                               fonte_quote="chiusura", medie_lega=medie_lega)


def metriche(componenti, pf, pq):
    predizioni, reali, _, probabilita = bt.valuta_tutte(componenti, pf, 0.0, pq, RHO)
    acc = accuracy_score(reali, predizioni)
    rps_medio = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita, reali)])
    return acc, rps_medio


if __name__ == "__main__":
    for medie_lega in ["storiche", "pesate"]:
        print(f"\n{'=' * 78}")
        print(f"GRID SEARCH — medie di lega: {medie_lega.upper()}  (Shin + quote di chiusura)")
        print(f"{'=' * 78}\n", flush=True)

        print("Precalcolo componenti per stagione...", flush=True)
        componenti_per_stagione = {s: prepara_stagione(s, medie_lega) for s in STAGIONI}

        risultati = []
        for pf in GRIGLIA_FORMA:
            for pq in GRIGLIA_QUOTE:
                if pf + pq > 1.0:
                    continue
                accs, rpss = [], []
                for s in STAGIONI:
                    a, r = metriche(componenti_per_stagione[s], pf, pq)
                    accs.append(a); rpss.append(r)
                risultati.append({
                    "forma": pf, "quote": pq, "storico": round(1 - pf - pq, 2),
                    "acc": float(np.mean(accs)), "rps": float(np.mean(rpss)),
                    "acc_per_stagione": [float(a) for a in accs],
                })

        per_acc = sorted(risultati, key=lambda r: -r["acc"])
        per_rps = sorted(risultati, key=lambda r: r["rps"])

        print("\n  Migliori 8 per ACCURATEZZA media:")
        print(f"    {'forma':>6} {'quote':>6} {'stor.':>6} | {'acc':>7} {'rps':>7} | per stagione (2025/2024/2023)")
        for r in per_acc[:8]:
            st = "/".join(f"{a:.1%}" for a in r["acc_per_stagione"])
            print(f"    {r['forma']:>6.2f} {r['quote']:>6.2f} {r['storico']:>6.2f} | "
                  f"{r['acc']:>6.2%} {r['rps']:>7.4f} | {st}")

        print("\n  Migliori 5 per RPS medio:")
        for r in per_rps[:5]:
            print(f"    {r['forma']:>6.2f} {r['quote']:>6.2f} {r['storico']:>6.2f} | "
                  f"{r['acc']:>6.2%} {r['rps']:>7.4f}")

        attuale = next(r for r in risultati if r["forma"] == 0.10 and r["quote"] == 0.90)
        st = "/".join(f"{a:.1%}" for a in attuale["acc_per_stagione"])
        print(f"\n  Configurazione ATTUALE (forma=0.10, quote=0.90): "
              f"acc={attuale['acc']:.2%} rps={attuale['rps']:.4f} | {st}", flush=True)
