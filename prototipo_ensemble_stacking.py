"""
Fase 3, punto 2 della ROADMAP: ensemble stacking tra il modello statistico
Dixon-Coles (storico/forma/scontri diretti + quote di chiusura con Shin) e il
prototipo gradient boosting (Fase 2, punto 3), con un meta-learner allenato
OUT-OF-FOLD sul training set (mai sulle stesse righe usate per allenare i due
modelli di base, altrimenti il meta-learner vedrebbe predizioni "facili" e
sovrastimerebbe il proprio beneficio).

Il meta-learner e' un HistGradientBoostingClassifier poco profondo (non una
regressione logistica/media pesata fissa) apposta per realizzare il
raffinamento suggerito da una revisione esterna e gia' annotato in ROADMAP:
non un peso fisso tipo "90% quote sempre", ma un blend condizionato al
contesto — un modello ad alberi puo' imparare da solo, dalle 6 probabilita'
di input (3 del modello statistico + 3 del gradient boosting), quando dare
piu' peso a quale componente, senza doverlo specificare a mano.

Elo non e' incluso come componente separata (gia' testato e risultato
negativo in Fase 2, incluso qui non aggiungerebbe altro che rumore).
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import backtesting as bt
import prototipo_gradient_boosting as pgb
from modello import rps

PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO = 0.10, 0.0, 0.90, -0.10
ORDINE_CLASSI = ["1", "X", "2"]
N_FOLD_OOF = 5


def _gb_base(random_state=0):
    return HistGradientBoostingClassifier(
        max_iter=100, max_depth=2, learning_rate=0.03, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=random_state,
    )


def _proba_ordinata(modello_gb, X):
    classi = list(modello_gb.classes_)
    p = modello_gb.predict_proba(X)
    idx_map = [classi.index(c) for c in ORDINE_CLASSI]
    return p[:, idx_map]


def prob_statistiche(comp_list):
    probs = []
    for comp in comp_list:
        r = bt.valuta_componente(comp, PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO)
        probs.append([r["1"], r["X"], r["2"]])
    return np.array(probs)


def prob_gb_oof(df_features, colonne):
    """Predizioni gradient boosting fuori-campione sullo stesso training set,
    con k-fold: necessarie per allenare il meta-learner senza data leakage."""
    kf = KFold(n_splits=N_FOLD_OOF, shuffle=True, random_state=0)
    oof = np.zeros((len(df_features), 3))
    for train_idx, val_idx in kf.split(df_features):
        modello_gb = _gb_base()
        modello_gb.fit(df_features.iloc[train_idx][colonne], df_features.iloc[train_idx]["esito"])
        oof[val_idx] = _proba_ordinata(modello_gb, df_features.iloc[val_idx][colonne])
    return oof


def valuta_stagione(stagione_test, n_stagioni_training=pgb.N_STAGIONI_TRAINING):
    stagioni_disponibili = sorted(bt.df["Stagione"].astype(str).unique())
    idx_test = stagioni_disponibili.index(stagione_test)
    stagioni_training = stagioni_disponibili[max(0, idx_test - n_stagioni_training):idx_test]

    comp_train = [c for s in stagioni_training for c in pgb.calcola_componenti_per_stagione(s)]
    comp_test = pgb.calcola_componenti_per_stagione(stagione_test)
    colonne = pgb.COLONNE_FEATURE

    df_train = pgb.componenti_in_dataframe(comp_train)
    df_test = pgb.componenti_in_dataframe(comp_test)

    p_stat_train = prob_statistiche(comp_train)
    p_stat_test = prob_statistiche(comp_test)

    p_gb_train_oof = prob_gb_oof(df_train, colonne)
    modello_gb_finale = _gb_base()
    modello_gb_finale.fit(df_train[colonne], df_train["esito"])
    p_gb_test = _proba_ordinata(modello_gb_finale, df_test[colonne])

    X_meta_train = np.hstack([p_stat_train, p_gb_train_oof])
    X_meta_test = np.hstack([p_stat_test, p_gb_test])
    y_train, y_test = df_train["esito"].to_numpy(), df_test["esito"].to_numpy()

    meta = _gb_base(random_state=1)
    meta.fit(X_meta_train, y_train)
    p_meta_test = _proba_ordinata(meta, X_meta_test)

    def metriche(probabilita, reali):
        predizioni = np.array(ORDINE_CLASSI)[np.argmax(probabilita, axis=1)]
        acc = accuracy_score(reali, predizioni)
        rps_medio = np.mean([rps({"1": p[0], "X": p[1], "2": p[2]}, r) for p, r in zip(probabilita, reali)])
        return acc, rps_medio

    return metriche(p_stat_test, y_test), metriche(p_gb_test, y_test), metriche(p_meta_test, y_test)


if __name__ == "__main__":
    print("Confronto: modello statistico da solo vs gradient boosting da solo vs ensemble stacking\n", flush=True)

    riepilogo = {"Statistico": [], "Gradient boosting": [], "Stacking": []}
    for stagione in ["2025", "2024", "2023"]:
        print(f"=== Stagione di test: {stagione} ===", flush=True)
        (acc_s, rps_s), (acc_g, rps_g), (acc_m, rps_m) = valuta_stagione(stagione)
        riepilogo["Statistico"].append((acc_s, rps_s))
        riepilogo["Gradient boosting"].append((acc_g, rps_g))
        riepilogo["Stacking"].append((acc_m, rps_m))
        print(f"  Statistico:        acc={acc_s:.2%} rps={rps_s:.4f}")
        print(f"  Gradient boosting: acc={acc_g:.2%} rps={rps_g:.4f}")
        print(f"  Stacking:          acc={acc_m:.2%} rps={rps_m:.4f}\n", flush=True)

    print("=== RIEPILOGO (confronta con il modello statistico: 54.87% acc, 0.1889 rps medi su 3 stagioni) ===")
    for nome, valori in riepilogo.items():
        accs = [v[0] for v in valori]; rpss = [v[1] for v in valori]
        print(f"  {nome}: acc medio={np.mean(accs):.2%} rps medio={np.mean(rpss):.4f}")
