"""
Prototipo di modello a gradient boosting (Fase 2, punto 3 della ROADMAP), che usa
le stesse componenti già validate del modello statistico (xG storico/forma/
scontri diretti, rating Elo, probabilità implicite delle quote) come feature per
un classificatore HistGradientBoosting, invece di combinarle con una media
pesata scelta a mano.

Usiamo HistGradientBoostingClassifier di scikit-learn (già una dipendenza del
progetto) al posto di XGBoost: XGBoost richiede la libreria di sistema libomp,
non disponibile in questo ambiente, mentre HistGradientBoosting è concettualmente
lo stesso tipo di modello (alberi con istogrammi) e gestisce nativamente i NaN
(comodo per le partite senza quote o senza scontri diretti pregressi).

Le feature per OGNI partita (training e test) sono calcolate con la stessa
logica walk-forward point-in-time già validata in pages/backtesting.py: nessuna
approssimazione, nessun dato futuro rispetto alla partita da prevedere.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

import backtesting as bt
from modello import rps

N_STAGIONI_TRAINING = 10  # con 5 c'era overfitting severo; con 10 il divario train/test si riduce parecchio
N_PARTITE_TIRI = 5  # finestra per le medie recenti di tiri in porta / corner

COLONNE_FEATURE = ["xG_casa_storico", "xG_trasf_storico", "xG_casa_forma", "xG_trasf_forma",
                   "xG_casa_scontri", "xG_trasf_scontri", "elo_casa", "elo_trasferta", "elo_diff",
                   "prob_1_quote", "prob_X_quote", "prob_2_quote"]

COLONNE_FEATURE_TIRI = COLONNE_FEATURE + [
    "tiri_porta_fatti_casa", "tiri_porta_subiti_casa", "corner_fatti_casa", "corner_subiti_casa",
    "tiri_porta_fatti_trasf", "tiri_porta_subiti_trasf", "corner_fatti_trasf", "corner_subiti_trasf",
]


def calcola_media_recente(df, squadra, prima_di_idx, col_casa, col_trasferta, n=N_PARTITE_TIRI):
    """Media recente (ultime n partite, casa+trasferta) di una statistica di
    partita (tiri in porta, corner...) non ancora usata dal modello di
    produzione. col_casa è la colonna quando la squadra gioca in casa,
    col_trasferta quando gioca fuori (es. per "fatti": HST/AST; per "subiti":
    AST/HST, invertite). Stessa logica walk-forward di calcola_forma_bt."""
    df_prima = df.iloc[:prima_di_idx]
    casa = df_prima[df_prima["HomeTeam"] == squadra].tail(n)
    trasferta = df_prima[df_prima["AwayTeam"] == squadra].tail(n)
    valori = list(casa[col_casa].dropna()) + list(trasferta[col_trasferta].dropna())
    return np.mean(valori) if valori else np.nan


def componenti_in_dataframe(componenti):
    righe = []
    for c in componenti:
        righe.append({
            "xG_casa_storico": c["xG_casa_storico"], "xG_trasf_storico": c["xG_trasf_storico"],
            "xG_casa_forma": c["xG_casa_forma"], "xG_trasf_forma": c["xG_trasf_forma"],
            "xG_casa_scontri": c["xG_casa_scontri"] if c["scontri_validi"] else np.nan,
            "xG_trasf_scontri": c["xG_trasf_scontri"] if c["scontri_validi"] else np.nan,
            "elo_casa": c["elo_casa"], "elo_trasferta": c["elo_trasferta"],
            "elo_diff": (c["elo_casa"] - c["elo_trasferta"]) if (pd.notna(c["elo_casa"]) and pd.notna(c["elo_trasferta"])) else np.nan,
            "prob_1_quote": c["prob_1_quote"] if c["quote_presenti"] else np.nan,
            "prob_X_quote": c["prob_X_quote"] if c["quote_presenti"] else np.nan,
            "prob_2_quote": c["prob_2_quote"] if c["quote_presenti"] else np.nan,
            "tiri_porta_fatti_casa": c.get("tiri_porta_fatti_casa", np.nan),
            "tiri_porta_subiti_casa": c.get("tiri_porta_subiti_casa", np.nan),
            "corner_fatti_casa": c.get("corner_fatti_casa", np.nan),
            "corner_subiti_casa": c.get("corner_subiti_casa", np.nan),
            "tiri_porta_fatti_trasf": c.get("tiri_porta_fatti_trasf", np.nan),
            "tiri_porta_subiti_trasf": c.get("tiri_porta_subiti_trasf", np.nan),
            "corner_fatti_trasf": c.get("corner_fatti_trasf", np.nan),
            "corner_subiti_trasf": c.get("corner_subiti_trasf", np.nan),
            "esito": c["esito"],
        })
    return pd.DataFrame(righe)


def calcola_componenti_per_stagione(stagione_test, half_life=730, n_forma=3, con_tiri=False):
    """Rialloca train/test/statistiche del modulo backtesting per una singola
    stagione (walk-forward corretto: train = tutto ciò che precede) e restituisce
    le componenti per-partita di quella stagione. Se con_tiri=True, aggiunge le
    medie recenti di tiri in porta/corner (non ancora nel modello di produzione).

    Non usa bt.precompute_tutte così com'è: quella funzione scarta le partite
    senza storico valido, perdendo l'allineamento con l'indice originale di
    test_df di cui questa funzione ha bisogno per calcolare le feature sui tiri
    sullo stesso sottoinsieme di partite."""
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2
    bt.modello_elo_casa, bt.modello_elo_trasferta = bt.calibra_regressione_elo(bt.train_df, bt.elo_df)

    train_df, test_df = bt.train_df, bt.test_df
    elo_casa_arr = bt.elo_asof_batch(bt.elo_df, test_df["HomeTeam"], test_df["Date"])
    elo_trasf_arr = bt.elo_asof_batch(bt.elo_df, test_df["AwayTeam"], test_df["Date"])

    componenti = []
    for i in range(len(test_df)):
        comp = bt.precompute_componente(i, half_life, n_forma, elo_casa=elo_casa_arr[i], elo_trasferta=elo_trasf_arr[i])
        if comp is None:
            continue
        if con_tiri:
            riga = test_df.iloc[i]
            casa, trasferta = riga["HomeTeam"], riga["AwayTeam"]
            idx_globale = len(train_df) + i
            df_fino_a_ora = pd.concat([train_df, test_df.iloc[:i + 1]])
            comp.update(
                tiri_porta_fatti_casa=calcola_media_recente(df_fino_a_ora, casa, idx_globale, "HST", "AST"),
                tiri_porta_subiti_casa=calcola_media_recente(df_fino_a_ora, casa, idx_globale, "AST", "HST"),
                corner_fatti_casa=calcola_media_recente(df_fino_a_ora, casa, idx_globale, "HC", "AC"),
                corner_subiti_casa=calcola_media_recente(df_fino_a_ora, casa, idx_globale, "AC", "HC"),
                tiri_porta_fatti_trasf=calcola_media_recente(df_fino_a_ora, trasferta, idx_globale, "AST", "HST"),
                tiri_porta_subiti_trasf=calcola_media_recente(df_fino_a_ora, trasferta, idx_globale, "HST", "AST"),
                corner_fatti_trasf=calcola_media_recente(df_fino_a_ora, trasferta, idx_globale, "AC", "HC"),
                corner_subiti_trasf=calcola_media_recente(df_fino_a_ora, trasferta, idx_globale, "HC", "AC"),
            )
        componenti.append(comp)
    return componenti


def valuta_stagione(stagione_test, n_stagioni_training=N_STAGIONI_TRAINING, con_tiri=False):
    colonne = COLONNE_FEATURE_TIRI if con_tiri else COLONNE_FEATURE
    stagioni_disponibili = sorted(bt.df["Stagione"].astype(str).unique())
    idx_test = stagioni_disponibili.index(stagione_test)
    stagioni_training = stagioni_disponibili[max(0, idx_test - n_stagioni_training):idx_test]

    df_train = pd.concat([componenti_in_dataframe(calcola_componenti_per_stagione(s, con_tiri=con_tiri)) for s in stagioni_training], ignore_index=True)
    df_test = componenti_in_dataframe(calcola_componenti_per_stagione(stagione_test, con_tiri=con_tiri))

    modello = HistGradientBoostingClassifier(
        max_iter=100, max_depth=2, learning_rate=0.03, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=0,
    )
    modello.fit(df_train[colonne], df_train["esito"])

    classi = list(modello.classes_)
    probabilita_test = modello.predict_proba(df_test[colonne])
    predizioni = modello.predict(df_test[colonne])

    acc_train = accuracy_score(df_train["esito"], modello.predict(df_train[colonne]))
    acc = accuracy_score(df_test["esito"], predizioni)
    ll = log_loss(df_test["esito"], probabilita_test, labels=classi)
    idx_1, idx_2, idx_X = classi.index("1"), classi.index("2"), classi.index("X")
    rps_medio = np.mean([
        rps({"1": p[idx_1], "2": p[idx_2], "X": p[idx_X]}, r)
        for p, r in zip(probabilita_test, df_test["esito"])
    ])
    return acc, rps_medio, ll, acc_train, len(df_train), len(df_test)


if __name__ == "__main__":
    con_tiri = "--con-tiri" in sys.argv
    print(f"Feature tiri/corner: {'SI' if con_tiri else 'NO'}\n", flush=True)
    risultati = {}
    for stagione in ["2025", "2024", "2023"]:
        print(f"=== Stagione di test: {stagione} ===", flush=True)
        acc, rps_medio, ll, acc_train, n_train, n_test = valuta_stagione(stagione, con_tiri=con_tiri)
        risultati[stagione] = (acc, rps_medio, ll)
        print(f"  n_train={n_train} n_test={n_test} acc_train={acc_train:.1%}")
        print(f"  HistGradientBoosting: acc={acc:.1%} rps={rps_medio:.4f} logloss={ll:.4f}", flush=True)
        print()

    print("=== RIEPILOGO (confronta con il modello statistico: 54.87% acc, 0.1889 rps medi su 3 stagioni) ===")
    accs = [v[0] for v in risultati.values()]
    rpss = [v[1] for v in risultati.values()]
    for s, (acc, rps_medio, ll) in risultati.items():
        print(f"  {s}: acc={acc:.1%} rps={rps_medio:.4f} logloss={ll:.4f}")
    print(f"  MEDIA: acc={np.mean(accs):.2%} rps={np.mean(rpss):.4f}")
