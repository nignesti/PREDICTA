"""
Fase 0: rivalutazione dell'estensione multi-campionato (Fase 3, punto 3) con il
protocollo di misura corretto (protocollo.py).

L'esperimento era stato archiviato come "positivo sulla calibrazione, neutro
sull'accuratezza, non adottato": l'RPS migliorava in tutte e 3 le stagioni
(0.1884 contro 0.1896 con solo Serie A, meglio persino del modello statistico
0.1889) ma l'accuratezza restava sotto, e il criterio dell'epoca chiedeva un
guadagno su entrambe le metriche.

Quel criterio era sbagliato: l'accuratezza non aveva la potenza per bocciare
nulla, mentre l'RPS — l'unica metrica con potenza a questi campioni — dava il
multi-campionato avanti in 3 stagioni su 3. Qui si rifa' il confronto con il
bootstrap appaiato sull'RPS e la finestra a 7 stagioni, e con la configurazione
delle quote allineata alla produzione (vedi la nota in
prototipo_gradient_boosting.py).
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import backtesting as bt
import prototipo_gradient_boosting as pgb
import prototipo_gradient_boosting_multiliga as pml
import protocollo

STAGIONI = protocollo.STAGIONI_TEST
ORDINE = list(protocollo.ORDINE_CLASSI)


def _gb():
    return HistGradientBoostingClassifier(
        max_iter=100, max_depth=2, learning_rate=0.03, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=0,
    )


def _proba_ordinata(modello, X):
    classi = list(modello.classes_)
    p = modello.predict_proba(X)
    return p[:, [classi.index(c) for c in ORDINE]]


def raccogli(stagioni, pool_altre_leghe):
    """Probabilita' per-partita dei due gradient boosting (training solo Serie A
    contro training esteso alle altre 4 leghe) sulle stesse partite di test."""
    p_solo, p_esteso, reali = [], [], []
    stagioni_disponibili = sorted(bt.df["Stagione"].astype(str).unique())
    colonne = pgb.COLONNE_FEATURE

    for stagione in stagioni:
        print(f"  stagione {stagione}...", flush=True)
        idx = stagioni_disponibili.index(stagione)
        stagioni_training = stagioni_disponibili[max(0, idx - pgb.N_STAGIONI_TRAINING):idx]

        df_train = pd.concat(
            [pgb.componenti_in_dataframe(pgb.calcola_componenti_per_stagione(s)) for s in stagioni_training],
            ignore_index=True)
        df_test = pgb.componenti_in_dataframe(pgb.calcola_componenti_per_stagione(stagione))
        df_train_esteso = pd.concat([df_train, pool_altre_leghe], ignore_index=True)

        for df, accumulatore in ((df_train, p_solo), (df_train_esteso, p_esteso)):
            modello = _gb()
            modello.fit(df[colonne], df["esito"])
            accumulatore.extend(_proba_ordinata(modello, df_test[colonne]).tolist())

        reali.extend(df_test["esito"].tolist())

    return p_solo, p_esteso, reali


if __name__ == "__main__":
    print("Fase 0 — rivalutazione multi-campionato con protocollo corretto")
    print(f"Finestra di test: {len(STAGIONI)} stagioni ({', '.join(sorted(STAGIONI))})")
    print(f"Configurazione quote: {pgb.METODO_QUOTE} / {pgb.FONTE_QUOTE} / medie {pgb.MEDIE_LEGA}\n", flush=True)

    print("Costruzione del pool delle altre leghe...", flush=True)
    pool = pml.costruisci_pool_altre_leghe()
    print(f"Pool: {len(pool)} righe di training aggiuntive\n", flush=True)

    p_solo, p_esteso, reali = raccogli(STAGIONI, pool)
    print(f"\nRaccolte {len(reali)} partite.\n", flush=True)

    esito = protocollo.confronta("GB multi-campionato", p_esteso, "GB solo Serie A", p_solo, reali)
    print(esito)
    protocollo.riepiloga([esito])
