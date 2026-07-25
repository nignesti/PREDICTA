"""
Fase 0: rivalutazione di ensemble stacking e gradient boosting con il protocollo
di misura corretto (protocollo.py).

Perche' rifarlo. I due esperimenti erano stati archiviati come negativi/misti
sulla base dell'accuratezza media su 3 stagioni. Da allora sono emersi due
problemi che invalidano quel giudizio:

1. **Metrica senza potenza.** Fra due configurazioni cambia previsione solo il
   ~2% delle partite: l'accuratezza su ~1.100 partite non distingue differenze
   di mezzo punto percentuale, ne' in positivo ne' in negativo. Entrambi gli
   esperimenti avevano pero' l'RPS MIGLIORE di qualunque altra cosa misurata nel
   progetto (0.1880 lo stacking, 0.1896 il gradient boosting contro 0.1889 del
   modello statistico), e l'RPS ha molta piu' potenza.

2. **Baseline sbagliato.** `prototipo_gradient_boosting.py` chiamava
   `bt.precompute_componente` senza passare la configurazione delle quote,
   ereditando i default "proporzionale"/"apertura"/"storiche": cioe' il modello
   PRE-Fase 2. Sia il baseline statistico sia le feature `prob_*_quote` del
   gradient boosting erano quindi calcolati col metodo vecchio, mentre la
   produzione usa Shin + quote di chiusura. Corretto in questa fase.

Qui i tre modelli vengono ricalcolati sulle stesse partite e confrontati con
`protocollo.confronta`, sulla finestra standard a 7 stagioni.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np

import backtesting as bt
import prototipo_gradient_boosting as pgb
import prototipo_ensemble_stacking as pes
import protocollo

STAGIONI = protocollo.STAGIONI_TEST


def raccogli(stagioni):
    """Calcola, per ogni stagione, le probabilita' dei tre modelli sulle stesse
    partite, e le concatena. Restituisce (prob_statistico, prob_gb, prob_meta,
    reali) — tutte allineate riga per riga."""
    p_stat, p_gb, p_meta, reali = [], [], [], []
    for stagione in stagioni:
        print(f"  stagione {stagione}...", flush=True)
        stagioni_disponibili = sorted(bt.df["Stagione"].astype(str).unique())
        idx = stagioni_disponibili.index(stagione)
        stagioni_training = stagioni_disponibili[max(0, idx - pgb.N_STAGIONI_TRAINING):idx]

        comp_train = [c for s in stagioni_training for c in pgb.calcola_componenti_per_stagione(s)]
        comp_test = pgb.calcola_componenti_per_stagione(stagione)
        colonne = pgb.COLONNE_FEATURE

        df_train = pgb.componenti_in_dataframe(comp_train)
        df_test = pgb.componenti_in_dataframe(comp_test)

        stat_train = pes.prob_statistiche(comp_train)
        stat_test = pes.prob_statistiche(comp_test)

        gb_train_oof = pes.prob_gb_oof(df_train, colonne)
        gb_finale = pes._gb_base()
        gb_finale.fit(df_train[colonne], df_train["esito"])
        gb_test = pes._proba_ordinata(gb_finale, df_test[colonne])

        meta = pes._gb_base(random_state=1)
        meta.fit(np.hstack([stat_train, gb_train_oof]), df_train["esito"].to_numpy())
        meta_test = pes._proba_ordinata(meta, np.hstack([stat_test, gb_test]))

        p_stat.extend(stat_test.tolist())
        p_gb.extend(gb_test.tolist())
        p_meta.extend(meta_test.tolist())
        reali.extend(df_test["esito"].tolist())

    return p_stat, p_gb, p_meta, reali


if __name__ == "__main__":
    print("Fase 0 — rivalutazione con protocollo corretto")
    print(f"Finestra di test: {len(STAGIONI)} stagioni ({', '.join(sorted(STAGIONI))})")
    print(f"Configurazione quote: {pgb.METODO_QUOTE} / {pgb.FONTE_QUOTE} / medie {pgb.MEDIE_LEGA}\n", flush=True)

    p_stat, p_gb, p_meta, reali = raccogli(STAGIONI)
    print(f"\nRaccolte {len(reali)} partite.\n", flush=True)

    esiti = [
        protocollo.confronta("ensemble stacking", p_meta, "modello statistico", p_stat, reali),
        protocollo.confronta("gradient boosting", p_gb, "modello statistico", p_stat, reali),
        protocollo.confronta("ensemble stacking", p_meta, "gradient boosting", p_gb, reali),
    ]
    for e in esiti:
        print(e)
        print()

    protocollo.riepiloga(esiti)
