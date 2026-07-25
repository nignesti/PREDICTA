"""
Il modello statistico batte il mercato sull'Over/Under 2.5?

Sull'1X2 la risposta e' no: misurato su 12.421 partite di 5 campionati, il
modello e' leggermente ma misurabilmente peggiore, e il suo peso ottimale e'
zero (vedi ROADMAP.md).

L'Over/Under merita pero' una misura separata, per una ragione strutturale:

- Il modello Dixon-Coles **modella i gol**. Una distribuzione di probabilita'
  sul numero di gol di ciascuna squadra e' la sua uscita nativa, e l'Over/Under
  2.5 si legge direttamente da li' sommando le celle con piu' di 2 gol totali.
- L'1X2 invece obbliga a convertire i gol in "chi vince", buttando via
  informazione: un 3-0 e un 1-0 sono lo stesso esito. Il modello viene giudicato
  su una domanda piu' povera di quella che sa rispondere.
- In piu' il mercato Over/Under muove meno volume e meno attenzione di quello
  1X2, quindi ha piu' margine per essere meno efficiente.

E' la prima ipotesi del progetto con una ragione strutturale per funzionare,
invece di essere "proviamo ad aggiungere un dato".

Protocollo identico al resto: 5 campionati, 7 stagioni (2019-2025), walk-forward,
verdetto su Brier con bootstrap appaiato (su due esiti l'RPS coincide col Brier),
accuratezza con McNemar come metrica descrittiva.
"""
import os
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
import pandas as pd

import multilega as ml
import protocollo

FILE_CACHE = "componenti_multilega_ou.csv.gz"
COLONNE_CACHE = ["xG_casa_storico", "xG_trasf_storico", "xG_casa_forma", "xG_trasf_forma",
                 "xG_casa_scontri", "xG_trasf_scontri", "scontri_validi", "quote_presenti",
                 "prob_1_quote", "prob_X_quote", "prob_2_quote", "quote_ou_presenti",
                 "prob_over_mercato", "esito_over", "esito", "stagione", "lega", "data"]


def carica_o_calcola(forza=False):
    if os.path.exists(FILE_CACHE) and not forza:
        print(f"Riuso le componenti da {FILE_CACHE}.", flush=True)
        comp = pd.read_csv(FILE_CACHE).to_dict("records")
        for c in comp:
            c["elo_casa"] = c["elo_trasferta"] = None
            c["stagione"] = str(c["stagione"])
        return comp
    print(f"Calcolo componenti su {len(ml.LEGHE)} leghe (~25 minuti)...", flush=True)
    comp = ml.raccogli_tutte(protocollo.STAGIONI_TEST)
    pd.DataFrame(comp)[COLONNE_CACHE].to_csv(FILE_CACHE, index=False)
    return comp


if __name__ == "__main__":
    comp = carica_o_calcola("--ricalcola" in sys.argv)
    comp = [c for c in comp if c["quote_ou_presenti"]]
    esiti = [bool(c["esito_over"]) for c in comp]
    print(f"\n{len(comp)} partite con quote Over/Under, su {len(ml.LEGHE)} campionati.")
    print(f"Frequenza reale di Over 2.5: {np.mean(esiti):.1%}\n", flush=True)

    p_mercato = [p[0] for p in ml.probabilita_over_mercato(comp)]
    p_modello = [p[0] for p in ml.probabilita_over_modello(comp, peso_quote=0.0)]

    print("=" * 92)
    print("IL MODELLO BATTE IL MERCATO SULL'OVER/UNDER 2.5?")
    print("=" * 92)
    esito = protocollo.confronta_binario("modello statistico", p_modello,
                                         "solo mercato", p_mercato, esiti)
    print(esito)

    print("\n\nBlend modello + mercato, al variare del peso delle quote")
    print("-" * 92)
    print(f"{'peso quote':>11} {'Brier':>9} {'accuratezza':>12}   verdetto vs mercato puro")
    esiti_blend = []
    for pq in (0.0, 0.25, 0.50, 0.75, 0.90, 1.0):
        p = [x[0] for x in ml.probabilita_over_modello(comp, peso_quote=pq)]
        b = protocollo.brier_per_partita(p, esiti).mean()
        acc = np.mean((np.array(p) >= 0.5) == np.array(esiti))
        if pq == 1.0:
            nota = "(= mercato puro, riferimento)"
        else:
            e = protocollo.confronta_binario(f"peso {pq}", p, "mercato", p_mercato, esiti)
            esiti_blend.append(e)
            nota = f"{e.verdetto}  IC[{e.ic_rps[0]:+.5f}, {e.ic_rps[1]:+.5f}]"
        print(f"{pq:>11.2f} {b:>9.5f} {acc:>11.2%}   {nota}", flush=True)

    print("\n\nBreakdown per campionato (modello puro vs mercato)")
    print("-" * 92)
    per_lega = []
    for codice, nome in ml.LEGHE.items():
        idx = [i for i, c in enumerate(comp) if c["lega"] == codice]
        if not idx:
            continue
        e = protocollo.confronta_binario(f"modello [{nome}]", [p_modello[i] for i in idx],
                                         "mercato", [p_mercato[i] for i in idx],
                                         [esiti[i] for i in idx])
        per_lega.append(e)
        print(e)
        print()

    protocollo.riepiloga([esito] + per_lega)
