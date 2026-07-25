"""
Priorita' 1 della ROADMAP: il modello statistico batte il mercato, misurato su
5 campionati invece che solo sulla Serie A.

La Fase 0 ha lasciato il progetto in questa situazione: su 2.659 partite di
Serie A il vantaggio del modello sul mercato e' +0.34 punti percentuali con
McNemar p = 0.28, cioe' indistinguibile dal rumore — e il calcolo di potenza
diceva che per risolverlo servirebbero decine di stagioni. La Serie A ne produce
una all'anno, quindi la domanda centrale del progetto era di fatto senza
risposta.

Le altre quattro leghe principali erano gia' scaricate ma usate solo come
training del gradient boosting. Qui vengono usate come TEST: stesse 7 stagioni
(2019-2025), stessa configurazione di produzione, ogni lega trattata come un
mondo chiuso (medie di campionato e confronti fra squadre interni alla lega).
Il campione passa da 2.659 a ~12.500 partite, sopra la soglia di potenza.

Salva le componenti calcolate in `componenti_multilega.csv.gz` (~26 minuti di
calcolo): gli esperimenti successivi possono riusarle senza ricalcolare, perche'
le componenti non dipendono dai pesi del blend.
"""
import os
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import pandas as pd

import multilega as ml
import protocollo

FILE_CACHE = "componenti_multilega.csv.gz"
COLONNE_CACHE = ["xG_casa_storico", "xG_trasf_storico", "xG_casa_forma", "xG_trasf_forma",
                 "xG_casa_scontri", "xG_trasf_scontri", "scontri_validi", "quote_presenti",
                 "prob_1_quote", "prob_X_quote", "prob_2_quote", "esito", "stagione", "lega", "data"]


def carica_o_calcola(forza_ricalcolo=False):
    if os.path.exists(FILE_CACHE) and not forza_ricalcolo:
        print(f"Riuso le componenti da {FILE_CACHE} (cancellalo per ricalcolare).", flush=True)
        df = pd.read_csv(FILE_CACHE)
        comp = df.to_dict("records")
        for c in comp:
            c["elo_casa"] = c["elo_trasferta"] = None
            c["stagione"] = str(c["stagione"])
        return comp

    print(f"Calcolo componenti su {len(ml.LEGHE)} leghe, stagioni {min(protocollo.STAGIONI_TEST)}-"
          f"{max(protocollo.STAGIONI_TEST)} (richiede ~25 minuti)...", flush=True)
    comp = ml.raccogli_tutte(protocollo.STAGIONI_TEST)
    pd.DataFrame(comp)[COLONNE_CACHE].to_csv(FILE_CACHE, index=False)
    print(f"Componenti salvate in {FILE_CACHE}.", flush=True)
    return comp


if __name__ == "__main__":
    componenti = carica_o_calcola(forza_ricalcolo="--ricalcola" in sys.argv)
    componenti = [c for c in componenti if c["quote_presenti"]]
    print(f"\n{len(componenti)} partite con quote, su {len(ml.LEGHE)} campionati.\n", flush=True)

    p_modello = ml.probabilita_modello(componenti)
    p_mercato = ml.probabilita_mercato(componenti)
    reali = [c["esito"] for c in componenti]

    print("=" * 92)
    print("DOMANDA CENTRALE: il modello batte il mercato?")
    print("=" * 92)
    esito_totale = protocollo.confronta("modello (forma+quote)", p_modello, "solo mercato", p_mercato, reali)
    print(esito_totale)

    print("\n\nBreakdown per campionato")
    print("-" * 92)
    esiti_lega = []
    for codice, nome in ml.LEGHE.items():
        idx = [i for i, c in enumerate(componenti) if c["lega"] == codice]
        if not idx:
            continue
        e = protocollo.confronta(f"modello [{nome}]", [p_modello[i] for i in idx],
                                 "mercato", [p_mercato[i] for i in idx], [reali[i] for i in idx])
        esiti_lega.append(e)
        print(e)
        print()

    protocollo.riepiloga([esito_totale] + esiti_lega)
