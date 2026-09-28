"""
Closing line value (CLV) delle quote a valore segnalate da quote_live_serie_a.py.

Stessa logica di predicta/nba/quote/clv_nba.py (vedi quel modulo per il
perche' del CLV invece del solo risultato). Unica differenza: probabilita_eque
qui restituisce una terna/coppia di probabilita' nella colonna 'probabilita'
invece di un solo 'p1', quindi _p_esito seleziona la componente giusta in base
all'esito invece di dedurla per complemento a 1.

Dati: gli snapshot in quote_live/ (raccolti da raccogli_quote_serie_a.py via
GitHub Actions, prefisso file serie_a_*).

    python -m predicta.serie_a.quote.clv_serie_a
"""
import numpy as np
import pandas as pd

import predicta.serie_a.quote.quote_live_serie_a as ql

SOGLIA_EV = 0.02  # stessa soglia di default della pagina "Valore Serie A"


def _p_esito(probabilita, mercato, esito):
    return probabilita[ql.ESITI[mercato].index(esito)]


def chiusure(snapshot, book_sharp=ql.BOOK_SHARP):
    """Per ogni (partita, mercato): probabilita' equa e linea nell'ultimo
    snapshot pre-partita in cui la partita e' quotata."""
    pre = snapshot[snapshot["istante"] < snapshot["inizio"]]
    if pre.empty:
        return pd.DataFrame(columns=["id", "mercato", "linea_chiusura", "probabilita_chiusura", "istante_chiusura"])
    # per mercato: lo snapshot pre-partita ha solo l'1X2, il totale chiude prima
    ultimo = pre.groupby(["id", "mercato"])["istante"].transform("max")
    finali = pre[pre["istante"] == ultimo]
    eque = []
    for (id_, istante), righe in finali.groupby(["id", "istante"]):
        e = ql.probabilita_eque(righe.drop(columns="istante"), book_sharp)
        eque.append(e.assign(istante_chiusura=istante))
    eque = pd.concat(eque, ignore_index=True)
    return eque.rename(columns={"linea_rif": "linea_chiusura", "probabilita": "probabilita_chiusura"})[
        ["id", "mercato", "linea_chiusura", "probabilita_chiusura", "istante_chiusura"]]


def segnalazioni(snapshot, soglia_ev=SOGLIA_EV, book_sharp=ql.BOOK_SHARP):
    """Le quote a valore di ogni snapshot pre-partita, come le avrebbe viste
    la pagina in quel momento. Una stessa quota segnalata in piu' snapshot
    conta una volta sola (la prima)."""
    pre = snapshot[snapshot["istante"] < snapshot["inizio"]]
    tutte = []
    for istante, righe in pre.groupby("istante"):
        v = ql.trova_valore(righe.drop(columns="istante"), soglia_ev=soglia_ev, book_sharp=book_sharp)
        if not v.empty:
            tutte.append(v.assign(istante=istante))
    if not tutte:
        return pd.DataFrame()
    s = pd.concat(tutte, ignore_index=True).sort_values("istante", kind="stable")
    return s.drop_duplicates(subset=["id", "mercato", "esito", "book", "linea"]).reset_index(drop=True)


def calcola_clv(snapshot, soglia_ev=SOGLIA_EV, book_sharp=ql.BOOK_SHARP):
    """Una riga per segnalazione con CLV = quota presa x p_equa di chiusura - 1.
    Solo segnalazioni fatte prima dello snapshot di chiusura e, per il totale,
    solo se la linea di chiusura di Pinnacle e' la stessa della quota presa
    (l'1X2 non ha linea, quindi non e' mai escluso per questo motivo).
    'linea_mossa' conta queste esclusioni."""
    s = segnalazioni(snapshot, soglia_ev, book_sharp)
    c = chiusure(snapshot, book_sharp)
    if s.empty or c.empty:
        return pd.DataFrame(), 0
    unite = s.merge(c, on=["id", "mercato"])
    unite = unite[unite["istante"] < unite["istante_chiusura"]]
    linea_presa = unite["linea"].fillna(0.0).astype(float)
    linea_chiusura = unite["linea_chiusura"].astype(float)
    stessa_linea = (unite["mercato"] == "h2h") | np.isclose(linea_presa, linea_chiusura)
    linea_mossa = int((~stessa_linea).sum())
    unite = unite[stessa_linea].copy()
    unite["p_chiusura"] = [
        _p_esito(p, m, e) for p, m, e in zip(unite["probabilita_chiusura"], unite["mercato"], unite["esito"])
    ]
    unite["CLV"] = unite["quota"] * unite["p_chiusura"] - 1
    return unite.reset_index(drop=True), linea_mossa


def riepilogo_clv(clv, n_bootstrap=5000, rng=None):
    """Media del CLV con intervallo di confidenza bootstrap al 95%."""
    if clv.empty:
        return None
    valori = clv["CLV"].to_numpy()
    rng = rng or np.random.default_rng(12345)
    medie = valori[rng.integers(0, len(valori), size=(n_bootstrap, len(valori)))].mean(axis=1)
    basso, alto = np.percentile(medie, [2.5, 97.5])
    return {
        "n": len(valori),
        "clv_medio": float(valori.mean()),
        "ic_95": (float(basso), float(alto)),
        "quota_positivi": float((valori > 0).mean()),
        "ev_segnalato_medio": float(clv["EV"].mean()),
    }


def main():
    snapshot = ql.carica_snapshot()
    if snapshot.empty:
        print(f"Nessuno snapshot in {ql.CARTELLA_SNAPSHOT}/: servono le raccolte di raccogli_quote_serie_a.py")
        return
    print(f"{snapshot['istante'].nunique()} snapshot, {snapshot['id'].nunique()} partite "
          f"({snapshot['istante'].min():%d/%m/%Y} - {snapshot['istante'].max():%d/%m/%Y})")
    clv, linea_mossa = calcola_clv(snapshot)
    r = riepilogo_clv(clv)
    if r is None:
        print("Nessuna segnalazione confrontabile con una chiusura (servono snapshot successivi).")
        return
    print(f"Segnalazioni: {r['n']} (escluse {linea_mossa} per linea mossa)")
    print(f"EV segnalato medio: {r['ev_segnalato_medio']:+.2%}")
    print(f"CLV medio: {r['clv_medio']:+.2%}  IC 95% [{r['ic_95'][0]:+.2%}, {r['ic_95'][1]:+.2%}]")
    print(f"Segnalazioni con CLV positivo: {r['quota_positivi']:.0%}")
    print(clv.groupby("mercato")["CLV"].agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
