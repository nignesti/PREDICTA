"""
Closing line value (CLV) delle quote a valore segnalate da quote_live_nba.py.

Perche' il CLV e non il risultato: una scommessa a +3% di valore atteso vince
o perde quasi come una moneta, quindi per distinguere un vantaggio vero dal
caso servono migliaia di esiti. Il CLV confronta invece la quota presa con la
probabilita' equa di Pinnacle a ridosso dell'inizio (la "chiusura", il prezzo
piu' informato che esista): se le quote segnalate battono sistematicamente la
chiusura, il vantaggio e' reale, e lo si vede con qualche centinaio di
segnalazioni. E' la metrica standard con cui si valuta chi scommette.

Dati: gli snapshot in quote_live/ (raccolti da raccogli_quote_nba.py via
GitHub Actions). La "chiusura" di una partita e' l'ultimo snapshot con
Pinnacle prima dell'inizio: con due raccolte al giorno e' una chiusura
approssimata (fino a qualche ora prima della palla a due), che sottostima il
movimento reale ma non introduce distorsioni a favore.

    python clv_nba.py
"""
import numpy as np
import pandas as pd

import predicta.nba.quote.quote_live_nba as ql

SOGLIA_EV = 0.02  # stessa soglia di default della pagina "Quote live e valore"


def _p_esito(p1, esito):
    return p1 if esito in ("casa", "over") else 1 - p1


def chiusure(snapshot, book_sharp=ql.BOOK_SHARP):
    """Per ogni (partita, mercato): probabilita' equa del primo esito e linea
    nell'ultimo snapshot pre-partita in cui la partita e' quotata."""
    pre = snapshot[snapshot["istante"] < snapshot["inizio"]]
    if pre.empty:
        return pd.DataFrame(columns=["id", "mercato", "linea_chiusura", "p1_chiusura", "istante_chiusura"])
    ultimo = pre.groupby("id")["istante"].transform("max")
    finali = pre[pre["istante"] == ultimo]
    eque = []
    for (id_, istante), righe in finali.groupby(["id", "istante"]):
        e = ql.probabilita_eque(righe.drop(columns="istante"), book_sharp)
        eque.append(e.assign(istante_chiusura=istante))
    eque = pd.concat(eque, ignore_index=True)
    return eque.rename(columns={"linea_rif": "linea_chiusura", "p1": "p1_chiusura"})[
        ["id", "mercato", "linea_chiusura", "p1_chiusura", "istante_chiusura"]]


def segnalazioni(snapshot, soglia_ev=SOGLIA_EV, book_sharp=ql.BOOK_SHARP):
    """Le quote a valore di ogni snapshot pre-partita, come le avrebbe viste
    la pagina in quel momento. Una stessa quota segnalata in piu' snapshot
    conta una volta sola (la prima): chi scommette la prende una volta."""
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
    Solo segnalazioni fatte prima dello snapshot di chiusura (altrimenti il
    confronto e' con se stesse) e, per spread e totale, solo se la linea di
    chiusura di Pinnacle e' la stessa della quota presa: se la linea si e'
    mossa la probabilita' di chiusura si riferisce a un'altra scommessa.
    'linea_mossa' conta queste esclusioni."""
    s = segnalazioni(snapshot, soglia_ev, book_sharp)
    c = chiusure(snapshot, book_sharp)
    if s.empty or c.empty:
        return pd.DataFrame(), 0
    unite = s.merge(c, on=["id", "mercato"])
    unite = unite[unite["istante"] < unite["istante_chiusura"]]
    linea_rif = np.where((unite["mercato"] == "spreads") & (unite["esito"] == "trasferta"),
                         -unite["linea"].astype(float), unite["linea"].astype(float))
    stessa_linea = (unite["mercato"] == "h2h") | np.isclose(linea_rif, unite["linea_chiusura"].astype(float))
    linea_mossa = int((~stessa_linea).sum())
    unite = unite[stessa_linea].copy()
    unite["p_chiusura"] = [_p_esito(p, e) for p, e in zip(unite["p1_chiusura"], unite["esito"])]
    unite["CLV"] = unite["quota"] * unite["p_chiusura"] - 1
    return unite.reset_index(drop=True), linea_mossa


def riepilogo_clv(clv, n_bootstrap=5000, rng=None):
    """Media del CLV con intervallo di confidenza bootstrap al 95%: il
    vantaggio e' dimostrato solo se l'intervallo sta tutto sopra zero."""
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
        print(f"Nessuno snapshot in {ql.CARTELLA_SNAPSHOT}/: servono le raccolte di raccogli_quote_nba.py")
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
