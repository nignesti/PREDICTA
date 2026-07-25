"""
Ricerca esaustiva della configurazione migliore per la schedina.

Esplora insieme le due famiglie di scelte che determinano il risultato:

  MODELLO      peso forma, peso scontri diretti, peso quote, rho di Dixon-Coles
  STRATEGIA    mercati ammessi (1X2 / Over-Under / entrambi), soglia di confidenza

e valuta ogni combinazione su 12.421 partite di 5 campionati (2019-2025),
walk-forward, con le stesse componenti gia' validate.

Il criterio non e' l'accuratezza generica su tutte le partite, ma quello che
conta davvero per una schedina:

  1. ACCURATEZZA DELLE SELEZIONI — quanto sono azzeccati i pronostici che
     finiscono effettivamente in schedina (cioe' quelli sopra soglia). E' il
     numero che, elevato al numero di partite, da' la probabilita' di schedina
     piena.
  2. CALIBRAZIONE — se una configurazione dichiara 70% e ne azzecca il 60%,
     la probabilita' di schedina piena che mostra all'utente e' una bugia.
     Misurata come scarto medio assoluto fra confidenza dichiarata e frequenza
     reale, per fasce di confidenza.
  3. DISPONIBILITA' — quante partite superano la soglia. Una configurazione
     precisissima che seleziona 3 partite a giornata non permette una schedina
     da 13.

Il calcolo della matrice dei punteggi e' vettorizzato su tutte le partite in una
volta: senza questo, migliaia di configurazioni non sarebbero praticabili.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import itertools

import numpy as np
import pandas as pd
from scipy.stats import poisson

import protocollo
from valida_over_under import carica_o_calcola

MAX_GOL = 10
N_SCHEDINA = 13

# --- griglia esplorata ---
GRIGLIA_FORMA = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
GRIGLIA_SCONTRI = [0.0, 0.10, 0.20]
GRIGLIA_QUOTE = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
GRIGLIA_RHO = [-0.20, -0.10, 0.0]
GRIGLIA_MERCATI = ["1X2", "Over/Under", "entrambi"]
GRIGLIA_SOGLIA = [0.55, 0.60, 0.65, 0.70, 0.75]


def esiti_vettoriali(xg_casa, xg_trasferta, rho):
    """Probabilita' di 1, X, 2 e Over 2.5 per TUTTE le partite in una volta.

    Equivalente a distribuzione_punteggi + esiti_da_matrice applicati partita per
    partita, ma con un solo passaggio vettoriale: e' cio' che rende possibile
    esplorare migliaia di configurazioni invece di poche decine."""
    gol = np.arange(MAX_GOL + 1)
    p_casa = poisson.pmf(gol[None, :], xg_casa[:, None])
    p_trasf = poisson.pmf(gol[None, :], xg_trasferta[:, None])
    m = p_casa[:, :, None] * p_trasf[:, None, :]

    # Vincolo di validita' di Dixon-Coles, per partita: fuori da questa regione
    # tau produce fattori negativi (vedi modello.rho_ammissibile).
    limite = np.maximum(-1.0 / xg_casa, -1.0 / xg_trasferta)
    tetto = np.minimum(1.0 / (xg_casa * xg_trasferta), 1.0)
    r = np.clip(rho, limite, tetto)

    m[:, 0, 0] *= 1 - xg_casa * xg_trasferta * r
    m[:, 0, 1] *= 1 + xg_casa * r
    m[:, 1, 0] *= 1 + xg_trasferta * r
    m[:, 1, 1] *= 1 - r
    m /= m.sum(axis=(1, 2), keepdims=True)

    i, j = np.meshgrid(gol, gol, indexing="ij")
    p1 = m[:, i > j].sum(axis=1)
    px = m[:, i == j].sum(axis=1)
    p2 = m[:, i < j].sum(axis=1)
    over = m[:, (i + j) > 2.5].sum(axis=1)
    return p1, px, p2, over


def prepara(componenti):
    """Estrae in array numpy tutto cio' che serve alla ricerca."""
    def col(nome):
        return np.array([c[nome] for c in componenti], dtype=float)

    return {
        "xg_cs": col("xG_casa_storico"), "xg_ts": col("xG_trasf_storico"),
        "xg_cf": col("xG_casa_forma"), "xg_tf": col("xG_trasf_forma"),
        "xg_csc": col("xG_casa_scontri"), "xg_tsc": col("xG_trasf_scontri"),
        "scontri_validi": np.array([bool(c["scontri_validi"]) for c in componenti]),
        "q1": col("prob_1_quote"), "qx": col("prob_X_quote"), "q2": col("prob_2_quote"),
        "q_over": col("prob_over_mercato"),
        "esito": np.array([c["esito"] for c in componenti]),
        "over_reale": np.array([bool(c["esito_over"]) for c in componenti]),
        "lega": np.array([c["lega"] for c in componenti]),
    }


def probabilita(dati, pf, ps, pq, rho):
    """Probabilita' finali dei quattro esiti (1, X, 2, Over) per una configurazione."""
    ps_eff = np.where(dati["scontri_validi"], ps, 0.0)
    peso_storico = np.maximum(0.0, 1 - pf - ps_eff)
    tot = peso_storico + pf + ps_eff
    xg_casa = (peso_storico * dati["xg_cs"] + pf * dati["xg_cf"] + ps_eff * dati["xg_csc"]) / tot
    xg_trasf = (peso_storico * dati["xg_ts"] + pf * dati["xg_tf"] + ps_eff * dati["xg_tsc"]) / tot
    xg_casa = np.maximum(0.05, xg_casa)
    xg_trasf = np.maximum(0.05, xg_trasf)

    p1, px, p2, over = esiti_vettoriali(xg_casa, xg_trasf, rho)
    p1 = (1 - pq) * p1 + pq * dati["q1"]
    px = (1 - pq) * px + pq * dati["qx"]
    p2 = (1 - pq) * p2 + pq * dati["q2"]
    over = (1 - pq) * over + pq * dati["q_over"]
    return p1, px, p2, over


def valuta(dati, p1, px, p2, over, mercati, soglia):
    """Metriche della schedina per una configurazione gia' calcolata."""
    n = len(p1)
    # Candidato 1X2: esito piu' probabile fra i tre.
    tris = np.vstack([p1, px, p2]).T
    i_1x2 = tris.argmax(axis=1)
    conf_1x2 = tris.max(axis=1)
    ok_1x2 = np.array(["1", "X", "2"])[i_1x2] == dati["esito"]

    # Candidato Over/Under: esito piu' probabile fra i due.
    conf_ou = np.maximum(over, 1 - over)
    ok_ou = (over >= 0.5) == dati["over_reale"]

    if mercati == "1X2":
        conf, ok = conf_1x2, ok_1x2
    elif mercati == "Over/Under":
        conf, ok = conf_ou, ok_ou
    else:
        usa_ou = conf_ou > conf_1x2
        conf = np.where(usa_ou, conf_ou, conf_1x2)
        ok = np.where(usa_ou, ok_ou, ok_1x2)

    sel = conf >= soglia
    n_sel = int(sel.sum())
    if n_sel < N_SCHEDINA * 3:      # troppo poche per un giudizio affidabile
        return None

    conf_sel, ok_sel = conf[sel], ok[sel]
    tasso = float(ok_sel.mean())
    dichiarata = float(conf_sel.mean())

    # Calibrazione: scarto medio assoluto fra dichiarato e reale, per fasce.
    scarti, pesi = [], []
    for lo in np.arange(0.5, 1.0, 0.05):
        m = (conf_sel >= lo) & (conf_sel < lo + 0.05)
        if m.sum() >= 50:
            scarti.append(abs(conf_sel[m].mean() - ok_sel[m].mean()))
            pesi.append(m.sum())
    calibrazione = float(np.average(scarti, weights=pesi)) if scarti else np.nan

    return {
        "n_selezionate": n_sel,
        "quota_selezionate": n_sel / n,
        "tasso_reale": tasso,
        "confidenza_dichiarata": dichiarata,
        "scarto_calibrazione": calibrazione,
        "bias": dichiarata - tasso,          # >0 = la configurazione si sopravvaluta
        "p_schedina_13": tasso ** N_SCHEDINA,
        "brier_selezioni": float(np.mean((conf_sel - ok_sel.astype(float)) ** 2)),
    }


if __name__ == "__main__":
    comp = carica_o_calcola()
    comp = [c for c in comp if c["quote_presenti"] and c["quote_ou_presenti"]]
    dati = prepara(comp)
    print(f"Ricerca su {len(comp)} partite, 5 campionati, 7 stagioni.\n", flush=True)

    combinazioni_modello = [(pf, ps, pq, rho)
                            for pf in GRIGLIA_FORMA for ps in GRIGLIA_SCONTRI
                            for pq in GRIGLIA_QUOTE for rho in GRIGLIA_RHO
                            if pf + ps + pq <= 1.0001]
    totale = len(combinazioni_modello) * len(GRIGLIA_MERCATI) * len(GRIGLIA_SOGLIA)
    print(f"{len(combinazioni_modello)} configurazioni di modello x {len(GRIGLIA_MERCATI)} mercati "
          f"x {len(GRIGLIA_SOGLIA)} soglie = {totale} combinazioni\n", flush=True)

    risultati = []
    for indice, (pf, ps, pq, rho) in enumerate(combinazioni_modello, start=1):
        p1, px, p2, over = probabilita(dati, pf, ps, pq, rho)
        for mercati, soglia in itertools.product(GRIGLIA_MERCATI, GRIGLIA_SOGLIA):
            m = valuta(dati, p1, px, p2, over, mercati, soglia)
            if m is None:
                continue
            risultati.append(dict(forma=pf, scontri=ps, quote=pq, rho=rho,
                                  mercati=mercati, soglia=soglia, **m))
        if indice % 25 == 0:
            print(f"  {indice}/{len(combinazioni_modello)} configurazioni di modello...", flush=True)

    df = pd.DataFrame(risultati)
    df.to_csv("ricerca_configurazione.csv", index=False)
    print(f"\n{len(df)} combinazioni valutate, salvate in ricerca_configurazione.csv\n")

    def mostra(titolo, sotto, ordina, crescente=True, n=10):
        print("=" * 104)
        print(titolo)
        print(sotto)
        print("=" * 104)
        print(f"{'forma':>6} {'scont':>6} {'quote':>6} {'rho':>6} {'mercati':>11} {'soglia':>7} "
              f"{'sel.':>6} {'reale':>7} {'dich.':>7} {'bias':>7} {'calib':>6} {'P(13/13)':>9}")
        for _, r in df.sort_values(ordina, ascending=crescente).head(n).iterrows():
            print(f"{r.forma:>6.2f} {r.scontri:>6.2f} {r.quote:>6.2f} {r.rho:>6.2f} "
                  f"{r.mercati:>11} {r.soglia:>7.0%} {r.n_selezionate:>6.0f} "
                  f"{r.tasso_reale:>6.1%} {r.confidenza_dichiarata:>6.1%} {r.bias:>+6.1%} "
                  f"{r.scarto_calibrazione:>5.1%} {r.p_schedina_13:>8.3%}")
        print()

    mostra("MIGLIORI PER ACCURATEZZA DELLE SELEZIONI",
           "quanto sono azzeccati i pronostici che finiscono in schedina",
           "tasso_reale", crescente=False)

    mostra("MIGLIORI PER CALIBRAZIONE",
           "quanto la confidenza dichiarata corrisponde alla frequenza reale",
           "scarto_calibrazione", crescente=True)

    # Configurazioni utilizzabili davvero: abbastanza partite per una schedina
    # da 13 su una giornata di 5 campionati (~50 partite a settimana).
    utili = df[df["quota_selezionate"] >= 0.25].copy()
    print("=" * 104)
    print("MIGLIOR COMPROMESSO — accuratezza massima fra le configurazioni che")
    print("selezionano almeno il 25% delle partite (abbastanza per una schedina da 13)")
    print("=" * 104)
    for _, r in utili.sort_values("tasso_reale", ascending=False).head(10).iterrows():
        print(f"{r.forma:>6.2f} {r.scontri:>6.2f} {r.quote:>6.2f} {r.rho:>6.2f} "
              f"{r.mercati:>11} {r.soglia:>7.0%} {r.n_selezionate:>6.0f} "
              f"{r.tasso_reale:>6.1%} {r.confidenza_dichiarata:>6.1%} {r.bias:>+6.1%} "
              f"{r.scarto_calibrazione:>5.1%} {r.p_schedina_13:>8.3%}")
    print()

    print("=" * 104)
    print("EFFETTO DEI SINGOLI PARAMETRI (media del tasso reale, a parita' di tutto il resto)")
    print("=" * 104)
    for parametro in ("quote", "forma", "scontri", "rho", "mercati", "soglia"):
        print(f"\n  {parametro}:")
        for valore, gruppo in df.groupby(parametro):
            etichetta = f"{valore:.2f}" if isinstance(valore, float) else str(valore)
            print(f"    {etichetta:>12}: tasso reale medio {gruppo['tasso_reale'].mean():.2%}, "
                  f"calibrazione {gruppo['scarto_calibrazione'].mean():.2%}")
