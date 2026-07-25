"""
Componenti del modello statistico calcolate in modo uniforme su piu' campionati
(Fase 0, priorita' 1 della ROADMAP).

Perche' serve. La Fase 0 ha mostrato che gli effetti in gioco richiedono 15-65
stagioni di test per essere misurati, mentre la Serie A ne produce una all'anno:
con 2.659 partite ogni esperimento risulta indistinguibile dal rumore. Le altre
quattro leghe principali sono gia' scaricate ma vengono usate solo come training
del gradient boosting. Usarle anche come TEST porta il campione da 2.659 a
~12.500 partite sulle stesse 7 stagioni, sopra la soglia di potenza necessaria.

Cosa fa questo modulo. Carica ciascuna lega, ne ricostruisce le stagioni e
calcola le componenti per-partita (storico, forma, scontri diretti, quote) con
la STESSA configurazione di produzione usata per la Serie A: quote di chiusura
convertite con Shin, medie di lega pesate nel tempo, emivita 730 giorni,
finestra forma di 3 partite. Ogni lega e' un mondo chiuso: le medie di
campionato e i confronti fra squadre restano interni alla lega, perche' i
livelli di gol e il vantaggio campo differiscono fra campionati.

Differenze rispetto a prototipo_gradient_boosting_multiliga.py, che calcolava
componenti simili per il solo training:
- usa `medie_lega_pesate` invece della media semplice su tutto lo storico di
  lega (era il bug #3 corretto per la Serie A: rapportare statistiche decadute
  a 2 anni a medie di 30 stagioni introduce un bias pro-casa del ~15%);
- restituisce anche stagione, data e lega, per poter selezionare la finestra di
  test e produrre breakdown per campionato;
- calcola le componenti SOLO per le partite delle stagioni di test, usando
  tutto lo storico precedente come training: molto piu' economico che
  calcolarle per l'intera storia di ogni lega.
"""
import os

import numpy as np
import pandas as pd

from modello import (stats_pesate_squadre, medie_lega_pesate, probabilita_shin,
                     distribuzione_punteggi, esiti_da_matrice)

CARTELLA_ALTRE_LEGHE = "altre_leghe"
FILE_SERIE_A = "serie_a.csv"

# Codice football-data -> nome leggibile. "I1" e' la Serie A, letta pero' da
# serie_a.csv (gia' unito e verificato da unisci_dati.py) e non dai file grezzi.
LEGHE = {"I1": "Serie A", "E0": "Premier League", "SP1": "Liga", "D1": "Bundesliga", "F1": "Ligue 1"}

HALF_LIFE, N_FORMA, RHO = 730, 3, -0.10
PESO_FORMA, PESO_SCONTRI, PESO_QUOTE = 0.10, 0.0, 0.90

# Finestra di storico passata alle funzioni di media pesata. Senza questo limite
# il costo cresce quadraticamente (ogni partita riscansiona tutto lo storico
# precedente); con emivita 730 giorni i contributi oltre ~4 stagioni pesano meno
# dell'1% e il troncamento non cambia il risultato in modo apprezzabile.
MAX_STORICO_PARTITE = 1600

# Cascate di priorita' per la quota di consenso, come in unisci_dati.py.
CASCATA_APERTURA = {"H": ["AvgH", "BbAvH", "B365H"], "D": ["AvgD", "BbAvD", "B365D"],
                    "A": ["AvgA", "BbAvA", "B365A"]}
CASCATA_CHIUSURA = {"H": ["AvgCH", "B365CH"], "D": ["AvgCD", "B365CD"], "A": ["AvgCA", "B365CA"]}

# Over/Under 2.5 gol: stessa logica a cascata, apertura e chiusura.
CASCATA_OU_APERTURA = {"Over25": ["Avg>2.5", "BbAv>2.5", "B365>2.5", "P>2.5"],
                       "Under25": ["Avg<2.5", "BbAv<2.5", "B365<2.5", "P<2.5"]}
CASCATA_OU_CHIUSURA = {"COver25": ["AvgC>2.5", "B365C>2.5", "PC>2.5"],
                       "CUnder25": ["AvgC<2.5", "B365C<2.5", "PC<2.5"]}


def stagione_da_data(date):
    """Stagione sportiva (agosto-maggio) etichettata con l'anno di inizio, la
    stessa convenzione usata da `Stagione` in serie_a.csv."""
    return date.dt.year - (date.dt.month < 7).astype(int)


def carica_lega(codice):
    """Storico completo di una lega, con colonne normalizzate ai nomi usati dal
    resto del progetto (OddsAvg*/OddsAvgC*), ordinato per data."""
    if codice == "I1":
        df = pd.read_csv(FILE_SERIE_A)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["FTHG", "FTAG", "Date"])
        df["Stagione"] = df["Stagione"].astype(int)
        return df.sort_values("Date", kind="stable").reset_index(drop=True)

    cartella = os.path.join(CARTELLA_ALTRE_LEGHE, codice)
    pezzi = []
    for file in sorted(os.listdir(cartella)):
        if not file.endswith(".txt"):
            continue
        percorso = os.path.join(cartella, file)
        try:
            df = pd.read_csv(percorso, encoding="latin1", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(percorso, encoding="cp1252", on_bad_lines="skip", engine="python")
        if not {"HomeTeam", "AwayTeam", "FTHG"} <= set(df.columns):
            continue
        for esito, candidate in CASCATA_APERTURA.items():
            colonna = next((c for c in candidate if c in df.columns), None)
            df[f"OddsAvg{esito}"] = pd.to_numeric(df[colonna], errors="coerce") if colonna else np.nan
        for esito, candidate in CASCATA_CHIUSURA.items():
            colonna = next((c for c in candidate if c in df.columns), None)
            df[f"OddsAvgC{esito}"] = pd.to_numeric(df[colonna], errors="coerce") if colonna else np.nan
        for nome, candidate in {**CASCATA_OU_APERTURA, **CASCATA_OU_CHIUSURA}.items():
            colonna = next((c for c in candidate if c in df.columns), None)
            df[f"Odds{nome}"] = pd.to_numeric(df[colonna], errors="coerce") if colonna else np.nan
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        pezzi.append(df.dropna(subset=["FTHG", "FTAG", "Date"]))

    lega = pd.concat(pezzi, ignore_index=True).sort_values("Date", kind="stable").reset_index(drop=True)
    lega["Stagione"] = stagione_da_data(lega["Date"])
    return lega


def _probabilita_over_under(riga):
    """Probabilita' di (Over 2.5, Under 2.5) implicite nelle quote, con la stessa
    cascata chiusura -> apertura usata per l'1X2. None se le quote mancano."""
    for prefisso in ("OddsC", "Odds"):
        q = [riga.get(f"{prefisso}Over25"), riga.get(f"{prefisso}Under25")]
        if all(pd.notna(v) for v in q) and all(float(v) > 1.0 for v in q):
            return probabilita_shin([float(v) for v in q])
    return None


def _probabilita_quote(riga):
    """Probabilita' implicite con la configurazione di produzione: quota di
    chiusura convertita con Shin, con fallback per-partita sull'apertura."""
    for prefisso in ("OddsAvgC", "OddsAvg"):
        q = [riga.get(f"{prefisso}{e}") for e in ("H", "D", "A")]
        if all(pd.notna(v) for v in q):
            return probabilita_shin(q)
    return None


def componenti_lega(df_lega, stagioni_test, codice_lega=""):
    """Componenti per-partita delle sole partite nelle stagioni di test, con
    tutto lo storico precedente della stessa lega come training (walk-forward:
    nessun dato successivo alla partita da prevedere).

    Restituisce una lista di dict compatibili con `backtesting.valuta_componente`,
    arricchiti con lega/stagione/data."""
    stagioni_test = {int(s) for s in stagioni_test}
    indici_test = [i for i, s in enumerate(df_lega["Stagione"].to_numpy()) if int(s) in stagioni_test]

    componenti = []
    for i in indici_test:
        riga = df_lega.iloc[i]
        casa, trasferta, data = riga["HomeTeam"], riga["AwayTeam"], riga["Date"]
        df_prima = df_lega.iloc[max(0, i - MAX_STORICO_PARTITE):i]
        if len(df_prima) < 100:
            continue

        m_casa, m_trasferta = medie_lega_pesate(df_prima, data, HALF_LIFE)
        m_generale = (m_casa + m_trasferta) / 2

        stats = stats_pesate_squadre(df_prima, data_riferimento=data, half_life_giorni=HALF_LIFE)
        c = stats[stats["Squadra"] == casa]
        t = stats[stats["Squadra"] == trasferta]
        if c.empty or t.empty:
            continue

        xg_cs = (c["gol_fatti_casa_storico"].values[0] / m_casa) * (t["gol_subiti_trasferta_storico"].values[0] / m_trasferta) * m_casa
        xg_ts = (t["gol_fatti_trasferta_storico"].values[0] / m_trasferta) * (c["gol_subiti_casa_storico"].values[0] / m_casa) * m_trasferta

        import backtesting as bt  # import locale: bt esegue codice Streamlit al caricamento
        fatti_c, subiti_c, fatti_c_home, _ = bt.calcola_forma_bt(df_prima, casa, len(df_prima), N_FORMA)
        fatti_t, subiti_t, _, fatti_t_away = bt.calcola_forma_bt(df_prima, trasferta, len(df_prima), N_FORMA)
        xg_cf = (fatti_c_home / m_casa) * (max(subiti_t, 0.3) / m_trasferta) * m_casa if fatti_c_home > 0 else xg_cs
        xg_tf = (fatti_t_away / m_trasferta) * (max(subiti_c, 0.3) / m_casa) * m_trasferta if fatti_t_away > 0 else xg_ts

        gf_sc, gs_sc = bt.scontri_diretti_bt(df_prima, casa, trasferta, ultimi_n=10)
        if gf_sc is not None:
            xg_c_sc, xg_t_sc, scontri_validi = (gf_sc / m_generale) * m_casa, (gs_sc / m_generale) * m_trasferta, True
        else:
            xg_c_sc, xg_t_sc, scontri_validi = xg_cs, xg_ts, False

        quote = _probabilita_quote(riga)
        quote_ou = _probabilita_over_under(riga)
        esito = "1" if riga["FTHG"] > riga["FTAG"] else ("X" if riga["FTHG"] == riga["FTAG"] else "2")
        over_reale = (riga["FTHG"] + riga["FTAG"]) > 2.5

        componenti.append({
            "xG_casa_storico": xg_cs, "xG_trasf_storico": xg_ts,
            "xG_casa_forma": xg_cf, "xG_trasf_forma": xg_tf,
            "xG_casa_scontri": xg_c_sc, "xG_trasf_scontri": xg_t_sc,
            "scontri_validi": scontri_validi,
            "quote_presenti": quote is not None,
            "prob_1_quote": quote[0] if quote else 0.0,
            "prob_X_quote": quote[1] if quote else 0.0,
            "prob_2_quote": quote[2] if quote else 0.0,
            "elo_casa": None, "elo_trasferta": None,
            "quote_ou_presenti": quote_ou is not None,
            "prob_over_mercato": quote_ou[0] if quote_ou else 0.0,
            "esito_over": bool(over_reale),
            "esito": esito, "stagione": str(int(riga["Stagione"])),
            "lega": codice_lega, "data": data,
        })
    return componenti


def probabilita_modello(componenti, peso_forma=PESO_FORMA, peso_scontri=PESO_SCONTRI,
                        peso_quote=PESO_QUOTE, rho=RHO):
    """Probabilita' (1, X, 2) del modello statistico per ogni componente.
    Reimplementa il blend di `backtesting.valuta_componente` senza dipendere
    dalle sue variabili globali di modulo (Elo, train/test di Serie A)."""
    risultati = []
    for comp in componenti:
        ps = peso_scontri if comp["scontri_validi"] else 0.0
        pf, pq = peso_forma, peso_quote
        totale = pf + ps + pq
        if totale > 1:
            pf, ps, pq = pf / totale, ps / totale, pq / totale
            peso_storico = 0.0
        else:
            peso_storico = 1 - pf - ps - pq
        if not comp["quote_presenti"]:
            peso_storico += pq
            pq = 0.0

        peso_xg = peso_storico + pf + ps
        if peso_xg > 0:
            xg_casa = (peso_storico * comp["xG_casa_storico"] + pf * comp["xG_casa_forma"]
                       + ps * comp["xG_casa_scontri"]) / peso_xg
            xg_trasf = (peso_storico * comp["xG_trasf_storico"] + pf * comp["xG_trasf_forma"]
                        + ps * comp["xG_trasf_scontri"]) / peso_xg
        else:
            xg_casa, xg_trasf = comp["xG_casa_storico"], comp["xG_trasf_storico"]

        esiti = esiti_da_matrice(distribuzione_punteggi(max(0.05, xg_casa), max(0.05, xg_trasf), rho=rho))
        p1, px, p2 = esiti["p_1"], esiti["p_X"], esiti["p_2"]
        if comp["quote_presenti"]:
            p1 = (1 - pq) * p1 + pq * comp["prob_1_quote"]
            px = (1 - pq) * px + pq * comp["prob_X_quote"]
            p2 = (1 - pq) * p2 + pq * comp["prob_2_quote"]
        risultati.append([p1, px, p2])
    return risultati


def probabilita_mercato(componenti):
    """Probabilita' del solo mercato (quota di chiusura con Shin)."""
    return [[c["prob_1_quote"], c["prob_X_quote"], c["prob_2_quote"]] for c in componenti]


def raccogli_tutte(stagioni_test, leghe=None, verbose=True):
    """Componenti di tutte le leghe richieste, concatenate."""
    leghe = leghe or list(LEGHE)
    tutte = []
    for codice in leghe:
        if verbose:
            print(f"  {LEGHE[codice]} ({codice})...", end="", flush=True)
        df = carica_lega(codice)
        comp = componenti_lega(df, stagioni_test, codice_lega=codice)
        con_quote = sum(1 for c in comp if c["quote_presenti"])
        if verbose:
            print(f" {len(comp)} partite ({con_quote} con quote)", flush=True)
        tutte.extend(comp)
    return tutte


def probabilita_over_modello(componenti, peso_forma=PESO_FORMA, peso_scontri=PESO_SCONTRI,
                             peso_quote=0.0, rho=RHO):
    """Probabilita' di Over 2.5 secondo il modello statistico.

    E' l'uscita NATIVA del modello Dixon-Coles: si legge sommando le celle della
    matrice dei punteggi con piu' di 2 gol totali, senza la perdita di
    informazione che comporta convertire i gol in "chi vince". Il blend con il
    mercato Over/Under (peso_quote) avviene qui, non dentro l'xG."""
    risultati = []
    for comp in componenti:
        ps = peso_scontri if comp["scontri_validi"] else 0.0
        peso_storico = max(0.0, 1 - peso_forma - ps)
        peso_xg = peso_storico + peso_forma + ps
        if peso_xg > 0:
            xg_casa = (peso_storico * comp["xG_casa_storico"] + peso_forma * comp["xG_casa_forma"]
                       + ps * comp["xG_casa_scontri"]) / peso_xg
            xg_trasf = (peso_storico * comp["xG_trasf_storico"] + peso_forma * comp["xG_trasf_forma"]
                        + ps * comp["xG_trasf_scontri"]) / peso_xg
        else:
            xg_casa, xg_trasf = comp["xG_casa_storico"], comp["xG_trasf_storico"]

        esiti = esiti_da_matrice(distribuzione_punteggi(max(0.05, xg_casa), max(0.05, xg_trasf), rho=rho))
        p_over = float(esiti["over_25"])
        if peso_quote > 0 and comp["quote_ou_presenti"]:
            p_over = (1 - peso_quote) * p_over + peso_quote * comp["prob_over_mercato"]
        risultati.append([p_over, 1.0 - p_over])
    return risultati


def probabilita_over_mercato(componenti):
    return [[c["prob_over_mercato"], 1.0 - c["prob_over_mercato"]] for c in componenti]
