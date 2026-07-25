"""
Logica di previsione per la singola partita.

Estratto da app.py quando la pagina principale e' diventata il costruttore di
schedina: qui resta il modello (caricamento dati, forma, scontri diretti, blend
con le quote), mentre pages/1_Pronostico.py contiene solo l'interfaccia. La
separazione permette anche ai test di importare il modello senza far girare la UI.
"""
import numpy as np
import pandas as pd
import streamlit as st

from modello import (stats_pesate_squadre, distribuzione_punteggi, esiti_da_matrice,
                     probabilita_shin, medie_lega_pesate)

# Iperparametri Dixon-Coles validati via backtest (vedi pages/backtesting.py):
# EMIVITA_GIORNI: dopo quanti giorni una partita storica pesa la metà nelle medie
#   squadra (decadimento esponenziale, invece di media semplice su 33 stagioni).
# RHO_DIXON_COLES: correzione per i punteggi bassi (0-0, 1-0, 0-1, 1-1), dove un
#   Poisson indipendente sottostima sistematicamente i pareggi.
EMIVITA_GIORNI = 730
RHO_DIXON_COLES = -0.10
# PARTITE_FORMA: quante partite recenti entrano nella componente "forma". Deve
# restare allineato al default dello slider "Partite per forma" in
# pages/backtesting.py: è il valore risultato migliore nella grid search, e con i
# pesi di default (peso_storico = 0 per costruzione) è l'unico iperparametro che
# determina l'xG mostrato dalla dashboard. Prima era 5 qui e 3 nel backtest,
# quindi la dashboard non girava sulla configurazione validata.
PARTITE_FORMA = 3

# ------------------------------------------------------------
# DATI
# ------------------------------------------------------------
DATA_FILE = "serie_a.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    return df

df = load_data(DATA_FILE)
# Medie di lega con lo STESSO decadimento temporale delle statistiche di squadra.
# Con la media semplice su 33 stagioni (1.5135/1.1433 gol casa/trasferta) rapportata
# a statistiche decadute con emivita 730 giorni (che descrivono gli ultimi 2-4 anni,
# 1.3932/1.2121) si confrontavano due epoche diverse: il rapporto fra xG di casa e di
# trasferta risultava distorto di circa il 15% a favore della casa su ogni partita.
# Validato in valida_medie_lega.py: sul modello statistico puro vale +1.41 punti
# percentuali (49.96% contro 48.55%) e riduce le previsioni "1" dal 76.9% al 72.2%.
media_gol_casa, media_gol_trasferta = medie_lega_pesate(
    df, data_riferimento=df["Date"].max(), half_life_giorni=EMIVITA_GIORNI)
vantaggio_casa = media_gol_casa / media_gol_trasferta

# ------------------------------------------------------------
# FUNZIONI DI CALCOLO (condivise)
# ------------------------------------------------------------
def calcola_forma(df, squadra, ultime_n=PARTITE_FORMA):
    casa = df[df["HomeTeam"] == squadra].tail(ultime_n)
    trasferta = df[df["AwayTeam"] == squadra].tail(ultime_n)
    if len(casa) == 0 and len(trasferta) == 0:
        return None, None, None, None, [], []

    # Unisce casa e trasferta in ordine cronologico prima di tagliare le ultime N,
    # altrimenti (essendo trasferta aggiunta per seconda) la coda prenderebbe solo partite in trasferta.
    partite = pd.concat([casa, trasferta]).sort_index().tail(ultime_n)

    gol_fatti, gol_subiti, risultati = [], [], []
    for _, row in partite.iterrows():
        if row["HomeTeam"] == squadra:
            gol_fatti.append(row["FTHG"]); gol_subiti.append(row["FTAG"])
            risultati.append("V" if row["FTHG"] > row["FTAG"] else ("N" if row["FTHG"] == row["FTAG"] else "P"))
        else:
            gol_fatti.append(row["FTAG"]); gol_subiti.append(row["FTHG"])
            risultati.append("V" if row["FTAG"] > row["FTHG"] else ("N" if row["FTAG"] == row["FTHG"] else "P"))

    media_fatti = np.mean(gol_fatti) if gol_fatti else 0
    media_subiti = np.mean(gol_subiti) if gol_subiti else 0
    media_fatti_casa = casa["FTHG"].mean() if len(casa) > 0 else 0
    media_fatti_trasferta = trasferta["FTAG"].mean() if len(trasferta) > 0 else 0
    return media_fatti, media_subiti, media_fatti_casa, media_fatti_trasferta, risultati, gol_fatti

def scontri_diretti(df, squadra1, squadra2, ultimi_n=10):
    scontri = df[((df["HomeTeam"] == squadra1) & (df["AwayTeam"] == squadra2)) |
                 ((df["HomeTeam"] == squadra2) & (df["AwayTeam"] == squadra1))].tail(ultimi_n)
    if len(scontri) == 0:
        return None, None, None, None, None, None
    gol_fatti_s1, gol_subiti_s1 = [], []
    vittorie_s1, pareggi, vittorie_s2 = 0, 0, 0
    for _, row in scontri.iterrows():
        if row["HomeTeam"] == squadra1:
            gol_fatti_s1.append(row["FTHG"]); gol_subiti_s1.append(row["FTAG"])
            if row["FTHG"] > row["FTAG"]: vittorie_s1 += 1
            elif row["FTHG"] == row["FTAG"]: pareggi += 1
            else: vittorie_s2 += 1
        else:
            gol_fatti_s1.append(row["FTAG"]); gol_subiti_s1.append(row["FTHG"])
            if row["FTAG"] > row["FTHG"]: vittorie_s1 += 1
            elif row["FTAG"] == row["FTHG"]: pareggi += 1
            else: vittorie_s2 += 1
    return np.mean(gol_fatti_s1), np.mean(gol_subiti_s1), vittorie_s1, pareggi, vittorie_s2, scontri

# Statistiche storiche pesate nel tempo (le partite recenti contano di più di
# quelle di 30 anni fa) invece di una media semplice su tutta la storia.
stats = stats_pesate_squadre(df, data_riferimento=df["Date"].max(), half_life_giorni=EMIVITA_GIORNI)

def stima_probabilita(df, stats, squadra_casa, squadra_trasferta,
                      peso_forma=0.0, peso_scontri=0.0, peso_quote=1.0):
    """
    Combina: storico + forma + scontri diretti + quote dei bookmaker
    """
    casa = stats[stats["Squadra"] == squadra_casa]
    trasferta = stats[stats["Squadra"] == squadra_trasferta]
    if casa.empty or trasferta.empty:
        return None

    # --- STORICO ---
    # Forza attacco/difesa relativa alla media di campionato (stile Poisson classico),
    # non una media semplice: mediare ripetutamente con media_gol_casa/trasferta annulla
    # quasi ogni differenza tra squadre e fa collassare il modello su "vince sempre la casa".
    attacco_casa_storico = casa["gol_fatti_casa_storico"].values[0]
    difesa_casa_storico = casa["gol_subiti_casa_storico"].values[0]
    attacco_trasf_storico = trasferta["gol_fatti_trasferta_storico"].values[0]
    difesa_trasf_storico = trasferta["gol_subiti_trasferta_storico"].values[0]

    xG_casa_storico = (attacco_casa_storico / media_gol_casa) * (difesa_trasf_storico / media_gol_trasferta) * media_gol_casa
    xG_trasf_storico = (attacco_trasf_storico / media_gol_trasferta) * (difesa_casa_storico / media_gol_casa) * media_gol_trasferta

    # --- FORMA ---
    fatti_casa_forma, subiti_casa_forma, fatti_casa_home, _, _, _ = calcola_forma(df, squadra_casa)
    fatti_trasf_forma, subiti_trasf_forma, _, fatti_trasf_away, _, _ = calcola_forma(df, squadra_trasferta)

    if fatti_casa_forma is None:
        fatti_casa_forma, subiti_casa_forma, fatti_casa_home = attacco_casa_storico, difesa_casa_storico, attacco_casa_storico
    if fatti_trasf_forma is None:
        fatti_trasf_forma, subiti_trasf_forma, fatti_trasf_away = attacco_trasf_storico, difesa_trasf_storico, attacco_trasf_storico

    if fatti_casa_home > 0:
        xG_casa_forma = (fatti_casa_home / media_gol_casa) * (max(subiti_trasf_forma, 0.3) / media_gol_trasferta) * media_gol_casa
    else:
        xG_casa_forma = xG_casa_storico
    if fatti_trasf_away > 0:
        xG_trasf_forma = (fatti_trasf_away / media_gol_trasferta) * (max(subiti_casa_forma, 0.3) / media_gol_casa) * media_gol_trasferta
    else:
        xG_trasf_forma = xG_trasf_storico

    # --- SCONTRI DIRETTI ---
    media_gol_generale = (media_gol_casa + media_gol_trasferta) / 2
    scontri = scontri_diretti(df, squadra_casa, squadra_trasferta, ultimi_n=10)
    if scontri[0] is not None:
        gol_fatti_scontri, gol_subiti_scontri, _, _, _, _ = scontri
        xG_casa_scontri = (gol_fatti_scontri / media_gol_generale) * media_gol_casa
        xG_trasf_scontri = (gol_subiti_scontri / media_gol_generale) * media_gol_trasferta
    else:
        xG_casa_scontri, xG_trasf_scontri = xG_casa_storico, xG_trasf_storico
        peso_scontri = 0

    # --- QUOTE BOOKMAKER (NOVITÀ) ---
    # Prendiamo le quote medie degli ultimi scontri diretti (se disponibili).
    # Preferiamo la quota di consenso multi-bookmaker (OddsAvg*) a Bet365 da solo:
    # meno rumore da un singolo book, copertura pressoché totale dal 2011 in poi.
    quote_presenti = False
    prob_1_quote, prob_X_quote, prob_2_quote = 0, 0, 0
    colonne_quota = ("OddsAvgH", "OddsAvgD", "OddsAvgA") if "OddsAvgH" in df.columns else ("B365H", "B365D", "B365A")
    colonne_chiusura = ("OddsAvgCH", "OddsAvgCD", "OddsAvgCA")

    if colonne_quota[0] in df.columns and scontri[0] is not None:
        _, _, _, _, _, tabella_scontri = scontri
        if colonne_quota[0] in tabella_scontri.columns:
            # Gli scontri diretti includono ENTRAMBI gli orientamenti (Milan-Inter e
            # Inter-Milan): le colonne *H/*A si riferiscono alla squadra di casa di
            # QUELLA riga, non a squadra_casa. Usarle senza filtrare mescolava la
            # quota su Milan con quella su Inter, appiattendo la previsione verso la
            # parità e cancellando il vantaggio campo (su Milan-Inter: p_1 0.41
            # invece di 0.27, p_2 0.35 invece di 0.49 — errore di ~14 punti
            # percentuali su una componente pesata 0.90).
            # Teniamo solo le partite con lo stesso orientamento di quella da
            # prevedere: è la grandezza che ci serve davvero ("quanto paga il
            # mercato squadra_casa che ospita squadra_trasferta"), e conserva il
            # vantaggio campo invece di mediarlo via.
            stesso_orientamento = tabella_scontri["HomeTeam"] == squadra_casa
            tabella_quote = tabella_scontri[stesso_orientamento]
            if tabella_quote.empty:
                # Nessun precedente con questo orientamento (es. una sola sfida,
                # giocata in casa dell'altra): ripieghiamo su tutti i precedenti
                # scambiando H e A dove le squadre erano invertite. Meno preciso
                # (il vantaggio campo si perde) ma meglio che scartare le quote.
                tabella_quote = tabella_scontri
                inverti = tabella_quote["HomeTeam"] != squadra_casa
            else:
                inverti = pd.Series(False, index=tabella_quote.index)

            # Preferiamo la quota di CHIUSURA (a ridosso del fischio d'inizio,
            # incorpora più informazione di mercato dell'apertura): validata su 3
            # stagioni indipendenti in pages/backtesting.py, migliora l'accuratezza
            # media di 0.43 punti percentuali. Disponibile solo dal 2019: dove manca
            # (scontri diretti più vecchi) ricadiamo sulla quota di apertura per
            # quella singola partita, non sull'intera media.
            if all(c in tabella_quote.columns for c in colonne_chiusura):
                quota_h = tabella_quote[colonne_chiusura[0]].combine_first(tabella_quote[colonne_quota[0]])
                quota_d = tabella_quote[colonne_chiusura[1]].combine_first(tabella_quote[colonne_quota[1]])
                quota_a = tabella_quote[colonne_chiusura[2]].combine_first(tabella_quote[colonne_quota[2]])
            else:
                quota_h, quota_d, quota_a = (tabella_quote[colonne_quota[0]], tabella_quote[colonne_quota[1]],
                                             tabella_quote[colonne_quota[2]])

            # Riorienta: quota_1 = quota sulla vittoria di squadra_casa comunque.
            quota_1 = quota_h.where(~inverti, quota_a)
            quota_2 = quota_a.where(~inverti, quota_h)

            quote_valide = pd.DataFrame({"H": quota_1, "D": quota_d, "A": quota_2}).dropna()
            if len(quote_valide) > 0:
                # Converti quote in probabilità implicite e fai la media
                prob_1_quote = (1 / quote_valide["H"]).mean()
                prob_X_quote = (1 / quote_valide["D"]).mean()
                prob_2_quote = (1 / quote_valide["A"]).mean()
                # Rimuovi il margine del bookmaker con la correzione di Shin
                # (1992/1993) invece della normalizzazione proporzionale
                # semplice: valida su 3 stagioni indipendenti in
                # pages/backtesting.py, migliora leggermente l'RPS (calibrazione)
                # senza peggiorare l'accuratezza. "Quote equivalenti" perché qui
                # partiamo da probabilità già mediate su più scontri diretti,
                # non dalle quote di una singola partita.
                quote_equivalenti = [1 / prob_1_quote, 1 / prob_X_quote, 1 / prob_2_quote]
                prob_1_quote, prob_X_quote, prob_2_quote = probabilita_shin(quote_equivalenti)
                quote_presenti = True

    # --- COMBINAZIONE ---
    # I pesi ora sono: storico + forma + scontri + quote = 1
    peso_totale = peso_forma + peso_scontri + peso_quote
    if peso_totale > 1:
        # Normalizza se supera 1
        peso_forma /= peso_totale
        peso_scontri /= peso_totale
        peso_quote /= peso_totale
        peso_storico_final = 0
    else:
        peso_storico_final = 1 - peso_forma - peso_scontri - peso_quote

    if not quote_presenti:
        # Se non ci sono quote, ridistribuisci il peso
        peso_storico_final += peso_quote
        peso_quote = 0

    # Combinazione pesata per i gol attesi: storico+forma+scontri vanno rinormalizzati
    # a sommare 1 tra loro, perché "quote" non entra qui (entra dopo, sulle probabilità
    # finali) — altrimenti con peso_quote alto (es. 0.85) i pesi restanti (es. 0.15 di
    # scontri) scalano l'xG verso il basso invece di usarlo per intero.
    peso_xg_totale = peso_storico_final + peso_forma + peso_scontri
    if peso_xg_totale > 0:
        xG_casa = (peso_storico_final * xG_casa_storico +
                   peso_forma * xG_casa_forma +
                   peso_scontri * xG_casa_scontri) / peso_xg_totale
        xG_trasferta = (peso_storico_final * xG_trasf_storico +
                        peso_forma * xG_trasf_forma +
                        peso_scontri * xG_trasf_scontri) / peso_xg_totale
    else:
        xG_casa, xG_trasferta = xG_casa_storico, xG_trasf_storico

    # Il vantaggio campo è già incorporato sopra (ogni componente è scalata su
    # media_gol_casa o media_gol_trasferta), quindi qui non va riapplicato: raddoppiarlo
    # schiacciava il modello su "vince sempre la casa" a prescindere dalle squadre.

    # --- DISTRIBUZIONE ESATTA DEI PUNTEGGI (Dixon-Coles) ---
    # Poisson indipendenti + correzione tau per i punteggi bassi, al posto della
    # simulazione Monte Carlo: stesso modello concettuale ma deterministico (niente
    # rumore campionario) e senza sottostimare sistematicamente i pareggi.
    matrice_punteggi = distribuzione_punteggi(xG_casa, xG_trasferta, rho=RHO_DIXON_COLES)
    esiti = esiti_da_matrice(matrice_punteggi)
    p_1_base, p_X_base, p_2_base = esiti["p_1"], esiti["p_X"], esiti["p_2"]

    # Se abbiamo le quote, facciamo un blend finale
    if quote_presenti:
        p_1 = (1 - peso_quote) * p_1_base + peso_quote * prob_1_quote
        p_X = (1 - peso_quote) * p_X_base + peso_quote * prob_X_quote
        p_2 = (1 - peso_quote) * p_2_base + peso_quote * prob_2_quote
    else:
        p_1, p_X, p_2 = p_1_base, p_X_base, p_2_base

    top_risultati = esiti["top_risultati"]
    over_25, over_15, under_25 = esiti["over_25"], esiti["over_15"], esiti["under_25"]
    gol_totali_attesi = xG_casa + xG_trasferta

    return {
        "xG_casa": xG_casa,
        "xG_trasferta": xG_trasferta,
        "p_1": p_1,
        "p_X": p_X,
        "p_2": p_2,
        "p_1_base": p_1_base,  # senza quote
        "p_X_base": p_X_base,
        "p_2_base": p_2_base,
        "quote_presenti": quote_presenti,
        "prob_1_quote": prob_1_quote if quote_presenti else None,
        "prob_X_quote": prob_X_quote if quote_presenti else None,
        "prob_2_quote": prob_2_quote if quote_presenti else None,
        "top_risultati": top_risultati,
        "over_25": over_25,
        "under_25": 1 - over_25,
        "over_15": over_15,
        "gol_totali_attesi": gol_totali_attesi,
        "scontri": scontri
    }
