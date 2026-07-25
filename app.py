"""
PredictA — costruttore di schedina (homepage).

Inserisci le quote 1X2 di una giornata e la pagina calcola pronostico,
confidenza e — soprattutto — la probabilita' reale che la schedina esca piena.

Il modello di produzione assegna il 100% del peso al mercato: la previsione per
la singola partita e' la probabilita' implicita nelle quote, ripulita dal
margine del bookmaker con la correzione di Shin. Misurato su 12.459 partite di
5 campionati (2019-2025), e' il miglior previsore 1X2 disponibile e nessuna
componente statistica aggiunta lo migliora (vedi ROADMAP.md).

La leva vera su una schedina non e' quindi il modello, ma QUALI partite
includere: giocare il favorito su tutte le partite porta una schedina da 13 allo
0,033%, giocarlo solo su quelle con confidenza >= 70% la porta al 3,6%.
"""
import numpy as np
import pandas as pd
import streamlit as st

import schedina as sc

st.set_page_config(
    page_title="PredictA — Schedina",
    page_icon=":material/receipt_long:",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLONNE = ["Casa", "Trasferta", "Quota 1", "Quota X", "Quota 2"]
DATA_FILE = "serie_a.csv"


@st.cache_data
def carica_storico():
    """Storico Serie A, usato solo per precaricare una giornata di esempio."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    colonne_quota = ("OddsAvgCH", "OddsAvgCD", "OddsAvgCA")
    if not all(c in df.columns for c in colonne_quota):
        return None
    df = df.dropna(subset=list(colonne_quota))
    return df.sort_values("Date")


def giornata_di_esempio(df, n=13):
    """Ultime n partite disponibili nello storico, come riga di partenza."""
    ultime = df.tail(n)
    return pd.DataFrame({
        "Casa": ultime["HomeTeam"].to_list(),
        "Trasferta": ultime["AwayTeam"].to_list(),
        "Quota 1": ultime["OddsAvgCH"].round(2).to_list(),
        "Quota X": ultime["OddsAvgCD"].round(2).to_list(),
        "Quota 2": ultime["OddsAvgCA"].round(2).to_list(),
    })


def tabella_vuota(n=13):
    return pd.DataFrame({c: [None] * n for c in COLONNE})


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.logo("serie_a_logo.svg", size="large")
    st.markdown("### :material/tune: Come comporre la schedina")

    with st.container(border=True):
        soglia = st.slider(
            "Confidenza minima", 0.35, 0.85, 0.60, 0.05,
            help="Include solo le partite in cui l'esito piu' probabile supera questa soglia. "
                 "E' la leva piu' importante: misurato su 12.459 partite, una schedina da 13 "
                 "passa dallo 0,03% (tutte le partite) al 3,6% (solo confidenza >= 70%).",
        )
        max_partite = st.slider(
            "Numero massimo di partite", 2, 20, 13, 1,
            help="Ogni partita aggiunta moltiplica la quota ma riduce la probabilita' di schedina piena, "
                 "e aggiunge un altro margine del bookmaker al ritorno atteso.",
        )

    with st.container(border=True):
        st.markdown("**Come si legge**")
        st.caption(
            "La **confidenza** e' la probabilita' vera dell'esito, tolto il margine del bookmaker "
            "(correzione di Shin). E' calibrata: sulle partite in cui il modello dichiara 70-80%, "
            "l'esito si e' verificato nel 74,8% dei casi su 12.459 partite."
        )

# ------------------------------------------------------------
# INTESTAZIONE
# ------------------------------------------------------------
st.title("PredictA — Costruttore di schedina", text_alignment="center")
st.markdown(
    "Inserisci le quote 1X2 della giornata: la pagina sceglie i pronostici piu' solidi "
    "e ti dice la probabilita' reale che la schedina esca piena.",
    text_alignment="center",
)

st.space("medium")

storico = carica_storico()

with st.container(border=True):
    col_a, col_b = st.columns([3, 1], vertical_alignment="bottom")
    with col_a:
        st.markdown("**1. Inserisci le quote**")
        st.caption("Una riga per partita. Puoi incollare i dati da un foglio di calcolo, "
                   "oppure aggiungere righe con il + in fondo alla tabella.")
    with col_b:
        usa_esempio = st.toggle(
            "Precarica un esempio", value=False,
            help="Riempie la tabella con le ultime partite dello storico di Serie A, "
                 "per provare lo strumento senza digitare le quote.",
            disabled=storico is None,
        )

    if usa_esempio and storico is not None:
        partenza = giornata_di_esempio(storico, max_partite)
    else:
        partenza = tabella_vuota(max_partite)

    quote_inserite = st.data_editor(
        partenza,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key=f"editor_{usa_esempio}_{max_partite}",
        column_config={
            "Casa": st.column_config.TextColumn("Casa", width="medium"),
            "Trasferta": st.column_config.TextColumn("Trasferta", width="medium"),
            "Quota 1": st.column_config.NumberColumn("Quota 1", min_value=1.01, step=0.01, format="%.2f"),
            "Quota X": st.column_config.NumberColumn("Quota X", min_value=1.01, step=0.01, format="%.2f"),
            "Quota 2": st.column_config.NumberColumn("Quota 2", min_value=1.01, step=0.01, format="%.2f"),
        },
    )

# ------------------------------------------------------------
# ANALISI
# ------------------------------------------------------------
partite, scartate = [], 0
for _, riga in quote_inserite.iterrows():
    quote = [riga.get("Quota 1"), riga.get("Quota X"), riga.get("Quota 2")]
    if any(pd.isna(q) for q in quote):
        continue
    try:
        analisi = sc.analizza_partita(*quote)
    except (ValueError, TypeError):
        scartate += 1
        continue
    analisi["casa"] = riga.get("Casa") or "—"
    analisi["trasferta"] = riga.get("Trasferta") or "—"
    partite.append(analisi)

if scartate:
    st.warning(f":material/warning: {scartate} riga/e ignorata/e: le quote decimali devono essere maggiori di 1.")

if not partite:
    st.info(":material/info: Inserisci le quote di almeno una partita, oppure attiva "
            "**Precarica un esempio** per vedere subito come funziona.")
    st.stop()

ordinate = sc.ordina_per_confidenza(partite)
selezionate = [p for p in ordinate if p["confidenza"] >= soglia][:max_partite]
escluse = [p for p in ordinate if p not in selezionate]

st.space("medium")

if not selezionate:
    migliore = ordinate[0]["confidenza"]
    st.warning(
        f":material/warning: Nessuna partita raggiunge la confidenza minima del {soglia:.0%}. "
        f"La piu' sicura e' al {migliore:.0%}: abbassa la soglia nella barra laterale."
    )
    st.stop()

riepilogo = sc.riepiloga_schedina(selezionate)

# --- Metriche principali ---
st.markdown("### 2. La tua schedina")
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown(f"## {riepilogo['n_partite']}")
        st.caption("Partite selezionate")
        st.badge(f"su {len(partite)} inserite", color="gray")

with col2:
    with st.container(border=True):
        st.markdown(f"## {riepilogo['moltiplicatore']:,.0f}x".replace(",", "."))
        st.caption("Moltiplicatore")
        st.badge("quota totale", color="blue")

with col3:
    with st.container(border=True):
        st.markdown(f"## {riepilogo['p_tutte']:.2%}")
        st.caption("Probabilita' schedina piena")
        st.badge(f"1 su {riepilogo['una_su']:,.0f}".replace(",", "."), color="orange")

with col4:
    ritorno = riepilogo["ritorno_atteso"]
    with st.container(border=True):
        st.markdown(f"## {ritorno:.0%}")
        st.caption("Ritorno atteso")
        st.badge("per ogni euro giocato", color="red" if ritorno < 1 else "green")

st.caption(
    f":material/info: Il ritorno atteso e' probabilita' x moltiplicatore. Resta sotto il 100% perche' "
    f"le quote incorporano il margine del bookmaker ({riepilogo['margine_medio']:.1%} in media su queste "
    f"partite), che si somma a ogni partita aggiunta. Nessun modello puo' portarlo sopra 1: servirebbe un "
    f"vantaggio informativo sul mercato, che questo progetto ha misurato non esserci (vedi ROADMAP.md)."
)

st.space("medium")

# --- Distribuzione degli esiti corretti ---
col_sx, col_dx = st.columns([3, 2], gap="medium")

with col_sx:
    with st.container(border=True):
        st.markdown("**Pronostici selezionati**")
        tabella = pd.DataFrame([{
            "Partita": f"{p['casa']} – {p['trasferta']}",
            "Esito": p["pronostico"],
            "Confidenza": f"{p['confidenza']:.0%}",
            "Quota": f"{p['quota_pronostico']:.2f}",
        } for p in selezionate])
        st.dataframe(tabella, hide_index=True, width="stretch")

with col_dx:
    with st.container(border=True):
        st.markdown("**Quanti ne azzecchi, realisticamente**")
        distribuzione = riepilogo["distribuzione"]
        n = riepilogo["n_partite"]
        atteso = float(np.dot(np.arange(n + 1), distribuzione))
        st.metric("Esiti corretti attesi", f"{atteso:.1f} su {n}")

        righe_dist = []
        for k in range(n, max(-1, n - 5), -1):
            righe_dist.append({
                "Esiti corretti": f"{k} su {n}",
                "Probabilita'": f"{distribuzione[k]:.2%}",
            })
        st.dataframe(pd.DataFrame(righe_dist), hide_index=True, width="stretch")
        st.caption(
            f"Sbagliarne al massimo una ha probabilita' **{riepilogo['p_almeno_n_meno_1']:.1%}**, "
            f"molto piu' della schedina piena: se il tuo concorso premia anche il {n-1}, "
            f"e' l'esito su cui contare."
        )

# --- Partite escluse ---
if escluse:
    with st.expander(f":material/filter_alt: {len(escluse)} partite escluse dalla schedina"):
        st.caption("Sotto la soglia di confidenza, oppure oltre il numero massimo di partite. "
                   "Sono le piu' incerte: includerle abbassa la probabilita' di schedina piena.")
        st.dataframe(pd.DataFrame([{
            "Partita": f"{p['casa']} – {p['trasferta']}",
            "Esito piu' probabile": p["pronostico"],
            "Confidenza": f"{p['confidenza']:.0%}",
        } for p in escluse]), hide_index=True, width="stretch")

st.space("large")
st.caption(
    ":material/warning: Strumento dimostrativo ed educativo. Non costituisce invito al gioco d'azzardo. "
    "Il ritorno atteso di una schedina e' strutturalmente inferiore alla posta giocata: "
    "il gioco d'azzardo puo' causare dipendenza.",
    text_alignment="center",
)
