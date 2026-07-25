"""
PredictA — costruttore di schedina (homepage).

Scegli le squadre di una giornata, inserisci le quote, e la pagina calcola per
ogni partita il pronostico completo — storico pesato nel tempo, forma recente,
scontri diretti e quote di mercato — poi compone la schedina piu' solida e ne
misura la probabilita' reale di uscire piena.

A differenza della pagina "Pronostico", qui le quote usate nel blend sono quelle
della partita da prevedere (inserite da te), non la media dei precedenti scontri
diretti: e' il dato corretto, perche' descrive proprio l'incontro in questione.

Nota sui pesi. Misurato su 12.459 partite di 5 campionati, la componente
statistica non migliora la previsione rispetto alle sole quote (vedi ROADMAP.md).
I pesi restano regolabili e di default tutte e quattro le componenti
contribuiscono: puoi confrontare in pagina cosa dice il modello e cosa dice il
solo mercato, partita per partita.
"""
import numpy as np
import pandas as pd
import streamlit as st

import schedina as sc
from pronostico import df as df_storico, stats, stima_probabilita

st.set_page_config(
    page_title="PredictA — Schedina",
    page_icon=":material/receipt_long:",
    layout="wide",
    initial_sidebar_state="expanded",
)

SQUADRE = sorted(stats["Squadra"].unique().tolist())
COLONNE = ["Casa", "Trasferta", "Quota 1", "Quota X", "Quota 2"]


@st.cache_data
def giornata_di_esempio(n):
    """Ultime n partite dello storico con quote di chiusura note, come
    riga di partenza per chi vuole provare lo strumento subito."""
    d = df_storico.dropna(subset=["OddsAvgCH", "OddsAvgCD", "OddsAvgCA"]).tail(n)
    return pd.DataFrame({
        "Casa": d["HomeTeam"].to_list(),
        "Trasferta": d["AwayTeam"].to_list(),
        "Quota 1": d["OddsAvgCH"].round(2).to_list(),
        "Quota X": d["OddsAvgCD"].round(2).to_list(),
        "Quota 2": d["OddsAvgCA"].round(2).to_list(),
    })


def tabella_vuota(n):
    return pd.DataFrame({c: [None] * n for c in COLONNE})


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.logo("serie_a_logo.svg", size="large")

    st.markdown("### :material/tune: Pesi del modello")
    with st.container(border=True):
        peso_forma = st.slider(
            "Forma recente", 0.0, 1.0, 0.15, 0.05,
            help="Rendimento nelle ultime 3 partite di ciascuna squadra.")
        peso_scontri = st.slider(
            "Scontri diretti", 0.0, 0.5, 0.15, 0.05,
            help="Media gol dei precedenti fra queste due squadre.")
        peso_quote = st.slider(
            "Quote bookmaker", 0.0, 1.0, 0.60, 0.05,
            help="Peso delle quote che inserisci. Misurato su 12.459 partite di 5 campionati, "
                 "il solo mercato (peso 1.0) e' risultato piu' accurato del blend: qui il default "
                 "lascia contribuire tutte le componenti, ma puoi confrontare i due nella tabella.")
        peso_storico = 1 - peso_forma - peso_scontri - peso_quote
        if peso_storico < 0:
            st.error(":material/error: La somma dei pesi supera il 100%")
        else:
            st.caption(f"Storico (media pesata nel tempo): **{peso_storico:.0%}**")

    st.markdown("### :material/receipt_long: Composizione")
    with st.container(border=True):
        soglia = st.slider(
            "Confidenza minima", 0.35, 0.85, 0.60, 0.05,
            help="Include solo le partite in cui l'esito piu' probabile supera questa soglia. "
                 "E' la leva piu' importante sulla probabilita' di schedina piena.")
        max_partite = st.slider("Numero massimo di partite", 2, 20, 13, 1)

# ------------------------------------------------------------
# INTESTAZIONE
# ------------------------------------------------------------
st.title("PredictA — Costruttore di schedina", text_alignment="center")
st.markdown(
    "Scegli le partite della giornata e inserisci le quote: il modello calcola storico, "
    "forma, scontri diretti e mercato, e compone la schedina piu' solida.",
    text_alignment="center",
)

st.space("medium")

with st.container(border=True):
    col_a, col_b = st.columns([3, 1], vertical_alignment="bottom")
    with col_a:
        st.markdown("**1. Partite e quote**")
        st.caption(f"Scegli le squadre dai menu a tendina ({len(SQUADRE)} squadre di Serie A "
                   "presenti nello storico) e inserisci le quote decimali. "
                   "Aggiungi righe con il + in fondo alla tabella.")
    with col_b:
        usa_esempio = st.toggle("Precarica un esempio", value=False)

    partenza = giornata_di_esempio(max_partite) if usa_esempio else tabella_vuota(max_partite)

    inserite = st.data_editor(
        partenza,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key=f"editor_{usa_esempio}_{max_partite}",
        column_config={
            "Casa": st.column_config.SelectboxColumn("Casa", options=SQUADRE, width="medium"),
            "Trasferta": st.column_config.SelectboxColumn("Trasferta", options=SQUADRE, width="medium"),
            "Quota 1": st.column_config.NumberColumn("Quota 1", min_value=1.01, step=0.01, format="%.2f"),
            "Quota X": st.column_config.NumberColumn("Quota X", min_value=1.01, step=0.01, format="%.2f"),
            "Quota 2": st.column_config.NumberColumn("Quota 2", min_value=1.01, step=0.01, format="%.2f"),
        },
    )

if peso_storico < 0:
    st.error(":material/error: Correggi i pesi nella barra laterale prima di continuare.")
    st.stop()

# ------------------------------------------------------------
# CALCOLO DEL MODELLO COMPLETO, PARTITA PER PARTITA
# ------------------------------------------------------------
ESITI = ("1", "X", "2")
partite, problemi = [], []

for numero, (_, riga) in enumerate(inserite.iterrows(), start=1):
    casa, trasferta = riga.get("Casa"), riga.get("Trasferta")
    quote = (riga.get("Quota 1"), riga.get("Quota X"), riga.get("Quota 2"))
    casa = None if pd.isna(casa) else casa
    trasferta = None if pd.isna(trasferta) else trasferta

    if not casa and not trasferta and all(pd.isna(q) for q in quote):
        continue
    if not casa or not trasferta:
        problemi.append(f"riga {numero}: scegli entrambe le squadre")
        continue
    if casa == trasferta:
        problemi.append(f"riga {numero}: {casa} non può giocare contro sé stessa")
        continue
    if any(pd.isna(q) for q in quote):
        problemi.append(f"riga {numero} ({casa}–{trasferta}): mancano una o più quote")
        continue
    if any(float(q) <= 1.0 for q in quote):
        problemi.append(f"riga {numero} ({casa}–{trasferta}): le quote devono essere maggiori di 1")
        continue

    modello = stima_probabilita(df_storico, stats, casa, trasferta,
                                peso_forma, peso_scontri, peso_quote,
                                quote_partita=tuple(float(q) for q in quote))
    if modello is None:
        problemi.append(f"riga {numero}: dati storici insufficienti per {casa} o {trasferta}")
        continue

    prob = [modello["p_1"], modello["p_X"], modello["p_2"]]
    i = int(np.argmax(prob))
    mercato = sc.analizza_partita(*[float(q) for q in quote])

    partite.append({
        "casa": casa, "trasferta": trasferta,
        "pronostico": ESITI[i], "confidenza": prob[i],
        "quota_pronostico": float(quote[i]),
        "margine": mercato["margine"],
        "prob_1": prob[0], "prob_X": prob[1], "prob_2": prob[2],
        "xG_casa": modello["xG_casa"], "xG_trasferta": modello["xG_trasferta"],
        "pronostico_mercato": mercato["pronostico"], "confidenza_mercato": mercato["confidenza"],
        "senza_quote": max(modello["p_1_base"], modello["p_X_base"], modello["p_2_base"]),
        "pronostico_senza_quote": ESITI[int(np.argmax([modello["p_1_base"], modello["p_X_base"], modello["p_2_base"]]))],
    })

for messaggio in problemi:
    st.warning(f":material/warning: {messaggio}")

if not partite:
    st.info(":material/info: Compila almeno una partita, oppure attiva **Precarica un esempio**.")
    st.stop()

ordinate = sc.ordina_per_confidenza(partite)
selezionate = [p for p in ordinate if p["confidenza"] >= soglia][:max_partite]
escluse = [p for p in ordinate if p not in selezionate]

st.space("medium")

if not selezionate:
    st.warning(f":material/warning: Nessuna partita raggiunge il {soglia:.0%} di confidenza "
               f"(la più sicura è al {ordinate[0]['confidenza']:.0%}). Abbassa la soglia.")
    st.stop()

riepilogo = sc.riepiloga_schedina(selezionate)

# --- Metriche ---
st.markdown("### 2. La tua schedina")
c1, c2, c3, c4 = st.columns(4, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown(f"## {riepilogo['n_partite']}")
        st.caption("Partite selezionate")
        st.badge(f"su {len(partite)} inserite", color="gray")
with c2:
    with st.container(border=True):
        st.markdown(f"## {riepilogo['moltiplicatore']:,.0f}x".replace(",", "."))
        st.caption("Moltiplicatore")
        st.badge("quota totale", color="blue")
with c3:
    with st.container(border=True):
        st.markdown(f"## {riepilogo['p_tutte']:.2%}")
        st.caption("Probabilità schedina piena")
        st.badge(f"1 su {riepilogo['una_su']:,.0f}".replace(",", "."), color="orange")
with c4:
    ritorno = riepilogo["ritorno_atteso"]
    with st.container(border=True):
        st.markdown(f"## {ritorno:.0%}")
        st.caption("Ritorno atteso")
        st.badge("per ogni euro giocato", color="red" if ritorno < 1 else "green")

st.caption(
    f":material/info: Ritorno atteso = probabilità × moltiplicatore. Resta sotto il 100% perché le quote "
    f"incorporano il margine del bookmaker ({riepilogo['margine_medio']:.1%} in media su queste partite), "
    f"che si somma a ogni partita aggiunta."
)

st.space("medium")

# --- Dettaglio ---
col_sx, col_dx = st.columns([3, 2], gap="medium")

with col_sx:
    with st.container(border=True):
        st.markdown("**Pronostici selezionati**")
        st.dataframe(pd.DataFrame([{
            "Partita": f"{p['casa']} – {p['trasferta']}",
            "Esito": p["pronostico"],
            "Confidenza": f"{p['confidenza']:.0%}",
            "Quota": f"{p['quota_pronostico']:.2f}",
            "Gol attesi": f"{p['xG_casa']:.1f} – {p['xG_trasferta']:.1f}",
            "Solo mercato": f"{p['pronostico_mercato']} ({p['confidenza_mercato']:.0%})",
            "Solo modello": f"{p['pronostico_senza_quote']} ({p['senza_quote']:.0%})",
        } for p in selezionate]), hide_index=True, width="stretch")
        st.caption(
            "**Solo mercato** è ciò che dicono le sole quote; **solo modello** è ciò che dicono "
            "storico, forma e scontri diretti senza le quote. La colonna Confidenza è il blend "
            "dei due, con i pesi che hai impostato: quando le due colonne divergono, il blend sta "
            "mediando due opinioni diverse."
        )

with col_dx:
    with st.container(border=True):
        st.markdown("**Quanti ne azzecchi, realisticamente**")
        distribuzione = riepilogo["distribuzione"]
        n = riepilogo["n_partite"]
        st.metric("Esiti corretti attesi", f"{float(np.dot(np.arange(n + 1), distribuzione)):.1f} su {n}")
        st.dataframe(pd.DataFrame([
            {"Esiti corretti": f"{k} su {n}", "Probabilità": f"{distribuzione[k]:.2%}"}
            for k in range(n, max(-1, n - 5), -1)
        ]), hide_index=True, width="stretch")
        st.caption(f"Sbagliarne al massimo una ha probabilità **{riepilogo['p_almeno_n_meno_1']:.1%}**, "
                   f"molto più della schedina piena.")

if escluse:
    with st.expander(f":material/filter_alt: {len(escluse)} partite escluse"):
        st.dataframe(pd.DataFrame([{
            "Partita": f"{p['casa']} – {p['trasferta']}",
            "Esito più probabile": p["pronostico"],
            "Confidenza": f"{p['confidenza']:.0%}",
        } for p in escluse]), hide_index=True, width="stretch")

st.space("large")
st.caption(
    ":material/warning: Strumento dimostrativo ed educativo. Non costituisce invito al gioco d'azzardo. "
    "Il ritorno atteso di una schedina è strutturalmente inferiore alla posta giocata: "
    "il gioco d'azzardo può causare dipendenza.",
    text_alignment="center",
)
