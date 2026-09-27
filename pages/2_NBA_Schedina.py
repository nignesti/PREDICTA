"""
PredictA — costruttore di schedina, sezione NBA.

Stessa idea della sezione Serie A (pages/1_Serie_A_Schedina.py) ma sui mercati
tipici delle scommesse NBA, tutti a due esiti perche' l'NBA non ha pareggi:

- **Moneyline**: chi vince la partita.
- **Spread**: chi copre l'handicap sui punti (es. Casa -4.5).
- **Totale punti**: se il totale (Casa+Trasferta) supera una linea (es. Over 224.5).

Il modello e' un rating Elo con moltiplicatore sul margine di vittoria (stile
FiveThirtyEight), non Dixon-Coles: quel modello stima gol attesi per un
mercato a tre esiti col pareggio, che qui non esiste. Vedi modello_nba.py e
pronostico_nba.py per i dettagli, scarica_nba.py / unisci_dati_nba.py per
come si genera nba_storico.csv.

A differenza della prima versione di questa pagina, ora esiste uno storico
quote (nba_quote_storico.csv, da nba_2010-2026.csv via unisci_quote_nba.py):
come per la Serie A, "Precarica un esempio" prende una giornata reale con le
sue quote di moneyline e spread/totale, invece di partire da zero.
"""
import numpy as np
import pandas as pd
import streamlit as st

import modello_nba as mn
import pronostico_nba as pn
import quote_live_nba
import schedina as sc
import schedina_nba as sn
import unisci_quote_nba
import valida_nba

st.set_page_config(
    page_title="PredictA — Schedina NBA",
    page_icon=":material/sports_basketball:",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    SQUADRE = pn.squadre_disponibili()
except pn.DatiNBANonDisponibili as errore:
    st.title("PredictA — Costruttore di schedina (NBA)", text_alignment="center")
    st.warning(f":material/warning: {errore}")
    st.stop()

COLONNE = ["Casa", "Trasferta",
           "Quota vittoria Casa", "Quota vittoria Trasferta",
           "Linea spread (Casa)", "Quota spread Casa", "Quota spread Trasferta",
           "Linea totale", "Quota Over", "Quota Under"]
MERCATI = [sn.MONEYLINE, sn.SPREAD, sn.TOTALE]
REGOLE = {
    "Il più probabile": "confidenza",
    "Quello che paga di più, sopra soglia": "quota",
}
# I dati grezzi (nba_2010-2026.csv) riportano la linea reale di spread/totale ma
# non una quota per coprirla: a differenza del moneyline, l'americano -110 su
# entrambi i lati e' quasi universale nel mercato NBA, quindi e' l'unica scelta
# ragionevole per precompilare l'esempio senza inventare un numero a caso.
QUOTA_STANDARD_SPREAD_TOTALE = 1.91


def tabella_vuota(n):
    return pd.DataFrame({c: [None] * n for c in COLONNE})


def _americano_a_decimale(moneyline):
    ml = np.asarray(moneyline, dtype=float)
    positivo = ml > 0
    decimale = np.empty_like(ml)
    decimale[positivo] = 1 + ml[positivo] / 100.0
    decimale[~positivo] = 1 + 100.0 / -ml[~positivo]
    return decimale


@st.cache_data
def giornata_di_esempio(n):
    """Ultime n partite dello storico con moneyline reale nota, come riga di
    partenza per chi vuole provare lo strumento subito. Spread e totale usano
    la linea reale (dal file quote) ma la quota standard del mercato USA
    (QUOTA_STANDARD_SPREAD_TOTALE): i dati grezzi non includono una quota per
    quei due mercati, solo la linea (vedi unisci_quote_nba.py)."""
    quote = unisci_quote_nba.carica_quote(unisci_quote_nba.FILE_GREZZO)
    d = quote.dropna(subset=["moneyline_home", "moneyline_away", "spread", "total"]).tail(n)
    spread_casa = valida_nba.spread_casa_segnato(d)
    return pd.DataFrame({
        "Casa": d["HomeTeam"].to_list(),
        "Trasferta": d["AwayTeam"].to_list(),
        "Quota vittoria Casa": _americano_a_decimale(d["moneyline_home"]).round(2),
        "Quota vittoria Trasferta": _americano_a_decimale(d["moneyline_away"]).round(2),
        "Linea spread (Casa)": spread_casa.round(1),
        "Quota spread Casa": QUOTA_STANDARD_SPREAD_TOTALE,
        "Quota spread Trasferta": QUOTA_STANDARD_SPREAD_TOTALE,
        "Linea totale": d["total"].round(1).to_list(),
        "Quota Over": QUOTA_STANDARD_SPREAD_TOTALE,
        "Quota Under": QUOTA_STANDARD_SPREAD_TOTALE,
    })


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### :material/tune: Peso del modello")
    with st.container(border=True):
        peso_quote = st.slider(
            "Quote bookmaker (moneyline)", 0.0, 1.0, 1.0, 0.05,
            help="Peso delle quote moneyline che inserisci rispetto al rating Elo. Default 1.0: "
                 "misurato su 8.919 partite di 7 stagioni (valida_nba.py, walk-forward, "
                 "moneyline reale + proxy da spread dove il moneyline manca), l'Elo calibrato "
                 "batte 'vince sempre la casa' ma perde contro il mercato (Brier peggiore di "
                 "+0.011, accuratezza -2.62 punti). Stessa conclusione del calcio (ROADMAP.md): "
                 "il mercato resta il miglior previsore disponibile.")
        st.caption(f"Rating Elo: **{1 - peso_quote:.0%}**. Spread e totale usano sempre e solo "
                   "le quote: sullo storico i modelli di quei due mercati fanno peggio di una "
                   "moneta (valida_nba.py).")

    st.markdown("### :material/receipt_long: Composizione")
    with st.container(border=True):
        soglia = st.slider("Confidenza minima", 0.35, 0.95, 0.60, 0.05)
        max_partite = st.slider("Numero massimo di partite", 2, 15, 8, 1)
        mercati = st.multiselect(
            "Mercati ammessi", MERCATI, default=MERCATI,
            help="Per ogni partita si gioca un solo esito: quello con la confidenza piu' "
                 "alta fra i mercati che spunti qui e per cui hai inserito le quote.")
        regola = st.radio("Con più esiti sopra soglia, gioca", list(REGOLE))

# ------------------------------------------------------------
# INTESTAZIONE
# ------------------------------------------------------------
st.title("PredictA — Costruttore di schedina (NBA)", text_alignment="center")
st.markdown(
    "Scegli le squadre della giornata e inserisci le quote per moneyline, spread e/o "
    "totale punti: il modello combina il rating Elo con il mercato e compone la schedina "
    "piu' solida.",
    text_alignment="center",
)

_, regressione_applicata, ultima_partita = pn.rating_per_data()
if regressione_applicata:
    st.info(f":material/update: Storico fermo al {ultima_partita:%d/%m/%Y}, prima della stagione in corso: "
            "i rating Elo sono regrediti di 1/3 verso la media per le rose cambiate in estate. "
            "Scambi e infortuni della offseason non sono nel modello: nelle prime settimane "
            "affidati alle quote.")

st.space("medium")

with st.container(border=True):
    col_a, col_b = st.columns([3, 1], vertical_alignment="bottom")
    with col_a:
        st.markdown("**1. Partite e quote**")
        st.caption(f"Scegli le squadre dai menu a tendina ({len(SQUADRE)} squadre presenti nello "
                   "storico) e inserisci le quote decimali dei mercati che vuoi usare — lascia "
                   "vuote quelle dei mercati che non ti interessano. Aggiungi righe con il + in "
                   "fondo alla tabella.")
    with col_b:
        fonti = ["Nessuna", "Esempio storico"]
        if "quote_live_nba" in st.session_state:
            fonti.append("Quote live (Pinnacle)")
        precarica = st.selectbox(
            "Precarica", fonti,
            help="Le quote live compaiono qui dopo averle scaricate nella pagina "
                 "'Quote live e valore' (nessun credito in più).")

    if precarica == "Esempio storico":
        partenza = giornata_di_esempio(max_partite)
    elif precarica == "Quote live (Pinnacle)":
        partenza = quote_live_nba.quote_sharp_per_schedina(
            quote_live_nba.quote_in_tabella(st.session_state.quote_live_nba[0])
        ).reindex(columns=COLONNE)
    else:
        partenza = tabella_vuota(max_partite)

    inserite = st.data_editor(
        partenza,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key=f"editor_nba_{precarica}_{max_partite}",
        column_config={
            "Casa": st.column_config.SelectboxColumn("Casa", options=SQUADRE, width="small"),
            "Trasferta": st.column_config.SelectboxColumn("Trasferta", options=SQUADRE, width="small"),
            "Quota vittoria Casa": st.column_config.NumberColumn("Vitt. Casa", min_value=1.01, step=0.01, format="%.2f"),
            "Quota vittoria Trasferta": st.column_config.NumberColumn("Vitt. Trasf.", min_value=1.01, step=0.01, format="%.2f"),
            "Linea spread (Casa)": st.column_config.NumberColumn("Linea spread", step=0.5, format="%.1f",
                                                                  help="Handicap della squadra di Casa: negativo se favorita (es. -4.5)."),
            "Quota spread Casa": st.column_config.NumberColumn("Quota spread Casa", min_value=1.01, step=0.01, format="%.2f"),
            "Quota spread Trasferta": st.column_config.NumberColumn("Quota spread Trasf.", min_value=1.01, step=0.01, format="%.2f"),
            "Linea totale": st.column_config.NumberColumn("Linea totale", step=0.5, format="%.1f"),
            "Quota Over": st.column_config.NumberColumn("Quota Over", min_value=1.01, step=0.01, format="%.2f"),
            "Quota Under": st.column_config.NumberColumn("Quota Under", min_value=1.01, step=0.01, format="%.2f"),
        },
    )

    with st.expander(":material/help: Legenda colonne"):
        st.markdown(
            "- **Casa / Trasferta** — le due squadre della partita.\n"
            "- **Vitt. Casa / Vitt. Trasf.** — quota decimale (es. 1.85) per la vittoria "
            "della rispettiva squadra, mercato **moneyline**. Lascia entrambe vuote se non "
            "vuoi giocare questo mercato su questa partita.\n"
            "- **Linea spread** — l'handicap sui punti della squadra di Casa, con il segno: "
            "**negativo** se la Casa è favorita (es. **-4.5** vuol dire che deve vincere di "
            "più di 4.5 punti per coprire), **positivo** se è sfavorita (es. **+6.5**).\n"
            "- **Quota spread Casa / Quota spread Trasf.** — quota decimale per coprire "
            "quella linea, rispettivamente per Casa e Trasferta (di solito vicine a 1.91, "
            "corrispondente al taglio standard USA -110).\n"
            "- **Linea totale** — il totale punti (Casa + Trasferta) su cui si gioca Over/Under "
            "(es. 224.5).\n"
            "- **Quota Over / Quota Under** — quota decimale per il totale sopra o sotto "
            "quella linea.\n\n"
            "Non serve compilare tutte le colonne di ogni riga: basta una coppia completa "
            "(le due quote di un mercato, più la linea per spread e totale) per far comparire "
            "quel mercato tra i candidati della partita."
        )

# ------------------------------------------------------------
# CALCOLO, PARTITA PER PARTITA
# ------------------------------------------------------------
partite, problemi = [], []


def _quote_valide(*valori):
    return all(v is not None and not pd.isna(v) and float(v) > 1.0 for v in valori)


for numero, (_, riga) in enumerate(inserite.iterrows(), start=1):
    casa, trasferta = riga.get("Casa"), riga.get("Trasferta")
    casa = None if pd.isna(casa) else casa
    trasferta = None if pd.isna(trasferta) else trasferta

    tutte_le_quote = (riga.get("Quota vittoria Casa"), riga.get("Quota vittoria Trasferta"),
                      riga.get("Linea spread (Casa)"), riga.get("Quota spread Casa"), riga.get("Quota spread Trasferta"),
                      riga.get("Linea totale"), riga.get("Quota Over"), riga.get("Quota Under"))
    if not casa and not trasferta and all(pd.isna(v) for v in tutte_le_quote):
        continue
    if not casa or not trasferta:
        problemi.append(f"riga {numero}: scegli entrambe le squadre")
        continue
    if casa == trasferta:
        problemi.append(f"riga {numero}: {casa} non può giocare contro sé stessa")
        continue

    modello = pn.stima_probabilita_nba(casa, trasferta)
    if modello is None:
        problemi.append(f"riga {numero}: dati storici insufficienti per {casa} o {trasferta}")
        continue

    comune = {
        "casa": casa, "trasferta": trasferta,
        "elo_casa": modello["elo_casa"], "elo_trasferta": modello["elo_trasferta"],
    }
    candidati = []

    if _quote_valide(riga.get("Quota vittoria Casa"), riga.get("Quota vittoria Trasferta")):
        q1, q2 = float(riga["Quota vittoria Casa"]), float(riga["Quota vittoria Trasferta"])
        ml = sn.analizza_mercato_due_vie(sn.MONEYLINE, casa, trasferta, q1, q2,
                                         prob_1_modello=modello["p_casa_modello"], peso_quote=peso_quote)
        ml_mercato = sn.analizza_mercato_due_vie(sn.MONEYLINE, casa, trasferta, q1, q2)
        candidati.append(dict(comune, mercato=sn.MONEYLINE,
                              pronostico=ml["pronostico"], confidenza=ml["confidenza"],
                              quota_pronostico=ml["quota_pronostico"], margine=ml["margine"],
                              solo_mercato=f"{ml_mercato['pronostico']} ({ml_mercato['confidenza']:.0%})",
                              solo_modello=f"{casa if modello['p_casa_modello'] >= 0.5 else trasferta} "
                                          f"({max(modello['p_casa_modello'], 1 - modello['p_casa_modello']):.0%})"))

    linea_spread = riga.get("Linea spread (Casa)")
    if (linea_spread is not None and not pd.isna(linea_spread)
            and _quote_valide(riga.get("Quota spread Casa"), riga.get("Quota spread Trasferta"))):
        linea_spread = float(linea_spread)
        q1, q2 = float(riga["Quota spread Casa"]), float(riga["Quota spread Trasferta"])
        p_copre_modello = mn.probabilita_copre_spread(modello["margine_atteso"], modello["sigma_margine"], linea_spread)
        esito_casa = f"{casa} {linea_spread:+.1f}"
        esito_trasferta = f"{trasferta} {-linea_spread:+.1f}"
        # Niente blend: sul test 2020-2026 il modello Elo->margine ha Brier
        # peggiore di una probabilita' fissa al 50% (valida_nba.confronto_spread),
        # quindi mescolarlo alle quote puo' solo peggiorarle. Resta mostrato
        # nella colonna "Solo modello" come riferimento.
        sp = sn.analizza_mercato_due_vie(sn.SPREAD, esito_casa, esito_trasferta, q1, q2)
        sp_mercato = sp
        candidati.append(dict(comune, mercato=sn.SPREAD,
                              pronostico=sp["pronostico"], confidenza=sp["confidenza"],
                              quota_pronostico=sp["quota_pronostico"], margine=sp["margine"],
                              solo_mercato=f"{sp_mercato['pronostico']} ({sp_mercato['confidenza']:.0%})",
                              solo_modello=f"{esito_casa if p_copre_modello >= 0.5 else esito_trasferta} "
                                          f"({max(p_copre_modello, 1 - p_copre_modello):.0%})"))

    linea_totale = riga.get("Linea totale")
    if (linea_totale is not None and not pd.isna(linea_totale)
            and _quote_valide(riga.get("Quota Over"), riga.get("Quota Under"))):
        linea_totale = float(linea_totale)
        q1, q2 = float(riga["Quota Over"]), float(riga["Quota Under"])
        p_over_modello = mn.probabilita_over_totale(modello["totale_atteso"], modello["sigma_totale"], linea_totale)
        esito_over, esito_under = f"Over {linea_totale:g}", f"Under {linea_totale:g}"
        # Niente blend, stesso motivo dello spread (valida_nba.confronto_totale).
        tt = sn.analizza_mercato_due_vie(sn.TOTALE, esito_over, esito_under, q1, q2)
        tt_mercato = tt
        candidati.append(dict(comune, mercato=sn.TOTALE,
                              pronostico=tt["pronostico"], confidenza=tt["confidenza"],
                              quota_pronostico=tt["quota_pronostico"], margine=tt["margine"],
                              solo_mercato=f"{tt_mercato['pronostico']} ({tt_mercato['confidenza']:.0%})",
                              solo_modello=f"{esito_over if p_over_modello >= 0.5 else esito_under} "
                                          f"({max(p_over_modello, 1 - p_over_modello):.0%})"))

    ammessi = [c for c in candidati if c["mercato"] in mercati]
    if not candidati:
        problemi.append(f"riga {numero} ({casa}-{trasferta}): manca almeno una coppia di quote complete")
    elif ammessi:
        sopra = [c for c in ammessi if c["confidenza"] >= soglia]
        if REGOLE[regola] == "quota" and sopra:
            partite.append(max(sopra, key=lambda c: c["quota_pronostico"]))
        else:
            partite.append(max(ammessi, key=lambda c: c["confidenza"]))

for messaggio in problemi:
    st.warning(f":material/warning: {messaggio}")

if not mercati:
    st.warning(":material/warning: Spunta almeno un mercato nella barra laterale.")
    st.stop()

if not partite:
    st.info(":material/info: Compila almeno una partita con una coppia di quote complete "
            "(vittoria, spread o totale).")
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
        st.markdown(f"## {riepilogo['moltiplicatore']:,.1f}x".replace(",", "."))
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
    f"incorporano il margine del bookmaker ({riepilogo['margine_medio']:.1%} in media su queste partite)."
)

st.space("medium")

# --- Dettaglio ---
col_sx, col_dx = st.columns([3, 2], gap="medium")

with col_sx:
    with st.container(border=True):
        st.markdown("**Pronostici selezionati**")
        st.dataframe(pd.DataFrame([{
            "Partita": f"{p['casa']} – {p['trasferta']}",
            "Mercato": p["mercato"],
            "Esito": p["pronostico"],
            "Confidenza": f"{p['confidenza']:.0%}",
            "Quota": f"{p['quota_pronostico']:.2f}",
            "Elo": f"{p['elo_casa']:.0f} – {p['elo_trasferta']:.0f}",
            "Solo mercato": p["solo_mercato"],
            "Solo modello": p["solo_modello"],
        } for p in selezionate]), hide_index=True, width="stretch")
        st.caption(
            "**Solo mercato** è ciò che dicono le sole quote; **solo modello** è ciò che dice "
            "il rating Elo senza le quote. La colonna Confidenza è il blend dei due per il "
            "moneyline, le sole quote per spread e totale."
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

if escluse:
    with st.expander(f":material/filter_alt: {len(escluse)} partite escluse"):
        st.dataframe(pd.DataFrame([{
            "Partita": f"{p['casa']} – {p['trasferta']}",
            "Mercato": p["mercato"],
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
