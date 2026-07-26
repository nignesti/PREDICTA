"""
Test del modulo schedina.

La parte piu' delicata e' la distribuzione del numero di esiti corretti: le
partite hanno probabilita' diverse, quindi NON e' una binomiale e un'
implementazione sbagliata darebbe numeri plausibili ma falsi — esattamente il
tipo di errore che questo progetto ha gia' pagato caro. I test la verificano
contro casi calcolabili a mano e contro la binomiale nel caso degenere in cui
le probabilita' coincidono.
"""
import numpy as np
import pytest
from scipy.stats import binom

import schedina


def test_analizza_partita_sceglie_il_favorito():
    r = schedina.analizza_partita(1.50, 4.00, 7.00)
    assert r["pronostico"] == "1"
    assert r["confidenza"] > 0.5
    assert r["quota_pronostico"] == 1.50


def test_analizza_partita_riconosce_il_favorito_in_trasferta():
    r = schedina.analizza_partita(7.00, 4.00, 1.50)
    assert r["pronostico"] == "2"
    assert r["quota_pronostico"] == 1.50


def test_analizza_partita_probabilita_sommano_a_uno():
    r = schedina.analizza_partita(2.10, 3.40, 3.60)
    assert r["prob_1"] + r["prob_X"] + r["prob_2"] == pytest.approx(1.0, abs=1e-9)


def test_analizza_partita_calcola_il_margine():
    # Quote eque (1/2 + 1/4 + 1/4 = 1): margine zero.
    assert schedina.analizza_partita(2.0, 4.0, 4.0)["margine"] == pytest.approx(0.0, abs=1e-9)
    # Quote con overround del 5%.
    r = schedina.analizza_partita(1.90, 3.80, 3.80)
    assert r["margine"] == pytest.approx(1 / 1.90 + 2 / 3.80 - 1, abs=1e-9)


def test_analizza_partita_rifiuta_quote_non_valide():
    for quote in [(1.0, 3.0, 3.0), (0.5, 3.0, 3.0), (2.0, 1.0, 3.0)]:
        with pytest.raises(ValueError):
            schedina.analizza_partita(*quote)


def test_distribuzione_una_sola_partita():
    d = schedina.distribuzione_numero_esatti([0.7])
    assert d == pytest.approx([0.3, 0.7])


def test_distribuzione_due_partite_calcolata_a_mano():
    # p = 0.6 e 0.5: 0 corrette = 0.4*0.5 = 0.20
    #                1 corretta = 0.6*0.5 + 0.4*0.5 = 0.50
    #                2 corrette = 0.6*0.5 = 0.30
    d = schedina.distribuzione_numero_esatti([0.6, 0.5])
    assert d == pytest.approx([0.20, 0.50, 0.30])


def test_distribuzione_somma_a_uno_e_lunghezza_corretta():
    p = [0.51, 0.63, 0.72, 0.58, 0.81]
    d = schedina.distribuzione_numero_esatti(p)
    assert len(d) == len(p) + 1
    assert d.sum() == pytest.approx(1.0, abs=1e-12)
    assert (d >= 0).all()


def test_distribuzione_coincide_con_la_binomiale_se_le_probabilita_sono_uguali():
    # Caso degenere: con probabilita' tutte uguali la Poisson binomiale si
    # riduce alla binomiale. E' il controllo indipendente piu' forte.
    n, p = 13, 0.7
    d = schedina.distribuzione_numero_esatti([p] * n)
    attesa = binom.pmf(np.arange(n + 1), n, p)
    assert d == pytest.approx(attesa, abs=1e-12)


def test_distribuzione_estremi_coincidono_col_prodotto():
    p = [0.6, 0.7, 0.8]
    d = schedina.distribuzione_numero_esatti(p)
    assert d[-1] == pytest.approx(0.6 * 0.7 * 0.8)          # tutte corrette
    assert d[0] == pytest.approx(0.4 * 0.3 * 0.2)           # nessuna corretta


def test_riepiloga_schedina_moltiplicatore_e_probabilita():
    selezioni = [schedina.analizza_partita(1.50, 4.00, 7.00),
                 schedina.analizza_partita(1.80, 3.60, 4.50)]
    r = schedina.riepiloga_schedina(selezioni)

    assert r["n_partite"] == 2
    assert r["moltiplicatore"] == pytest.approx(1.50 * 1.80)
    assert r["p_tutte"] == pytest.approx(selezioni[0]["confidenza"] * selezioni[1]["confidenza"])
    assert r["una_su"] == pytest.approx(1 / r["p_tutte"])
    assert r["ritorno_atteso"] == pytest.approx(r["moltiplicatore"] * r["p_tutte"])


def test_riepiloga_schedina_ritorno_atteso_sempre_sotto_uno_con_margine():
    # Con quote che incorporano un margine reale, il ritorno atteso di una
    # multipla deve risultare < 1: e' il punto centrale documentato nel modulo.
    selezioni = [schedina.analizza_partita(1.50, 4.20, 6.50) for _ in range(13)]
    r = schedina.riepiloga_schedina(selezioni)
    assert r["ritorno_atteso"] < 1.0
    assert r["margine_medio"] > 0


def test_riepiloga_schedina_almeno_n_meno_1_e_piu_probabile_della_piena():
    selezioni = [schedina.analizza_partita(1.60, 3.90, 5.50) for _ in range(10)]
    r = schedina.riepiloga_schedina(selezioni)
    assert r["p_almeno_n_meno_1"] > r["p_tutte"]


def test_riepiloga_schedina_vuota_restituisce_none():
    assert schedina.riepiloga_schedina([]) is None


def test_ordina_per_confidenza():
    partite = [schedina.analizza_partita(2.50, 3.20, 2.80),
               schedina.analizza_partita(1.30, 5.00, 9.00),
               schedina.analizza_partita(1.90, 3.50, 4.00)]
    ordinate = schedina.ordina_per_confidenza(partite)
    confidenze = [p["confidenza"] for p in ordinate]
    assert confidenze == sorted(confidenze, reverse=True)
    assert ordinate[0]["quota_pronostico"] == 1.30


def test_selezionare_le_piu_sicure_alza_la_probabilita_di_schedina_piena():
    # E' l'affermazione centrale del modulo: la selezione delle partite conta
    # piu' del modello. Con le stesse partite disponibili, prendere le 5 piu'
    # sicure deve battere nettamente il prenderne 5 a caso fra tutte.
    quote = [(1.25, 6.0, 11.0), (1.40, 4.8, 8.0), (1.55, 4.2, 6.0), (1.70, 3.9, 4.8),
             (1.90, 3.6, 4.0), (2.20, 3.3, 3.3), (2.60, 3.2, 2.7), (3.00, 3.3, 2.4)]
    partite = [schedina.analizza_partita(*q) for q in quote]

    migliori = schedina.ordina_per_confidenza(partite)[:5]
    peggiori = schedina.ordina_per_confidenza(partite)[-5:]

    assert (schedina.riepiloga_schedina(migliori)["p_tutte"]
            > 3 * schedina.riepiloga_schedina(peggiori)["p_tutte"])


# ------------------------------------------------------------
# Over / Under 2.5
# ------------------------------------------------------------

def test_over_under_sceglie_l_esito_piu_probabile():
    assert schedina.analizza_over_under(1.50, 2.60)["pronostico"] == "Over 2.5"
    assert schedina.analizza_over_under(2.60, 1.50)["pronostico"] == "Under 2.5"


def test_over_under_probabilita_sommano_a_uno():
    r = schedina.analizza_over_under(1.90, 1.95)
    assert r["prob_over"] + r["prob_under"] == pytest.approx(1.0, abs=1e-9)


def test_over_under_toglie_il_margine():
    # Quote 1.90/1.95: la somma di 1/quota supera 1, il margine va rimosso.
    r = schedina.analizza_over_under(1.90, 1.95)
    assert r["margine"] == pytest.approx(1 / 1.90 + 1 / 1.95 - 1, abs=1e-9)
    assert r["prob_over"] < 1 / 1.90   # la probabilita' vera e' sotto quella grezza


def test_over_under_blend_col_modello_sposta_la_previsione():
    # Mercato quasi in equilibrio, modello convinto dell'Under: con peso_quote
    # basso deve prevalere il modello, con peso_quote=1 solo il mercato.
    solo_mercato = schedina.analizza_over_under(1.90, 1.95, prob_over_modello=0.20, peso_quote=1.0)
    con_modello = schedina.analizza_over_under(1.90, 1.95, prob_over_modello=0.20, peso_quote=0.5)

    assert solo_mercato["pronostico"] == "Over 2.5"
    assert con_modello["pronostico"] == "Under 2.5"
    assert con_modello["prob_over"] < solo_mercato["prob_over"]


def test_over_under_peso_quote_uno_ignora_il_modello():
    con = schedina.analizza_over_under(1.80, 2.05, prob_over_modello=0.05, peso_quote=1.0)
    senza = schedina.analizza_over_under(1.80, 2.05)
    assert con["prob_over"] == pytest.approx(senza["prob_over"])


def test_over_under_rifiuta_quote_non_valide():
    with pytest.raises(ValueError):
        schedina.analizza_over_under(1.0, 2.0)


def test_schedina_puo_mescolare_1x2_e_over_under():
    # riepiloga_schedina deve funzionare su una lista mista: e' il motivo per cui
    # analizza_over_under restituisce lo stesso formato di analizza_partita.
    selezioni = [
        schedina.analizza_partita(1.50, 4.00, 7.00),
        schedina.analizza_over_under(1.70, 2.15),
        schedina.analizza_partita(1.80, 3.60, 4.50),
    ]
    r = schedina.riepiloga_schedina(selezioni)
    assert r["n_partite"] == 3
    assert r["moltiplicatore"] == pytest.approx(1.50 * 1.70 * 1.80)
    assert 0 < r["p_tutte"] < 1


def test_over_under_e_spesso_piu_sicuro_dell_1x2():
    # Con due esiti invece di tre, l'esito piu' probabile parte da una base piu'
    # alta: e' il motivo per cui includere l'Over/Under migliora la schedina.
    equilibrata = schedina.analizza_partita(2.80, 3.30, 2.60)      # 1X2 molto incerto
    ou = schedina.analizza_over_under(1.55, 2.45)                   # Over abbastanza netto
    assert ou["confidenza"] > equilibrata["confidenza"]


# ------------------------------------------------------------
# Doppia chance
# ------------------------------------------------------------
def test_doppia_chance_sceglie_la_coppia_piu_probabile():
    # Con un favorito netto in casa, la coppia piu' probabile e' 1X: prende il
    # favorito e il pareggio, e lascia fuori solo l'esito meno probabile.
    r = schedina.analizza_doppia_chance(1.50, 4.00, 7.00)
    assert r["pronostico"] == "1X"
    assert r["esiti_coperti"] == ("1", "X")
    assert r["mercato"] == "Doppia chance"


def test_doppia_chance_con_favorito_in_trasferta():
    r = schedina.analizza_doppia_chance(7.00, 4.00, 1.50)
    assert r["pronostico"] == "X2"


def test_doppia_chance_quota_e_realizzabile_con_due_giocate_1x2():
    # La quota non e' una stima: e' il pagamento di una posta divisa fra i due
    # esiti in proporzione a 1/q. Se questa identita' si rompe, il
    # moltiplicatore mostrato in pagina promette piu' di quanto si incassi.
    q1, qx, q2 = 1.50, 4.00, 7.00
    r = schedina.analizza_doppia_chance(q1, qx, q2)
    q_dc = r["quota_pronostico"]
    assert q_dc == pytest.approx(1.0 / (1.0 / q1 + 1.0 / qx))

    # Verifica diretta: 1 euro diviso fra "1" e "X", stesso incasso in entrambi i casi.
    peso_1 = (1 / q1) / (1 / q1 + 1 / qx)
    assert peso_1 * q1 == pytest.approx(q_dc)
    assert (1 - peso_1) * qx == pytest.approx(q_dc)


def test_doppia_chance_confidenza_e_la_somma_dei_due_esiti():
    singola = schedina.analizza_partita(2.10, 3.40, 3.60)
    dc = schedina.analizza_doppia_chance(2.10, 3.40, 3.60)
    coperti = {"1": singola["prob_1"], "X": singola["prob_X"], "2": singola["prob_2"]}
    atteso = sum(coperti[e] for e in dc["esiti_coperti"])
    assert dc["confidenza"] == pytest.approx(atteso)


def test_doppia_chance_e_sempre_piu_probabile_del_miglior_singolo():
    # E' il motivo per cui esiste nello strumento: rende giocabile una schedina
    # lunga dove l'1X2 non arriva. Deve valere su qualunque terna di quote.
    for quote in [(1.50, 4.00, 7.00), (2.80, 3.30, 2.60), (7.00, 3.90, 1.48),
                  (1.22, 6.00, 12.00), (2.50, 3.10, 2.80)]:
        singola = schedina.analizza_partita(*quote)
        dc = schedina.analizza_doppia_chance(*quote)
        assert dc["confidenza"] > singola["confidenza"]
        # ...e sempre a un moltiplicatore piu' basso: non e' un pasto gratis.
        assert dc["quota_pronostico"] < singola["quota_pronostico"]


def test_doppia_chance_accetta_probabilita_del_blend():
    # In app.py le probabilita' arrivano dal blend modello+mercato, non da Shin:
    # la funzione deve usarle invece di ricalcolarle dalle quote.
    r = schedina.analizza_doppia_chance(2.10, 3.40, 3.60, probabilita=[0.10, 0.20, 0.70])
    assert r["pronostico"] == "X2"
    assert r["confidenza"] == pytest.approx(0.90)


def test_doppia_chance_margine_positivo_su_quote_con_overround():
    r = schedina.analizza_doppia_chance(1.90, 3.60, 4.20)
    assert r["margine"] > 0
    # Su quote eque il margine sparisce.
    equo = schedina.analizza_doppia_chance(2.0, 4.0, 4.0)
    assert equo["margine"] == pytest.approx(0.0, abs=1e-9)


def test_doppia_chance_rifiuta_quote_non_valide():
    with pytest.raises(ValueError):
        schedina.analizza_doppia_chance(1.0, 3.40, 3.60)


def test_schedina_puo_mescolare_tutti_e_tre_i_mercati():
    selezioni = [
        schedina.analizza_partita(1.50, 4.00, 7.00),
        schedina.analizza_doppia_chance(2.80, 3.30, 2.60),
        schedina.analizza_over_under(1.70, 2.15),
    ]
    r = schedina.riepiloga_schedina(selezioni)
    assert r["n_partite"] == 3
    assert 0 < r["p_tutte"] < 1
