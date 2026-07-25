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
