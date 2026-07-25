"""
Scarica lo storico di altre leghe europee da football-data.co.uk (Fase 3, punto
3 della ROADMAP: estensione multi-campionato per aumentare il volume dati
della componente ML). Stessa fonte e stesso formato grezzo gia' usato per
stagioni/*.txt (Serie A, codice divisione I1): qui si scaricano Premier
League (E0), Liga (SP1), Bundesliga (D1) e Ligue 1 (F1), stesso intervallo di
stagioni (1993-2025).

Salva in altre_leghe/{codice}/{anno}.txt, con lo stesso formato di
stagioni/{anno}.txt cosi' unisci_dati.py e il resto della pipeline possono
essere riusati identici, cambiando solo la cartella di origine. Cartella
gitignored (dati grezzi rigenerabili con questo script, come stagioni/ non lo
e' perche' venne scaricata manualmente all'inizio del progetto, ma altre_leghe/
lo e' fin dall'origine per non appesantire il repository).
"""
import os
import time
import requests

LEGHE = {"E0": "Premier League", "SP1": "Liga", "D1": "Bundesliga", "F1": "Ligue 1"}
STAGIONI = list(range(1993, 2026))  # stesso intervallo di stagioni/*.txt
CARTELLA_BASE = "altre_leghe"
PAUSA_SECONDI = 1.0  # cortesia verso il server, stessa logica di scarica_elo.py


def codice_stagione(anno):
    """1993 -> '9394', 2005 -> '0506', 2023 -> '2324' (convenzione football-data.co.uk)."""
    fine = (anno + 1) % 100
    return f"{anno % 100:02d}{fine:02d}"


def scarica_lega(codice_lega, cartella_lega):
    os.makedirs(cartella_lega, exist_ok=True)
    scaricate, saltate = 0, 0
    for anno in STAGIONI:
        percorso = os.path.join(cartella_lega, f"{anno}.txt")
        if os.path.exists(percorso):
            saltate += 1
            continue
        url = f"https://www.football-data.co.uk/mmz4281/{codice_stagione(anno)}/{codice_lega}.csv"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 0:
                with open(percorso, "wb") as f:
                    f.write(r.content)
                scaricate += 1
                print(f"  {codice_lega} {anno}: OK ({len(r.content):,} byte)")
            else:
                print(f"  {codice_lega} {anno}: HTTP {r.status_code}, salto")
        except requests.RequestException as e:
            print(f"  {codice_lega} {anno}: errore ({e}), salto")
        time.sleep(PAUSA_SECONDI)
    return scaricate, saltate


if __name__ == "__main__":
    for codice, nome in LEGHE.items():
        print(f"=== {nome} ({codice}) ===")
        cartella_lega = os.path.join(CARTELLA_BASE, codice)
        scaricate, saltate = scarica_lega(codice, cartella_lega)
        print(f"  Totale: {scaricate} scaricate, {saltate} gia' presenti\n")
    print("Fatto.")
