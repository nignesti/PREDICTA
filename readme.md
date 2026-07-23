# 🔮 PredictA - Pronostici Serie A

Dashboard predittiva per partite di Serie A basata su machine learning, analisi statistica e quote dei bookmaker.

---

## 📋 Funzionalità

### Dashboard Principale
- **Pronostico 1X2** per qualsiasi match storico di Serie A
- **Gol attesi (xG)** calcolati con modello Poisson
- **Risultati esatti** più probabili (top 5)
- **Over/Under** 1.5 e 2.5 con gauge visivo
- **Forma recente** visualizzata con pallini 🟢🟡🔴
- **Scontri diretti** con tabella degli ultimi precedenti
- **Quote bookmaker** integrate nel modello (Bet365, Pinnacle)
- **Slider interattivi** per regolare i pesi del modello

### Backtesting
- **Validazione walk-forward** su 1-5 stagioni recenti a scelta (ognuna ~380 partite)
- **Metriche**: Accuratezza 1X2, F1 Score, RPS e Log-loss (calibrazione delle probabilità, non solo la previsione più probabile), Matrice di Confusione, breakdown per stagione
- **Confronto tra configurazioni**: solo storico, storico+forma, vecchi default, solo quote, ottimale validato
- **Iperparametri Dixon-Coles regolabili**: emivita delle statistiche storiche e correzione tau per i pareggi

---

## 🧠 Il Modello

Il sistema combina **quattro componenti** con pesi regolabili:

| Componente | Descrizione | Peso Default |
|------------|-------------|--------------|
| **Media storica** | Gol fatti/subiti, pesati con decadimento temporale (le partite recenti contano di più) | 0% |
| **Forma recente** | Performance nelle ultime N partite | 10% |
| **Scontri diretti** | Precedenti tra le due squadre | 0% |
| **Quote bookmaker** | Saggezza del mercato (consenso multi-bookmaker, Bet365/Pinnacle a seguire) | 90% |

I pesi di default sono il risultato di una grid search su backtest walk-forward, **validata su tre stagioni indipendenti** (2023, 2024, 2025) e non solo su quella usata per cercarli — vedi la sezione Risultati sotto per i dettagli. Il risultato più sorprendente: né lo storico pesato nel tempo né gli scontri diretti aggiungono valore misurabile una volta pesate bene le quote (probabilmente perché il mercato incorpora già efficientemente quell'informazione); un piccolo peso alla forma recentissima invece aiuta in modo consistente. Restano comunque slider liberi se vuoi sperimentare altre combinazioni.

### Algoritmo
1. **Calcolo xG**: forza attacco/difesa di ogni squadra relativa alla media di campionato (stile Poisson classico), combinata pesando storico, forma e scontri diretti
2. **Vantaggio casa**: incorporato direttamente nelle formule (medie gol casa/trasferta separate), non come correzione aggiuntiva
3. **Decadimento temporale**: le statistiche storiche pesano ogni partita con un decadimento esponenziale (emivita configurabile, default 730 giorni) invece di una media semplice su 33 stagioni
4. **Distribuzione esatta dei punteggi**: calcolo diretto della matrice di probabilità Poisson (non più simulazione Monte Carlo), con la **correzione tau di Dixon & Coles (1997)** per i punteggi bassi — un Poisson indipendente puro sottostima sistematicamente i pareggi
5. **Blend quote**: le probabilità finali fondono il modello statistico con le probabilità implicite delle quote

---

## 📊 Risultati Backtesting

Validati su **tre stagioni di test indipendenti** (2023, 2024, 2025), ognuna con le stagioni precedenti come training — non solo sull'unica stagione su cui è girata la grid search, per escludere che i pesi scelti fossero semplicemente rumore statistico:

| Configurazione | Acc. 2025 | Acc. 2024 | Acc. 2023 | **Media** |
|---|---|---|---|---|
| Solo storico (pesato nel tempo) | 43-46%* | - | - | ~44% |
| Solo mercato (quote) | 54.4% | 52.6% | 54.7% | 53.9% |
| Vecchio default (forma=0.5, scontri=0.15, quote=0.15) | 50.3% | - | - | ~50% |
| Default precedente (forma=0, scontri=0.15, quote=0.85) | 54.1% | 53.4% | 52.9% | 53.5% |
| **Nuovo default (forma=0.10, scontri=0, quote=0.90)** | **56.2%** | **53.7%** | **54.7%** | **54.9%** |

*\*Numero di riferimento dalla validazione iniziale su una sola stagione, prima dell'introduzione del decadimento temporale.*

*Benchmark "predici sempre 1" sul test 2025: 38.9%. Oltre all'accuratezza, il backtesting misura anche RPS e log-loss (calibrazione delle probabilità): sulla configurazione ottimale, RPS medio ≈ 0.188 sulle tre stagioni (più basso è meglio; una previsione uniforme dà circa 0.28, una perfetta dà 0). Usa il bottone **🔬 Confronta configurazioni** nella pagina di Backtesting per riprodurre questi numeri con i tuoi dati.*

**Cosa abbiamo imparato costruendo questi numeri (in ordine cronologico di scoperta):**
1. Il calcolo della "forma recente" usava quasi solo le partite in trasferta invece delle ultime N in ordine cronologico — bug corretto.
2. Le formule storico/forma pesavano ogni statistica come media semplice con la media di campionato, azzerando le differenze reali tra squadre e facendo collassare il modello su "vince sempre la casa" — corretto usando una forza attacco/difesa relativa alla media di lega (stile Poisson classico).
3. **Il dataset conteneva un bug serio**: `stagioni/2010.txt` era una copia esatta di `stagioni/2009.txt` — la stagione 2009/10 era contata due volte e la 2010/11 mancava del tutto. Corretto sostituendo il file con i dati reali della stagione mancante.
4. Con `peso_quote` alto, storico+forma+scontri non venivano rinormalizzati a sommare 1 tra loro: l'xG stimato collassava verso 0 (es. 0.24 gol attesi invece di ~1.6) perché le "quote" non entrano nel calcolo dell'xG ma nel blend finale delle probabilità — bug corretto.
5. Con tutti i bug corretti e il modello validato su tre stagioni indipendenti: **né lo storico né gli scontri diretti aggiungono valore misurabile sopra le sole quote di mercato**; solo un peso piccolo (10%) alla forma recentissima aiuta in modo consistente. Il vantaggio del modello completo sopra il solo mercato è reale ma modesto: **circa 1 punto percentuale di accuratezza media**, misurato onestamente su dati mai visti dalla fase di ricerca dei pesi.

---

## 🛠️ Tecnologie

- **Python 3.10+** (sviluppato e testato con Python 3.14)
- **Streamlit** - Dashboard interattiva
- **Pandas** - Manipolazione dati
- **NumPy** - Calcoli numerici
- **SciPy** - Distribuzione di Poisson per il modello Dixon-Coles
- **Plotly** - Grafici interattivi
- **Scikit-learn** - Metriche di validazione
- **Pytest** - Test automatici sul modello (vedi `tests/`)

---

## 📁 Struttura del Progetto

```
serie-a-dashboard/
├── app.py                  # Dashboard principale
├── modello.py              # Modulo condiviso: Dixon-Coles, decadimento temporale, RPS
├── unisci_dati.py          # Script per unire i CSV stagionali (Date, Stagione, quota di consenso)
├── test_dati.py            # Script di ispezione rapida di un file stagione
├── serie_a.csv             # Dataset completo (11.534 partite)
├── requirements.txt        # Dipendenze Python pinnate
├── pages/
│   └── backtesting.py      # Pagina di validazione (walk-forward multi-stagione)
├── tests/
│   ├── conftest.py
│   └── test_model.py       # Test automatici su forma, scontri diretti, pesi
├── stagioni/               # File .txt delle singole stagioni
│   ├── 1993.txt
│   ├── 1994.txt
│   └── ...
├── venv/                   # Ambiente virtuale Python
├── readme.md               # Questo file
└── ROADMAP.md              # Piano dettagliato Fase 2/3: fonti dati, letteratura, tecniche
```

---

## 🚀 Installazione

### Prerequisiti
- Python 3.10 o superiore
- pip (package manager Python)

### Setup

```bash
# 1. Clona o crea la cartella del progetto
mkdir serie-a-dashboard
cd serie-a-dashboard

# 2. Crea ambiente virtuale
python -m venv venv

# 3. Attiva ambiente virtuale
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Installa dipendenze
pip install -r requirements.txt

# 5. Scarica i dati
# Vai su https://www.football-data.co.uk/italym.php
# Scarica tutte le stagioni dal 1993 a oggi come .txt
# Mettili nella cartella "stagioni/"

# 6. Unisci i dati
python unisci_dati.py

# 7. (opzionale) Esegui i test automatici
pytest tests/

# 8. Avvia la dashboard
streamlit run app.py
```

---

## 🎮 Utilizzo

### Dashboard Principale
1. Seleziona due squadre dai menu a tendina
2. Regola i pesi nella sidebar (Forma, Scontri, Quote)
3. Clicca **CALCOLA**
4. Analizza probabilità, gol attesi, Over/Under e scontri diretti

### Backtesting
1. Clicca su **Backtesting** nella sidebar di Streamlit
2. Scegli quante stagioni recenti usare come test (walk-forward: quelle precedenti sono training), imposta i pesi del modello e gli iperparametri Dixon-Coles (emivita, rho)
3. Clicca **🚀 Esegui Backtesting** per validare i pesi scelti, oppure **🔬 Confronta configurazioni** per vedere fianco a fianco solo storico / storico+forma / vecchi default / solo quote / ottimale
4. Visualizza accuratezza, RPS, log-loss, matrice di confusione, metriche per classe e (se selezioni più stagioni) l'accuratezza stagione per stagione

---

## 📈 Prossimi Sviluppi

Con i pesi validati (forma leggera + quote) il modello batte le sole quote di ~1 punto percentuale medio, in modo stabile su tre stagioni indipendenti. È un margine reale ma piccolo: prima di investire in feature più complesse ha senso capire perché storico e scontri diretti, anche dopo averli sistemati (decadimento temporale, forza relativa alla media di lega), non aggiungono valore — probabile indizio che il mercato incorpora già quell'informazione meglio di come possiamo farlo con le sole medie sui gol.

Il piano dettagliato (fonti dati specifiche, letteratura accademica, tecniche di modellazione, errori comuni da evitare) è in **[ROADMAP.md](ROADMAP.md)**. In sintesi, i prossimi passi:

- [ ] **Sistema Elo o pi-ratings dinamico** al posto delle medie storiche/forma attuali
- [ ] **Expected Goals (xG) reali** al posto dei gol effettivi (fonti: Understat, FBref)
- [ ] **Modello XGBoost** con più feature (Elo, xG, riposo tra partite)
- [ ] **Value betting**, con cautela: il margine sopra il mercato è reale ma piccolo (~1%)
- [ ] **Multi-campionato**, **calendario automatico**, **NLP infortuni/calciomercato**

---

## ⚠️ Disclaimer

Questo software è a scopo **dimostrativo ed educativo**. Non costituisce invito al gioco d'azzardo. Le previsioni sono basate su modelli statistici e non garantiscono risultati reali. Il gioco d'azzardo può causare dipendenza.

La dashboard principale permette di scegliere **qualsiasi coppia di squadre** presenti nello storico, usando tutti i dati disponibili (anche successivi, se scegli un accoppiamento "datato"): è pensata come simulazione "chi vincerebbe oggi tra queste due squadre", non come previsione retrospettiva di una partita realmente giocata in una data specifica. La pagina di **Backtesting**, invece, rispetta rigorosamente l'ordine cronologico (nessun dato futuro rispetto alla partita da prevedere) ed è quella da usare per valutare l'accuratezza reale del modello.

---

## 📊 Dataset

- **Fonte**: [Football-Data.co.uk](https://www.football-data.co.uk/italym.php)
- **Periodo**: 1993 - 2026 (stagione 2025/26 in corso)
- **Partite**: 11.534, tutte le 33 stagioni presenti senza duplicati né buchi (verificato: nessuna partita duplicata, nessuna stagione mancante tra la prima e l'ultima — controllo automatico in `tests/test_model.py`)
- **Squadre**: 53
- **Colonne**: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Stagione, OddsAvgH/D/A (quota di consenso multi-bookmaker), B365H/D/A, PSH/D/A, oltre a tiri/corner/cartellini quando disponibili (non ancora usati dal modello)

---

## 👨‍💻 Sviluppato con
- Python
- Streamlit