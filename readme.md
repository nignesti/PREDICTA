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
- **Metriche**: Accuratezza 1X2, F1 Score, Matrice di Confusione, breakdown per stagione
- **Confronto tra modelli**: solo storico, storico+forma, completo, solo quote
- **Ottimizzazione dei pesi** per massimizzare l'accuratezza

---

## 🧠 Il Modello

Il sistema combina **quattro componenti** con pesi regolabili:

| Componente | Descrizione | Peso Default |
|------------|-------------|--------------|
| **Media storica** | Gol fatti/subiti medi dal 1993 a oggi | 20% |
| **Forma recente** | Performance nelle ultime 5 partite | 50% |
| **Scontri diretti** | Precedenti tra le due squadre | 15% |
| **Quote bookmaker** | Saggezza del mercato (Bet365/Pinnacle) | 15% |

### Algoritmo
1. **Calcolo xG**: forza attacco/difesa di ogni squadra relativa alla media di campionato (stile Poisson classico), combinata pesando storico, forma e scontri diretti
2. **Vantaggio casa**: incorporato direttamente nelle formule (medie gol casa/trasferta separate), non come correzione aggiuntiva
3. **Simulazione Monte Carlo**: 10.000 partite simulate con distribuzione di Poisson
4. **Blend quote**: le probabilità finali fondono il modello statistico con le probabilità implicite delle quote

---

## 📊 Risultati Backtesting

| Test | Configurazione | Accuratezza misurata |
|------|----------------|-------------------|
| Baseline | Solo storico | 43.0% |
| +Forma | Storico + Forma recente | 46.2% |
| Completo | Tutte le componenti (pesi default) | 50.3% |
| Solo Quote | Solo mercato bookmaker | 52.9% |

*Accuratezza misurata sulla pagina di Backtesting (walk-forward), stagione 2025 come test set (342 partite valutabili su 380), stagioni precedenti come training. Benchmark "predici sempre 1": 38.9% su questo test set.*

Nota: fino a poco fa il modello collassava di fatto su "vince sempre la casa" a prescindere dalle squadre (accuratezza indistinguibile dal benchmark), per due bug ora corretti: il calcolo della "forma recente" usava quasi solo le partite in trasferta, e le formule storico/forma pesavano ogni statistica come media semplice con la media di campionato, azzerando le differenze reali tra squadre invece di usare una forza attacco/difesa relativa alla media di lega. Usa il bottone **🔬 Confronta 4 configurazioni** nella pagina di Backtesting per riprodurre questi numeri con i tuoi dati.

---

## 🛠️ Tecnologie

- **Python 3.10+** (sviluppato e testato con Python 3.14)
- **Streamlit** - Dashboard interattiva
- **Pandas** - Manipolazione dati
- **NumPy** - Calcoli numerici e simulazioni
- **Plotly** - Grafici interattivi
- **Scikit-learn** - Metriche di validazione
- **Pytest** - Test automatici sul modello (vedi `tests/`)

---

## 📁 Struttura del Progetto

```
serie-a-dashboard/
├── app.py                  # Dashboard principale
├── unisci_dati.py          # Script per unire i CSV stagionali (aggiunge anche la colonna Stagione)
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
└── readme.md               # Questo file
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
2. Scegli quante stagioni recenti usare come test (walk-forward: quelle precedenti sono training) e imposta i pesi del modello
3. Clicca **🚀 Esegui Backtesting** per validare i pesi scelti, oppure **🔬 Confronta 4 configurazioni** per vedere fianco a fianco solo storico / storico+forma / completo / solo quote
4. Visualizza accuratezza, matrice di confusione, metriche per classe e (se selezioni più stagioni) l'accuratezza stagione per stagione

---

## 📈 Prossimi Sviluppi

Ordinati per dipendenza: ha poco senso investire in modelli più complessi (XGBoost) o in value betting finché non è chiaro se la componente statistica batte davvero il mercato — oggi da backtest il modello puro non ci riesce (50.3% vs 52.9% delle sole quote).

- [ ] **Sistema Elo dinamico** per il ranking squadre, al posto delle medie storiche/forma attuali (probabilmente il miglioramento con il miglior rapporto sforzo/beneficio)
- [ ] **Verificare se il modello batte il mercato** in modo sistematico, prima di costruirci sopra altre feature
- [ ] **Modello XGBoost** con più feature, una volta risolto il punto sopra
- [ ] **Calendario prossima giornata** con pronostici automatici
- [ ] **NLP News** per integrare infortuni e calciomercato
- [ ] **Value betting** per identificare quote vantaggiose (ha senso solo se il modello batte il mercato)
- [ ] **Multi-campionato** (Premier League, Liga, Bundesliga)

---

## ⚠️ Disclaimer

Questo software è a scopo **dimostrativo ed educativo**. Non costituisce invito al gioco d'azzardo. Le previsioni sono basate su modelli statistici e non garantiscono risultati reali. Il gioco d'azzardo può causare dipendenza.

La dashboard principale permette di scegliere **qualsiasi coppia di squadre** presenti nello storico, usando tutti i dati disponibili (anche successivi, se scegli un accoppiamento "datato"): è pensata come simulazione "chi vincerebbe oggi tra queste due squadre", non come previsione retrospettiva di una partita realmente giocata in una data specifica. La pagina di **Backtesting**, invece, rispetta rigorosamente l'ordine cronologico (nessun dato futuro rispetto alla partita da prevedere) ed è quella da usare per valutare l'accuratezza reale del modello.

---

## 📊 Dataset

- **Fonte**: [Football-Data.co.uk](https://www.football-data.co.uk/italym.php)
- **Periodo**: 1993 - 2025
- **Partite**: 11.534
- **Squadre**: 53
- **Colonne**: HomeTeam, AwayTeam, FTHG, FTAG, FTR, Stagione, B365H, B365D, B365A, PSH, PSD, PSA

---

## 👨‍💻 Sviluppato con
- Python
- Streamlit