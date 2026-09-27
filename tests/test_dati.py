import pandas as pd

# Prova a leggere il file
df = pd.read_csv("stagioni/1993.txt", encoding="latin1")

# Mostra le prime 3 righe
print("Colonne trovate:")
print(df.columns.tolist())
print("\nPrime 3 righe:")
print(df.head(3))
print(f"\nNumero di partite nel file: {len(df)}")