import pandas as pd

df = pd.read_csv("data/patients_dakar.csv")

print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

print("\n--- 5 premiers patients ---")
print(df.head())

print("\n--- Répartition des diagnostics ---")
for diag, count in df["diagnostic"].value_counts().items():
    pct = count / len(df) * 100
    print(f"  {diag:12s} : {count:3d} patients ({pct:.1f}%)")

print("\n--- Température moyenne par diagnostic ---")
for diag, temp in df.groupby("diagnostic")["temperature"].mean().items():
    print(f"  {diag:12s} : {temp:.1f}C")

print("=" * 50)
print("Exploration terminée !")