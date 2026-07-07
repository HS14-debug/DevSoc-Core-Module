import pandas as pd

MODELS = ["qwen3-0.6b", "ministral-8b", "qwen3-30b-a3b", "qwen3-30b-a3b-instruct"]

df = pd.read_parquet("routing_table.parquet")

print("=== Missing values per model score/cost ===")
for m in MODELS:
    print(m, "missing scores:", df[f"{m}_score"].isna().sum(),
          "| missing cost:", df[f"{m}_cost"].isna().sum())

print("\n=== Average quality score per model ===")
print(df[[f"{m}_score" for m in MODELS]].mean())

print("\n=== Average cost (tokens) per model ===")
print(df[[f"{m}_cost" for m in MODELS]].mean())

print("\n=== Difficulty distribution ===")
print(df["difficulty"].value_counts().sort_index())

print("\n=== Correlation: difficulty vs each model's score ===")
for m in MODELS:
    print(m, df["difficulty"].corr(df[f"{m}_score"]))

print("\n=== How often does the CHEAPEST model already get a perfect/near-perfect score? ===")
cheap = "qwen3-0.6b"
print(f"{cheap} score >= 9:", (df[f"{cheap}_score"] >= 9).mean().round(3), "of all turns")

print("\n=== How much better is the BEST model vs the CHEAPEST, on average? ===")
best_score = df[[f"{m}_score" for m in MODELS]].max(axis=1)
print("Avg best score:", best_score.mean().round(2))
print("Avg cheapest (qwen3-0.6b) score:", df[f"{cheap}_score"].mean().round(2))

print("\n=== Cost ratio: most expensive model vs cheapest ===")
print((df[[f"{m}_cost" for m in MODELS]].mean() / df[f"{cheap}_cost"].mean()).round(2))

print("\n=== Total tokens if you used the cheapest model for EVERY turn ===")
print(df[f"{cheap}_cost"].sum())
print("=== Total tokens if you used the most expensive model for EVERY turn ===")
most_exp = df[[f"{m}_cost" for m in MODELS]].mean().idxmax()
print(most_exp, "->", df[most_exp.replace('_cost','') + '_cost'].sum())
print(f"\nFor reference, your budget is 10,000 tokens per episode/stream — "
      f"so no policy can come close to answering all {len(df)} turns; "
      f"the episode design (how you sample a 'stream' of turns) matters a lot.")
