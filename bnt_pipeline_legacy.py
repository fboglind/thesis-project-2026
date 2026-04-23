"""bnt_pipeline.py"""
"""
BNT Cosine Similarity Scoring Pipeline
=======================================
Computes graded semantic similarity scores for Boston Naming Test responses
using KB-BERT embeddings and cosine similarity.

Usage:
    python bnt_pipeline.py --data path/to/BNTsyntheticData_v2.xlsx

Requirements:
    pip install pandas openpyxl numpy scikit-learn torch transformers matplotlib
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────

# def load_bnt_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """Load BNT spreadsheet and separate items from user metadata.
    
#     Returns:
#         items_df: DataFrame with columns ['gold'] + user columns, one row per BNT item
#         user_meta: DataFrame with columns ['user', 'gender', 'age', 'diagnosis']
#     """
#     raw = pd.read_excel(filepath)
#     user_cols = [c for c in raw.columns if c.startswith('User')]
    
#     # Items are rows where Gold is a real word (not metadata)
#     meta_labels = ['Gender:', 'Age:', 'Kategori:']
#     items_df = raw[raw['Gold'].notna() & ~raw['Gold'].isin(meta_labels)].copy()
#     items_df = items_df[['Gold'] + user_cols].reset_index(drop=True)
#     items_df.rename(columns={'Gold': 'gold'}, inplace=True)
#     items_df['gold'] = items_df['gold'].str.strip().str.lower()
    
#     # Metadata from bottom rows
#     user_meta = pd.DataFrame({
#         'user': user_cols,
#         'gender': raw.iloc[32][user_cols].values,
#         'age': pd.to_numeric(raw.iloc[33][user_cols].values, errors='coerce'),
#         'diagnosis': raw.iloc[34][user_cols].values,
#     })
    
#     return items_df, user_meta


# ──────────────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────────────

# Responses that indicate failure to name (not semantically scorable)
NON_RESPONSES = {
    'hhhm jag vet inte', 'jag vet inte', 'vet inte', 'pass',
    'ingen aning', 'vet ej', 'hmm', 'hm', 'jag kan inte',
    'något sånt', 'nåt sånt',
}

def normalize_response(text: str) -> str | None:
    """Normalize a BNT response for embedding comparison.
    
    Steps:
        1. Lowercase and strip whitespace
        2. Remove trailing hedges ("kanske", "tror jag")
        3. Remove leading articles ("en", "ett")
        4. Remove leading filler phrases ("det är", "jag tror")
        5. Flag non-responses (returns None)
    """
    if pd.isna(text) or str(text).strip() == '':
        return None
    
    text = str(text).strip().lower()
    
    # Check non-responses first
    if text in NON_RESPONSES:
        return None
    
    # Remove trailing hedges
    text = re.sub(r'\s*(kanske|tror jag|eller nåt|är det|va)$', '', text).strip()
    
    # Remove leading filler phrases (order matters — longer first)
    text = re.sub(
        r'^(det är |det ser ut som |jag tror det är |jag tror |en slags |nån slags |typ |liksom |bild på |säkert )',
        '', text
    ).strip()
    
    # Remove leading articles
    text = re.sub(r'^(en|ett|den|det|de)\s+', '', text).strip()
    
    # Check again after cleaning
    if text in NON_RESPONSES or text == '' or len(text) < 2:
        return None
    
    return text


def preprocess_responses(items_df: pd.DataFrame, user_meta: pd.DataFrame) -> pd.DataFrame:
    """Build long-format response table with normalized responses.
    
    Returns DataFrame with columns:
        gold, user, diagnosis, age, gender, raw_response, normalized,
        is_exact_match, is_non_response
    """
    user_cols = user_meta['user'].tolist()
    meta_lookup = user_meta.set_index('user')
    
    records = []
    for _, row in items_df.iterrows():
        gold = row['gold']
        for user in user_cols:
            raw_resp = row[user]
            norm = normalize_response(raw_resp)
            
            # Check exact match (gold word appears in normalized response)
            is_exact = False
            if norm is not None:
                # Exact match: normalized response IS the gold word
                is_exact = (norm == gold)
                # Also accept: gold word is the last token (e.g., "dromedal kamel" → kamel)
                if not is_exact:
                    tokens = norm.split()
                    is_exact = (tokens[-1] == gold) if tokens else False
            
            records.append({
                'gold': gold,
                'user': user,
                'diagnosis': meta_lookup.loc[user, 'diagnosis'],
                'age': meta_lookup.loc[user, 'age'],
                'raw_response': raw_resp,
                'normalized': norm,
                'is_exact_match': is_exact,
                'is_non_response': norm is None,
            })
    
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────
# 3. EMBEDDING & SIMILARITY
# ──────────────────────────────────────────────────────

class KBBertEmbedder:
    """Compute embeddings using KB-BERT (Swedish BERT).
    
    Uses the [CLS] token representation from the last hidden layer.
    Caches embeddings for repeated lookups.
    """
    
    def __init__(self, model_name: str = 'KB/bert-base-swedish-cased'):
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.cache: dict[str, np.ndarray] = {}
        self._torch = torch
        print("Model loaded.")
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for a text string. Returns 1D numpy array."""
        if text in self.cache:
            return self.cache[text]
        
        inputs = self.tokenizer(
            text, return_tensors='pt',
            padding=True, truncation=True, max_length=64
        )
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        
        # [CLS] token embedding
        emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        self.cache[text] = emb
        return emb
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n, dim) numpy array."""
        # Split into cached and uncached
        uncached = [t for t in texts if t not in self.cache]
        
        if uncached:
            # Batch encode uncached texts
            inputs = self.tokenizer(
                uncached, return_tensors='pt',
                padding=True, truncation=True, max_length=64
            )
            with self._torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            for text, emb in zip(uncached, embeddings):
                self.cache[text] = emb
        
        return np.array([self.cache[t] for t in texts])


class MockEmbedder:
    """Mock embedder for testing pipeline logic without GPU/model.
    
    Assigns random but deterministic embeddings. 
    Same word always gets the same embedding.
    """
    
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.cache: dict[str, np.ndarray] = {}
    
    def embed(self, text: str) -> np.ndarray:
        if text not in self.cache:
            rng = np.random.RandomState(hash(text) % 2**31)
            self.cache[text] = rng.randn(self.dim).astype(np.float32)
        return self.cache[text]
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


def compute_similarity_scores(
    responses: pd.DataFrame,
    embedder,
) -> pd.DataFrame:
    """Compute cosine similarity between each response and its gold target.
    
    For exact matches: score = 1.0
    For non-responses: score = 0.0
    For other responses: cosine_similarity(embed(response), embed(gold))
    
    Adds columns: cosine_sim, graded_score
    """
    df = responses.copy()
    df['cosine_sim'] = np.nan
    
    # Pre-embed all gold words
    gold_words = df['gold'].unique().tolist()
    print(f"Embedding {len(gold_words)} gold words...")
    gold_embeddings = {w: embedder.embed(w) for w in gold_words}
    
    # Get unique normalized responses that need embedding
    needs_embedding = df[~df['is_exact_match'] & ~df['is_non_response']]['normalized']
    unique_responses = needs_embedding.dropna().unique().tolist()
    print(f"Embedding {len(unique_responses)} unique responses...")
    
    # Embed in batches
    resp_embeddings = {}
    batch_size = 32
    for i in range(0, len(unique_responses), batch_size):
        batch = unique_responses[i:i+batch_size]
        embs = embedder.embed_batch(batch)
        for text, emb in zip(batch, embs):
            resp_embeddings[text] = emb
    
    # Compute similarities
    scores = []
    for _, row in df.iterrows():
        if row['is_exact_match']:
            scores.append(1.0)
        elif row['is_non_response']:
            scores.append(0.0)
        elif row['normalized'] in resp_embeddings:
            gold_emb = gold_embeddings[row['gold']].reshape(1, -1)
            resp_emb = resp_embeddings[row['normalized']].reshape(1, -1)
            sim = cosine_similarity(gold_emb, resp_emb)[0, 0]
            # Clamp to [0, 1] — negative cosine similarities → 0
            scores.append(max(0.0, float(sim)))
        else:
            scores.append(0.0)
    
    df['cosine_sim'] = scores
    
    # Binary score for comparison
    df['binary_score'] = df['is_exact_match'].astype(int)
    
    return df


# ──────────────────────────────────────────────────────
# 4. ANALYSIS
# ──────────────────────────────────────────────────────

def analyze_results(scored: pd.DataFrame) -> None:
    """Print analysis comparing binary vs graded scoring across groups."""
    
    diag_order = ['HC', 'MCI', 'non-AD', 'AD']
    
    print("\n" + "="*70)
    print("RESULTS: Binary vs. Graded Scoring by Diagnostic Group")
    print("="*70)
    
    # Per-group summary
    summary = []
    for diag in diag_order:
        sub = scored[scored['diagnosis'] == diag]
        n_users = sub['user'].nunique()
        summary.append({
            'Diagnosis': diag,
            'N_users': n_users,
            'Binary_mean': sub['binary_score'].mean(),
            'Graded_mean': sub['cosine_sim'].mean(),
            'Graded_std': sub.groupby('user')['cosine_sim'].mean().std(),
            'Non_response_rate': sub['is_non_response'].mean(),
        })
    
    summary_df = pd.DataFrame(summary)
    print("\nPer-group means:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # The key insight: difference between binary and graded
    print("\n" + "-"*70)
    print("KEY: Graded - Binary (information gained by graded scoring)")
    print("-"*70)
    for _, row in summary_df.iterrows():
        diff = row['Graded_mean'] - row['Binary_mean']
        print(f"  {row['Diagnosis']:8s}: +{diff:.3f} "
              f"({'graded captures more' if diff > 0.05 else 'similar'})")
    
    # Per-item analysis: which items show most graded variation?
    print("\n" + "-"*70)
    print("Items with most variation in graded scores (interesting for analysis)")
    print("-"*70)
    item_stats = scored.groupby('gold').agg(
        binary_mean=('binary_score', 'mean'),
        graded_mean=('cosine_sim', 'mean'),
        graded_std=('cosine_sim', 'std'),
        non_response=('is_non_response', 'mean'),
    ).sort_values('graded_std', ascending=False)
    print(item_stats.head(10).to_string(float_format='%.3f'))
    
    # Example: show graded scores for 'kamel' by diagnosis
    print("\n" + "-"*70)
    print("Example: Graded scores for 'kamel' by diagnosis")
    print("-"*70)
    kamel = scored[scored['gold'] == 'kamel']
    for diag in diag_order:
        sub = kamel[kamel['diagnosis'] == diag]
        if len(sub) > 0:
            scores = sub['cosine_sim']
            print(f"  {diag:8s}: mean={scores.mean():.3f}, "
                  f"std={scores.std():.3f}, "
                  f"min={scores.min():.3f}, max={scores.max():.3f}")
    
    # Show some specific response-score pairs for kamel
    print("\n  Sample response scores:")
    kamel_unique = kamel[~kamel['is_exact_match'] & ~kamel['is_non_response']]
    kamel_unique = kamel_unique.drop_duplicates(subset='normalized')
    kamel_unique = kamel_unique.sort_values('cosine_sim', ascending=False)
    for _, row in kamel_unique.head(10).iterrows():
        print(f"    '{row['normalized']:25s}' → {row['cosine_sim']:.3f} ({row['diagnosis']})")


def save_results(scored: pd.DataFrame, output_path: str) -> None:
    """Save scored results to CSV."""
    scored.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


# ──────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BNT Cosine Similarity Scoring')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to BNTsyntheticData_v2.xlsx')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock embeddings (for testing pipeline logic)')
    parser.add_argument('--output', type=str, default='bnt_scored_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()
    
    # Load data
    print("Loading BNT data...")
    items_df, user_meta = load_bnt_data(args.data)
    print(f"  {len(items_df)} items, {len(user_meta)} users")
    print(f"  Diagnoses: {dict(user_meta['diagnosis'].value_counts())}")
    
    # Preprocess
    print("\nPreprocessing responses...")
    responses = preprocess_responses(items_df, user_meta)
    n_total = len(responses)
    n_non = responses['is_non_response'].sum()
    n_exact = responses['is_exact_match'].sum()
    n_score = n_total - n_non - n_exact
    print(f"  {n_total} total, {n_exact} exact matches, "
          f"{n_non} non-responses, {n_score} to score with embeddings")
    
    # Embed and score
    if args.mock:
        print("\n⚠ Using MOCK embeddings (random vectors). "
              "Similarity scores are NOT meaningful.")
        print("  Run without --mock to use KB-BERT.\n")
        embedder = MockEmbedder()
    else:
        embedder = KBBertEmbedder()
    
    print("\nComputing similarity scores...")
    scored = compute_similarity_scores(responses, embedder)
    
    # Analyze
    analyze_results(scored)
    
    # Save
    save_results(scored, args.output)


if __name__ == '__main__':
    main()