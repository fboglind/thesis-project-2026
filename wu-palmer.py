from thesis_project.lexical import SaldoGraph
g = SaldoGraph.from_pickle('data/lexical/saldo.pkl')

# The kamel example from BNT results
target = g.lookup('kamel')[0]  # kamel..1 presumably
candidates = ['djur', 'åsna', 'häst', 'lama', 'puckelkamel',
              'däggdjur', 'ökendjur', 'ökenlastdjur']
print(f'Wu-Palmer similarity to kamel ({target}, depth={g.depth(target)}):')
for word in candidates:
    senses = g.lookup(word)
    if not senses:
        print(f'  {word:20s} OOV')
        continue
    best_sim = None
    best_sense = None
    for s in senses:
        sim = g.wu_palmer(target, s)
        if sim is not None and (best_sim is None or sim > best_sim):
            best_sim = sim
            best_sense = s
    if best_sim is None:
        print(f'  {word:20s} {senses[0]} (no path)')
    else:
        print(f'  {word:20s} {best_sense:20s} depth={g.depth(best_sense):2d}  sim={best_sim:.3f}')
