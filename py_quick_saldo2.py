from thesis_project.lexical import SaldoGraph
g = SaldoGraph.from_pickle('data/lexical/saldo.pkl')
s = 'kronärtskocka..1'
while s is not None:
    print(f'  depth={g.depth(s):2d}  {s:20s}  ({g.written_form(s)})')
    s = g.primary_descriptor(s)