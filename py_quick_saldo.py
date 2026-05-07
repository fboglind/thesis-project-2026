from thesis_project.lexical import SaldoGraph
g = SaldoGraph.from_pickle('data/lexical/saldo.pkl')
for sense_id in g.stats().get('orphan_senses', []):
    print(sense_id, g.written_form(sense_id))
