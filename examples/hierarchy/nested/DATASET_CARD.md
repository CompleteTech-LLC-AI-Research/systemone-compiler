# Nested reuse dataset card

Each synthetic row has two note ports and a Boolean switch for the second
subgraph instance. The train split includes one missing optional second note,
which exercises the typed default. Other splits omit that case because the
fixed default creates the same static leaf input and the split guard correctly
rejects duplicate projections. All other note strings and groups are disjoint.

The unclear note and disabled second call demonstrate review-required output;
the one-call budget case is a separate synthetic trace, not a quality row.
Gold Noul labels are illustrative and independent of the lexical mock. No note
comes from a person or operational system, and no review action is performed.
