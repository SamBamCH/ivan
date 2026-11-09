"""
The public's side of the slate, kept apart from ours, and our own candidate pool.

    public_projection   a replica of the public's projection (Stokastic) from the books' props and public inputs only
    public_field        the two-group field on the public projection (cached by its key) and the ownership read off it
    model               the ownership model fitted on real contests (features, fit, predict, the two-group simulator)
    generate            the generate step: the pool of candidate lineups the simulator picks from (worker processes)
    solver              one classic lineup by ILP under DraftKings' roster and salary rules
    optimal             the optimal lineup in each of the World's worlds and every player's share of them (the desk's Solve)
"""
