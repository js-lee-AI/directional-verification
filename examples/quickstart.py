"""Directional label selection on one stored query. CPU only, no downloads."""

import ddv

parent = "Susana Dosamantes"
candidates = ["Paulina Rubio", "Diego Luna", "Odiseo Bichir", "Selena Gomez"]

# Four-teacher means of the stored log probabilities for this query, rounded.
known = [[-2.46, -2.93, -2.87, -4.23]]        # parent after "{child}'s parent is", per token
reverse = [[-15.61, -13.60, -23.71, -14.30]]  # child after "{parent}'s child is", summed
context = [[-24.78, -21.12, -35.68, -18.51]]  # child after "The child is", summed

print("known direction:", ddv.select_label(known, candidates, parent))
print("reverse:        ", ddv.select_label(reverse, candidates, parent))
print("reverse with DC:", ddv.select_label(ddv.domain_context(reverse, context), candidates, parent))
