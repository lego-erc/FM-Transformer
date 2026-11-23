"""Flow-matching generator of events: one incoming particle in a cuboid of material, its outgoing particles and E_dep.

Two trained models generate an event. The multiplicity model (``multiplicity.model.MultModel``) samples the number of
outgoing particles of each species, the flow (``main.modules.LEGOLtng``) generates their tokens and the E_dep token,
and ``main.generate.GenerateOut`` runs the two in sequence.
"""
