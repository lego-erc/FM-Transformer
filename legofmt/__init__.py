"""Particle-physics flow-matching shower generator.

``main.LEGOLtng`` flows the base distribution to particles, generating the
species of each slot alongside its kinematics, so the multiplicity is the count
of slots that did not resolve to the empty class. ``main.generate.GenerateOut``
drives it. ``multiplicity.MultModel`` is retained for the inverse direction
(``GenerateIn``'s incoming-PID head).
"""
