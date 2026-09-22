"""Particle-physics flow-matching shower generator.

Two checkpointed models compose at inference time: ``multiplicity.MultModel``
sizes the output, ``main.LEGOLtng`` flows the base distribution to particles,
and ``main.generate.GenerateOut`` glues them together.
"""
