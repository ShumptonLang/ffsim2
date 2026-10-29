"""Closed-loop season simulation: managers act every simulated week.

The static engine (``ffsim.sim``) scores fixed rosters. Here each simulated
season is a path in which every manager, using only what is observable at the
time, claims and drops players week by week under the league's waiver rules, so
waiting, streaming and a shrinking free-agent pool fall out of the simulation.
"""
