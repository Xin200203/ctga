"""Experimental relaxed solver.

For v1 this reuses the beam-search solver so that the interface exists
without forcing a heavy differentiable optimization dependency.
"""

from __future__ import annotations

from ctga.graph2_match.solver_beam_qap import BeamQAPSolver


class QPRelaxSolver(BeamQAPSolver):
    pass
