"""Gameplay Analyzer -- classical-CV subsystem (Phase 2).

Turns a raw Clash Royale recording into structured events
(``gameplay_analysis.json``) so script generation is grounded in what actually
happened in the match. This package is **deliberately independent** of
:mod:`backend`: the only coupling between the analyzer and the production
pipeline is the ``gameplay_analysis.json`` file handoff. Nothing here imports
from ``backend`` and nothing in ``backend`` imports from here.

Slice 2A (this) builds the foundation every later detector needs: a
preprocessed, cached **card template library** from the bundled card art. Later
slices (frame extraction, ROI location, hand/play/arena/timer/crown/elixir
detection, event building) attach to :class:`analyzer.workflow.AnalyzerWorkflow`.
"""
