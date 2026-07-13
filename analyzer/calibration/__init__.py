"""Calibration framework (2C): where every UI element sits, per capture device.

Profiles are device-specific JSON files of fractional ROIs; :mod:`analyzer.
calibration.roi` crops/overlays them. Coordinates ship as placeholders and are
tuned against a real frame later -- the machinery (crop, overlay, load) is what
this slice delivers.
"""
