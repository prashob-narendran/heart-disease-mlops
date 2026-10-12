"""Shared pytest setup.

Force matplotlib's file-only backend before any test imports pyplot, so the
tests never try to open a GUI window (fails on Windows when Tcl/Tk is
missing, and on CI runners that have no display).
"""

import matplotlib

matplotlib.use("Agg")
