"""Offline VFR cross-country planning engine for the Cessna 172S.

Pure library code: no I/O beyond reading its own bundled data files, no UI
imports, no network. Everything here must run unmodified under Pyodide so the
same engine backs both the desktop dev server and the offline iPad PWA.
"""
