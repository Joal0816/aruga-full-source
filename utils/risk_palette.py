"""Single source of truth for risk-level colors and icons across all surfaces.

Escalation reads left-to-right as monotonic severity (conventional red =
worst): green -> blue -> yellow -> orange -> deep orange -> red.

EMERGENCY used to be purple and CONCERNING red on every surface — a
glance-reader (or anyone trained on standard alarm palettes) reads red as
the top state. This module fixes the hierarchy once; consumers import
instead of re-declaring (there were 5 divergent copies).

RISK_HEX: #rrggbb for tkinter/HTML/CSS. bgr(): OpenCV drawing order.
ICONS: emoji for UI surfaces only — cv2.putText cannot render emoji.
"""
RISK_HEX = {
    "NORMAL": "#34d399",
    "RESTING": "#60a5fa",
    "UNUSUAL": "#fbbf24",
    "SLUMP": "#fb923c",
    "CONCERNING": "#ea580c",
    "EMERGENCY": "#dc2626",
}

ICONS = {
    "NORMAL": "🟢",
    "RESTING": "🔵",
    "UNUSUAL": "⚠️",
    "SLUMP": "🟠",
    "CONCERNING": "🛑",
    "EMERGENCY": "🚨",
}

# cv2.putText only draws ASCII — emoji on video frames render as garbage.
# Badges/labels drawn on frames use these plain-text prefixes instead.
ASCII_TAG = {
    "NORMAL": "",
    "RESTING": "",
    "UNUSUAL": "(!) ",
    "SLUMP": "(!) ",
    "CONCERNING": "(!!) ",
    "EMERGENCY": "(!!!) ",
}


def bgr(risk: str, default=(220, 220, 220)):
    """OpenCV BGR tuple for a risk level."""
    hx = RISK_HEX.get(risk)
    if not hx:
        return default
    r, g, bl = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
    return (bl, g, r)


def css(risk: str, default="#f8fafc"):
    return RISK_HEX.get(risk, default)
