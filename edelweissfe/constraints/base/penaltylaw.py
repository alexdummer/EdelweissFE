#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""
The normal penalty force laws shared by the penalty contact constraints.

Sign convention: the gap :math:`g` is negative when the surfaces penetrate, and the normal force
:math:`f_n` carries the sign of :math:`g`, i.e. :math:`f_n \\le 0` in contact. The stiffness
:math:`\\mathrm{d}f_n/\\mathrm{d}g` is then positive for :math:`g < 0`.

``penaltyTimesArea`` is the penalty modulus times the area the contact point represents: a nodal
tributary area in node-based formulations, a quadrature weight in integrated ones. For the ``tabular``
law, which carries its own pressures, it is the area alone.
"""

import numpy as np

#: The supported penalty force laws.
contactTypes = ("linear", "quadratic")


def validatedContactType(contactType: str, allowTabular: bool = False) -> str:
    """Normalize a contact type option to lower case, and reject unsupported laws.

    Parameters
    ----------
    contactType
        The contact type as given in the input file.
    allowTabular
        Whether the calling constraint supports the ``tabular`` law (:class:`TabularPressureOverclosure`).

    Returns
    -------
    str
        The lower-case contact type.
    """
    contactType = contactType.lower()
    supported = contactTypes + (("tabular",) if allowTabular else ())
    if contactType not in supported:
        raise ValueError(f"Constraint type '{contactType}' is not supported. Use one of {', '.join(supported)}.")
    return contactType


class TabularPressureOverclosure:
    """A tabular pressure-overclosure law, as Abaqus' ``*Surface Behavior, pressure-overclosure=TABULAR``.

    The contact pressure :math:`p` is piecewise linear in the overclosure :math:`h = -g` between the table
    points, zero for :math:`h \\le 0` and extrapolated with the slope of the last segment beyond the
    table. The normal force of a contact point of area :math:`A` is :math:`f_n = -A \\, p(-g)`, with the
    stiffness :math:`\\mathrm{d}f_n/\\mathrm{d}g = A \\, p'(-g)`, which is piecewise constant.

    Parameters
    ----------
    pressures
        The pressures of the table points, starting with 0 and non-decreasing.
    overclosures
        The overclosures of the table points, starting with 0 and strictly increasing. (Abaqus also
        admits a clearance at zero pressure, i.e. a negative first overclosure; that is not supported.)
    """

    def __init__(self, pressures: np.ndarray, overclosures: np.ndarray):
        pressures = np.asarray(pressures, dtype=float)
        overclosures = np.asarray(overclosures, dtype=float)
        if pressures.ndim != 1 or pressures.shape != overclosures.shape or len(pressures) < 2:
            raise ValueError("A pressure-overclosure table needs at least two (pressure, overclosure) points.")
        if pressures[0] != 0.0 or overclosures[0] != 0.0:
            raise ValueError("A pressure-overclosure table must start at (0, 0).")
        if np.any(np.diff(overclosures) <= 0.0) or np.any(np.diff(pressures) < 0.0):
            raise ValueError(
                "The overclosures of a pressure-overclosure table must increase strictly, the pressures must not "
                "decrease."
            )
        if pressures[-1] <= 0.0:
            raise ValueError("A pressure-overclosure table must reach a positive pressure.")
        self.pressures = pressures
        self.overclosures = overclosures
        self.slopes = np.diff(pressures) / np.diff(overclosures)

    @classmethod
    def fromFile(cls, fileName: str) -> "TabularPressureOverclosure":
        """Read a table from a text file with one ``pressure, overclosure`` pair per line (Abaqus' order);
        commas or whitespace separate the columns, and lines starting with ``#`` or ``**`` are comments."""
        rows = []
        with open(fileName) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("**"):
                    continue
                rows.append([float(v) for v in line.replace(",", " ").split()[:2]])
        table = np.array(rows)
        return cls(table[:, 0], table[:, 1])

    @property
    def initialSlope(self) -> float:
        """The slope of the first segment."""
        return float(self.slopes[0])

    @property
    def firstSegmentLength(self) -> float:
        """The overclosure of the second table point, up to which the law is linear."""
        return float(self.overclosures[1])

    def pressure(self, overclosure: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The pressure and its derivative with respect to the overclosure, element-wise.

        Parameters
        ----------
        overclosure
            The overclosure(s) :math:`h = -g`.

        Returns
        -------
        tuple
            :math:`p(h)` and :math:`p'(h)`.
        """
        h = np.asarray(overclosure, dtype=float)
        segment = np.clip(np.searchsorted(self.overclosures, h, side="right") - 1, 0, len(self.slopes) - 1)
        slope = self.slopes[segment]
        p = self.pressures[segment] + slope * (h - self.overclosures[segment])
        closed = h > 0.0
        return np.where(closed, p, 0.0), np.where(closed, slope, 0.0)


def normalPenaltyForce(
    contactType: str,
    penaltyTimesArea: float | np.ndarray,
    gap: float | np.ndarray,
    table: TabularPressureOverclosure | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """The normal penalty force and its derivative with respect to the gap, at a closed contact point.

    Works element-wise on arrays of points as well as on single points.

    - ``linear``: :math:`f_n = k A \\, g`, with constant stiffness :math:`k A`.
    - ``quadratic``: :math:`f_n = -\\tfrac{1}{2} k A \\, g^2`, with stiffness :math:`-k A \\, g`.
    - ``tabular``: :math:`f_n = -A \\, p(-g)`, see :class:`TabularPressureOverclosure`.

    Parameters
    ----------
    contactType
        'linear', 'quadratic' or 'tabular', see :func:`validatedContactType`.
    penaltyTimesArea
        The penalty modulus times the area of the contact point(s); the area alone for 'tabular'.
    gap
        The gap(s), negative in contact.
    table
        The pressure-overclosure table, for 'tabular' only.

    Returns
    -------
    tuple
        The normal force(s) :math:`f_n` and the stiffness :math:`\\mathrm{d}f_n/\\mathrm{d}g`.
    """
    if contactType == "linear":
        normalForce = penaltyTimesArea * gap
        dNormalForce_dGap = penaltyTimesArea
    elif contactType == "tabular":
        p, dp = table.pressure(-np.asarray(gap, dtype=float))
        normalForce, dNormalForce_dGap = -penaltyTimesArea * p, penaltyTimesArea * dp
        if np.ndim(normalForce) == 0:
            return float(normalForce), float(dNormalForce_dGap)
    else:
        normalForce = -0.5 * penaltyTimesArea * gap**2
        dNormalForce_dGap = -penaltyTimesArea * gap
    return normalForce, dNormalForce_dGap


def closedNormalPenaltyForce(
    contactType: str,
    penaltyTimesArea: float | np.ndarray,
    gap: float | np.ndarray,
    closureTolerance: float,
    table: TabularPressureOverclosure | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """The normal penalty force and stiffness of a contact point that is closed, i.e. whose gap is below
    ``closureTolerance``, with the closure band smoothed so that force and stiffness are continuous.

    With ``closureTolerance = 0`` this is :func:`normalPenaltyForce`. Otherwise the contact starts at
    the gap ``closureTolerance`` instead of at 0, so that surfaces which start out touching are closed
    and stiff from the first iteration:

    - ``linear``: :math:`f_n = k A \\, (g - \\tfrac{1}{2} g_c)` for :math:`g \\le 0`, and
      :math:`f_n = -k A \\, (g_c - g)^2 / (2 g_c)` for :math:`0 < g < g_c` -- a quadratic transition
      whose force and stiffness vanish at :math:`g_c` and match the linear law at :math:`g = 0`, so
      that points at the border of the contact zone do not switch the tangent abruptly. The pressure
      of touching surfaces (gap 0) is :math:`\\tfrac{1}{2} k g_c`.
    - ``tabular``: as ``linear``, with the initial slope of the table as :math:`k` in the band and
      :math:`f_n = -A \\, p(-g + \\tfrac{1}{2} g_c)` for :math:`g \\le 0`. Continuous as long as
      :math:`\\tfrac{1}{2} g_c` lies within the first table segment, which the constraint checks.
    - ``quadratic``: the law shifted by :math:`g_c`, which is continuous at :math:`g_c` already.

    Parameters
    ----------
    contactType
        'linear' or 'quadratic', see :func:`validatedContactType`.
    penaltyTimesArea
        The penalty modulus times the area of the contact point(s).
    gap
        The gap(s), below ``closureTolerance``.
    closureTolerance
        The gap :math:`g_c \\ge 0` at which the contact closes.
    table
        The pressure-overclosure table, for 'tabular' only.

    Returns
    -------
    tuple
        The normal force(s) :math:`f_n` and the stiffness :math:`\\mathrm{d}f_n/\\mathrm{d}g`.
    """
    if closureTolerance == 0.0:
        return normalPenaltyForce(contactType, penaltyTimesArea, gap, table)
    if contactType == "quadratic":
        return normalPenaltyForce(contactType, penaltyTimesArea, gap - closureTolerance)
    inBand = gap > 0.0
    remaining = closureTolerance - gap
    if contactType == "tabular":
        bandStiffness = penaltyTimesArea * table.initialSlope
        closedForce, closedStiffness = normalPenaltyForce(
            contactType, penaltyTimesArea, np.minimum(gap, 0.0) - 0.5 * closureTolerance, table
        )
    else:
        bandStiffness = penaltyTimesArea
        closedForce = penaltyTimesArea * (gap - 0.5 * closureTolerance)
        closedStiffness = penaltyTimesArea
    normalForce = np.where(inBand, -0.5 * bandStiffness * remaining**2 / closureTolerance, closedForce)
    dNormalForce_dGap = np.where(inBand, bandStiffness * remaining / closureTolerance, closedStiffness)
    if np.ndim(normalForce) == 0:
        return float(normalForce), float(dNormalForce_dGap)
    return normalForce, dNormalForce_dGap
