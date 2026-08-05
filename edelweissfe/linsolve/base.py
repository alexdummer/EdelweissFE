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

"""The common base class every registered ``linsolver`` inherits, so the nonlinear solver can treat
all of them uniformly instead of special-casing per capability.

Every ``linsolve/<name>/__init__.py``'s ``createSolver(opts)`` factory returns an instance of a
:class:`LinearSolver` subclass, callable as ``(A, b) -> x``. Two setters -- :meth:`LinearSolver.setJournal`
and :meth:`LinearSolver.setFieldStructure` -- are part of that base contract with safe no-op-ish
defaults, so a caller can call either on *any* registered solver unconditionally, whether or not that
particular solver actually uses the information. This replaces an earlier design where field-structure
awareness was an isolated, ``isinstance``-checked opt-in mixin (only ``blockamg`` had it) -- adding a
second, parallel mixin for Journal awareness would have meant every new cross-cutting capability grows
its own special case at every call site. One base, grown as needed, avoids that.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldBlock:
    """One physical field's contiguous block in the DOF vector.

    Attributes
    ----------
    name
        The field name, e.g. ``"displacement"`` or ``"nonlocal damage"``.
    start, stop
        The half-open DOF range ``[start, stop)`` of the field (fields are contiguous and field-major).
    dimension
        The nodal dimension of the field: the number of components per node (e.g. ``3`` for a 3D
        displacement, ``1`` for a scalar damage field). Determines the field's near null-space -- a
        vector field's rigid-body translations, a scalar field's constant.
    """

    name: str
    start: int
    stop: int
    dimension: int


class LinearSolver:
    """Common base for every ``linsolver`` registry entry. Callable as ``(A, b) -> x``.

    Subclasses implement :meth:`__call__`. :meth:`setJournal` and :meth:`setFieldStructure` have safe
    defaults here (store-and-ignore, and a plain no-op respectively) so the nonlinear solver can call
    both on any solver without asking first which ones care -- a solver that needs one overrides it;
    everything else inherits a default that does nothing harmful.
    """

    _journal = None
    _fieldStructure: "list[FieldBlock] | None" = None

    def setJournal(self, journal) -> None:
        """Receive the shared :class:`~edelweissfe.journal.journal.Journal` instance.

        Default just stores it on ``self._journal`` for solvers that want to log through it; solvers
        with no logging needs simply never read the attribute.
        """
        self._journal = journal

    def setFieldStructure(self, fields: "list[FieldBlock]") -> None:
        """Receive the ordered field blocks of the DOF vector (in DOF order).

        Default no-op -- only field-split solvers (e.g. ``blockamg``) need this; every other solver
        simply ignores the call.
        """

    requestedP1FieldNames: frozenset = frozenset()
    """Vector field names this solver actually wants a P1 topology map for (§22.1), queried by the
    nonlinear solver *before* computing anything, via this attribute -- unlike
    :meth:`setFieldStructure`, computing a P1 map is not always safe to run unconditionally: it hard-
    errors on an element topology it cannot classify (found the hard way -- an unrelated model using a
    rigid-body-contact discretization crashed once this was briefly made unconditional for every
    solver). Default empty -- most solvers never need this and the nonlinear solver skips the
    computation entirely when this is empty, rather than computing it and hoping it happens to
    succeed on a model that never asked for it.
    """

    def setP1Maps(self, p1Maps: dict) -> None:
        """Receive the P1 corner/midside topology map (§22.1,
        :func:`edelweissfe.numerics.p1topology.buildP1Map`) for every field named in
        :attr:`requestedP1FieldNames`, computed by the nonlinear solver alongside
        :meth:`setFieldStructure`.

        Default no-op -- only a solver that has populated :attr:`requestedP1FieldNames` (e.g.
        ``blockamg``'s ``p1FieldNames`` option) ever has this called with a non-empty ``p1Maps``;
        every other solver simply ignores the call.

        Parameters
        ----------
        p1Maps
            Every vector field's map, keyed by field name: ``{fieldName: (isCorner, edgeEndpoints)}``.
        """

    def __call__(self, A, b):
        raise NotImplementedError
