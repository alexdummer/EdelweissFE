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
#  Daniel Reitmair daniel.reitmair@uibk.ac.at
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

from edelweissfe.config import registry
from edelweissfe.utils.misc import strCaseCmp


def getMaterialClass(materialName: str, provider: str = None) -> type:
    """Return the class implementing material ``materialName`` for the given ``provider``.

    ``provider`` selects a namespace, not a variant of one lookup, and is dispatched via an
    explicit table rather than the registry. The ``marmotmaterial`` provider ignores
    ``materialName`` and returns ``None``: a Marmot material has no Python class, since it is
    instantiated inside the C++/Cython element wrapper from its name and property array. ``None``
    signals the caller to keep the material as a ``{"name": ..., "properties": ...}`` record
    instead of constructing an object (see ``AbqModelConstructor.createMaterialsFromInputFile``).

    The ``edelweiss`` provider is resolved through the registry (``material`` category).

    The ``marmotmaterialpoint`` provider is different from every other provider here: it is not
    meant to be used through ``*material`` in an input file at all. ``materialName`` for it is a
    base-class token (``hypoelastic``, ``gradientenhancedhypoelastic``), not the name of a
    concrete Marmot material, and the returned class takes ``(materialName, materialProperties)``
    -- the *actual* Marmot material name plus its properties -- not the single
    ``materialProperties`` argument ``AbqModelConstructor.createMaterialsFromInputFile`` passes to
    every other provider's class. It exists purely so external code (e.g. a finite difference
    stencil in a downstream package) can look up the right point-wise material class generically
    and construct it itself with the material name and properties *it* has on hand; using it via
    ``*material, provider=marmotmaterialpoint`` raises or fails with a missing constructor
    argument instead.

    Parameters
    ----------
    materialName
        The name of the requested material.
    provider
        The name of the material provider.

    Returns
    -------
    type
        The material provider class type, or ``None`` for the ``marmotmaterial`` provider.

    Raises
    ------
    edelweissfe.config.registry.RegistryLookupError
        If ``provider`` is ``edelweiss`` and no material is registered under ``materialName``.
    """

    if provider is None:
        provider = "MarmotMaterial"

    if strCaseCmp(provider, "marmotmaterial"):
        # The material is created and owned by the Marmot element itself.

        return None

    if strCaseCmp(provider, "marmotmaterialpoint"):
        # A Marmot material evaluated point-wise, without an element in between.
        # Which class applies follows from the base class the material is registered for
        # in Marmot, not from its name, so the caller states the base class.
        if strCaseCmp(materialName, "hypoelastic"):
            from edelweissfe.materials.marmot.marmothypoelastic import (
                MarmotHypoElasticMaterial,
            )

            return MarmotHypoElasticMaterial

        if strCaseCmp(materialName, "gradientenhancedhypoelastic"):
            from edelweissfe.materials.marmot.marmotgradientenhancedhypoelastic import (
                MarmotGradientEnhancedHypoElasticMaterial,
            )

            return MarmotGradientEnhancedHypoElasticMaterial

        raise Exception(
            "Unknown Marmot material point base class '{:}'; expected 'hypoelastic' or "
            "'gradientenhancedhypoelastic'".format(materialName)
        )

    if strCaseCmp(provider, "edelweiss"):

        materialClass, _ = registry.lookup("material", materialName)

        return materialClass
