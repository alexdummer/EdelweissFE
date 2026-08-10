Linear solvers
==============

Linear solvers are defined in EdelweissFE after the ``*solver`` keyword using ``linsolver`` and an optional configuration file ``linsolverConfigFile`` as a data line.
The ``linsolverConfigFile`` needs to be in ``.json`` format.

Choose a linsolver after the ``*solver`` keyword:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=gmres
    linsolverConfigFile=opt.json

.. list-table:: Currently available linear solvers
    :width: 100%
    :widths: 15 1 25
    :header-rows: 1

    * - Name
      - Direct solver
      - Relevant module
    * - ``superlu``
      - ✓
      - ``scipy.sparse.linalg.spsolve``
    * - ``umfpack``
      - ✓
      - ``scipy.sparse.linalg.spsolve``
    * - ``pardiso``
      - ✓
      - ``edelweissfe.linsolve.pardiso.pardiso``
    * - ``panuapardiso``
      - ✓
      - ``edelweissfe.linsolve.panuapardiso.panuapardiso``
    * - ``klu``
      - ✓
      - ``edelweissfe.linsolve.klu.klu``
    * - ``petsclu``
      - ✓
      - ``edelweissfe.linsolve.petsclu.petsclu``
    * - ``mumps``
      - ✓
      - ``edelweissfe.linsolve.mumps.mumps``
    * - ``gmres``
      - ✗
      - ``edelweissfe.linsolve.gmres.gmres``
    * - ``amgcl``
      - ✗
      - ``edelweissfe.linsolve.amgcl.amgcl``
    * - ``inexactnewton``
      - ✗
      - ``edelweissfe.linsolve.inexactnewton.inexactnewton``
    * - ``blockamg``
      - ✗
      - ``edelweissfe.linsolve.blockamg.blockamg``
    * - ``matrixdump``
      - —
      - ``edelweissfe.linsolve.matrixdump.matrixdump``

Several linsolvers accept an optional configuration file ``linsolverConfigFile`` (a ``.json`` file), among them ``gmres``, ``amgcl``, ``inexactnewton`` and ``matrixdump``; the direct solvers ignore it (``pardiso`` and ``panuapardiso`` additionally read a single ``reuseSymbolicFactorization`` flag).

Choose the options for the linsolver (in this case ``gmres``) in an extra file:

.. code-block:: json

    	{
	"precondopts":
	{
	"presmoother": ["block_gauss_seidel", {"iterations": 15}],
	"postsmoother": ["block_gauss_seidel", {"iterations": 15}],
	},
	"linsolveopts": {"maxiter": 1, "restart": 1500}
	}


The ``inexactnewton`` solver
----------------------------

``inexactnewton`` is not a solver in its own right but a *modified-Newton–Krylov* scheme intended for large, coupled nonlinear models (for example penalty contact combined with adaptive mesh refinement and gradient-enhanced damage) where a direct factorization dominates the run time while the Jacobian changes only slightly from one Newton iteration to the next.

Instead of factorizing the system matrix on every Newton iteration, it keeps an **exact LU factorization of one iterate** (computed by a *delegate* direct solver, ``pardiso`` by default) and reuses it as a **preconditioner for GMRES** on the next few iterates. The linear tolerance follows an **Eisenstat–Walker forcing sequence** rather than being solved tightly: a Newton correction does not need the linear system solved to machine precision, so the reuse solves converge in only a handful of GMRES iterations. When the factorization goes stale it is refreshed automatically; the first solve of an increment and its large first correction — the iterates that precondition worst — are kept direct.

Because it exposes the ordinary ``(A, b) -> x`` interface, selecting it requires no other change to the analysis:

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=inexactnewton
    linsolverConfigFile=inexactnewton.json

All configuration keys are optional; the defaults are a turnkey configuration (the PARDISO delegate with the measured sweet-spot policy). The recognised keys:

.. list-table:: ``inexactnewton`` configuration keys
    :width: 100%
    :widths: 20 10 45
    :header-rows: 1

    * - Key
      - Default
      - Meaning
    * - ``delegate``
      - ``"pardiso"``
      - Factorizing backend supplying the lagged LU. ``"pardiso"`` in production, or ``"superlu"`` (SciPy, dependency-free) for testing or installs without the PARDISO extension.
    * - ``maxReuse``
      - ``8``
      - How many consecutive reuse solves one factorization may serve before it is refreshed.
    * - ``residualGrowthFactor``
      - ``4.0``
      - A solve whose ``||b||`` exceeds this multiple of the previous one is treated as a new increment (or a cutback) and refactorized.
    * - ``etaMin`` / ``etaMax``
      - ``1e-6`` / ``1e-3``
      - Clamp on the Eisenstat–Walker forcing tolerance (tightest / loosest a reuse solve may use).
    * - ``ewGamma`` / ``ewAlpha``
      - ``0.9`` / ``1.618…``
      - Eisenstat–Walker "choice 2" parameters, ``eta_k = ewGamma * (||b_k|| / ||b_{k-1}||) ** ewAlpha``.
    * - ``gmresRestart`` / ``gmresMaxOuter``
      - ``25`` / ``1``
      - GMRES Krylov dimension between restarts and maximum restart cycles; their product caps the iterations before a reuse falls back to a direct solve (default cap 25, just above the ~22-iteration break-even of the reference condensed system).
    * - ``staleIterationThreshold``
      - ``20``
      - A reuse converging in more iterations than this marks the region as hardening: refresh next iterate and grow the probe backoff. Set a little below the break-even.
    * - ``cheapIterationThreshold``
      - ``10``
      - A reuse converging within this many iterations marks the region as easy and resets the probe backoff.
    * - ``maxProbeBackoff``
      - ``8``
      - Ceiling on the direct-solve run inserted between reuse probes in a persistently hard region.
    * - ``verbose``
      - ``false``
      - Print one line per solve (refactorize?, forcing tolerance, iteration count).

Example configuration selecting the SuperLU delegate for a dependency-free run:

.. code-block:: json

    {
        "delegate": "superlu",
        "etaMax": 1e-3,
        "maxReuse": 8
    }


The ``blockamg`` solver
-----------------------

``blockamg`` is a field-split block-AMG solver for **large coupled multi-field systems** (e.g. displacement + gradient-enhanced damage). It is the O(n)-memory route to problem sizes a direct factorization cannot reach — past roughly a million DOFs its fill-in exceeds memory, whereas algebraic multigrid stays linear.

Applied *monolithically*, AMG is ineffective on such a coupled system (a single hierarchy cannot represent the disparate fields at once). ``blockamg`` instead builds **one AMG hierarchy per field** (AMGCL) and combines them with a **block Gauss–Seidel** sweep to precondition an outer GMRES, following Alkmim et al. (IJNME 2026). Per solve it equilibrates the system (symmetric diagonal scaling, to tame the large dynamic range), splits it into field blocks, and preconditions GMRES with the block sweep. Each field's near null-space is chosen from its **nodal dimension**: a vector field (dimension > 1, e.g. ``displacement``) gets its per-component rigid-body **translations**, a scalar field (e.g. ``nonlocal damage``) the default constant.

The block structure — which DOFs belong to which field, and each field's dimension — is **discovered automatically** from the ``DofManager`` and pushed into the solver by the nonlinear solver; nothing about the block layout is specified by hand. A ``linsolverConfigFile`` is therefore optional and carries only solver knobs. Requires the optional ``amgcl`` extension.

.. code-block:: edelweiss

    *solver, solver=NIST, name=theSolver
    linsolver=blockamg
    linsolverConfigFile=blockamg.json

.. code-block:: json

    {
        "outerTol": 1e-6,
        "sweeps": 1,
        "symmetric": true
    }

Recognised keys, all optional: ``outerTol`` / ``outerRestart`` / ``outerMaxiter`` (the outer GMRES), ``sweeps`` and ``symmetric`` (the block Gauss–Seidel), ``verbose``, and ``fieldPreconds`` (a mapping of field name, e.g. ``"displacement"``, to an AMGCL parameter tree overriding the dimension-based default for that field).

.. note::
   This is a *feasibility-grade* solver: on the reference model AMGCL's smoothed aggregation converges on the coupled system (~68 outer iterations with symmetric block Gauss–Seidel) but not tightly on the non-symmetric, condensed displacement block, so the count is O(100) rather than the O(30) a nonsymmetric-aware AMG (e.g. Trilinos/MueLu) would give. It is the right tool where the goal is to *fit in memory* at sizes a direct solver cannot, not to be fastest at moderate sizes.


The ``amgcl`` solver
--------------------

``amgcl`` is an iterative solver (Krylov method plus algebraic-multigrid or single-level preconditioner) built on the `AMGCL <https://github.com/ddemidov/amgcl>`_ library. Its ``linsolverConfigFile`` is forwarded as an AMGCL parameter tree; note that AMGCL silently ignores unknown parameter keys (warning only on stderr), so check its stderr if a configuration behaves unexpectedly.


The ``matrixdump`` diagnostic solver
------------------------------------

``matrixdump`` is not a solver but a diagnostic wrapper: it writes the equation systems it is handed to disk and then delegates the actual solve to a real linear solver, so a sequence of authentic ``(A, b)`` pairs can be replayed offline (see ``scripts/benchmark_linsolve.py``) instead of by rerunning the simulation. Its ``linsolverConfigFile`` selects the ``delegate`` solver, the dump ``directory``, and which solves to capture (``dumpAt`` / ``skipFirst`` / ``maxDumps`` / ``instances``).
