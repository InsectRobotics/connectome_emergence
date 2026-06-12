"""Experiment drivers + analysis/figure utilities for code.

Package marker for the ``scripts`` subpackage. This file
contains no executable code (no imports, no re-exports); its sole purpose is to make
the ``scripts/`` directory an importable Python package so that modules inside it can
be referenced as e.g. ``scripts.run_training``. Because there
are no symbols re-exported here, importing the package itself is essentially a no-op
beyond registering the package namespace.

Where this fits in the pipeline
--------------------------------
The repo implements a connectome-constrained spiking model of the larval Drosophila
olfactory pathway (OR responses from Kreher 2008 -> ORN -> LN -> PN -> KC <- APL ->
linear decoder over 28 odors), with connectivity fixed by the Winding 2023 connectome
and ~449 free biological parameters learned by ANN-to-SNN transfer (rate teacher ->
spiking student). The core model lives in ``model.py`` / ``layers.py``; this
``scripts`` package holds the *drivers* and *utilities* that exercise that model:

  * ``run_training.py``        - main teacher/student training + concentration-invariance
                                 driver (defines ``run_concentration_invariance`` etc.).
  * ``run_ablation.py``        - architecture ablations (e.g. ``ablate_gap_junctions``,
                                 ``ablate_apl``) used to attribute behaviour to circuit
                                 motifs.
  * ``run_odor_mixtures.py``   - odor-mixture experiments / Honegger-style mixture metrics.
  * ``run_task_complexity.py`` - task-complexity sweeps.
  * ``run_teacher_consistency.py`` - teacher-vs-teacher consistency checks.
  * ``recompute_one.py``       - recomputes ONE checkpoint's metrics via analysis/recompute.py
                                 (eval seed 42, deterministic) -> results/recompute_cache/ JSON;
                                 the notebook's R1/R3/R4/STD cells fan it out as parallel subprocesses.
  * ``regen.sh``               - shell entry point to regenerate results/figures end to end.

Note on canonical constants: the runtime constants that define the canonical results
(``N_STEPS=30``, sparsity-loss offset/target of 0.02/0.05, the ``g_soma`` clamp of
[1, 20] nS) are defined for these drivers; ``model.py``'s in-class ``compute_loss`` and
``n_steps`` now defaults to 30 in model.py; the drivers also set it explicitly.
"""
