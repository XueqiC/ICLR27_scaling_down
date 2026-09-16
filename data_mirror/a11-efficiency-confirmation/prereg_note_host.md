# A11 host note (2026-09-17 14:45 EDT, before any target forward pass)

The forward measurement registered for the LONI Slurm job runs instead on the workstation GPU 2 (NVIDIA RTX
6000 Ada, UUID GPU-ANONYMIZED), with the same code (analysis/a11_measure.py), frozen probes, model
revisions, bf16 weights and densities. The LONI job 1029752 was cancelled while still pending so the
cluster capacity could go to another project. The frozen a11_measure.py is byte-identical to the registered file; its Slurm-environment guard is
satisfied by setting SLURM_JOB_ID=the workstation-local-a11 in the the workstation runner (a logistics check, not part of
the registration); nothing in plan.json or predictions.json changes.
