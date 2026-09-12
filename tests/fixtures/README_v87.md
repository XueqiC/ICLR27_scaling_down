V87 offline fixtures:

- `v12_eval_legacy.json`: byte-for-byte copy of
  `results/v12-distill/gemma3-4b/claude-sonnet-4-6_full_600/eval.json`.
  SHA256: `1bae1161f09faddd59da022582ce2bde014a68163846ffd7b0ca3cd663d12843`.
  This is the only historical result read for preparation; tests read this copy.
- `v12_eval_with_synthetic_records.json`: all legacy fields preserved, with
  explicitly **synthetic**, nonuniform-length records reproducing each stored
  loss. These are arithmetic/reader fixtures, not recovered model outputs.
  Real historical sample NLLs are unavailable from aggregate-only JSON.
- `v54_symmetric_before.pt`: frozen test tensor outputs captured **before** the
  V54 extension. CPU generator seed 87; random 4x11 tensor, first row zero;
  fp32/bfloat16; bits 2/3/5/8; group sizes None/0/3/11/16. Forty exact outputs.
  Tests load with `weights_only=True` and compare using `torch.equal`.

No test reads from the project's `results/` tree.
