# P3a token-level audit of end-of-answer supervision (selection-round students)

Training examples rebuilt with `v12_distill.load_sft_records` and tokenised with `tokenize_sft_example` (the functions the students were trained with); stop conditions from `v15_accuracy_link`.

| Reference | Student | Examples | EOS id (token) | Targets with EOS supervised | Ending in newline | Ending in sentence end | Supervised tokens / epoch | Extra with EOS / epoch | Inference stop |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| pythia-410m--step120000 | EleutherAI/pythia-160m@step120000 | 1781 | 0 (<|endoftext|>) | 0 | 4 | 1437 | 153748 | 1781 | eos [0] + ('\nQuestion:', '\nContext:') |
| pythia-1.4b--step120000 | EleutherAI/pythia-410m@step120000 | 1781 | 0 (<|endoftext|>) | 0 | 4 | 1437 | 153748 | 1781 | eos [0] + ('\nQuestion:', '\nContext:') |
| gemma3-1b | google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 | 1781 | 1 (<eos>) | 0 | 4 | 1437 | 158391 | 1781 | eos [1] + ('\nQuestion:', '\nContext:') |
| gemma3-4b | google/gemma-3-1b-pt@fcf18a2a879aab110ca39f8bffbccd5d49d8eb29 | 1781 | 1 (<eos>) | 0 | 4 | 1437 | 158391 | 1781 | eos [1] + ('\nQuestion:', '\nContext:') |

## Sampled examples (last supervised tokens)

- pythia-410m--step120000 / qa: 530 tokens, 512 prompt tokens masked, 18 supervised; first label is the first completion token: True; last supervised tokens ['Ġfrom', 'Ġ305', 'Ġto', 'Ġ30', 'ĠBC', '.'] (ids [432, 26402, 281, 1884, 12895, 15]); target tail 'The Ptolemaic dynasty ruled for 275 years, from 305 to 30 BC.'
- pythia-410m--step120000 / qa: 514 tokens, 512 prompt tokens masked, 2 supervised; first label is the first completion token: True; last supervised tokens ['1957', '.'] (ids [35364, 15]); target tail '1957.'
- pythia-410m--step120000 / math: 187 tokens, 70 prompt tokens masked, 117 supervised; first label is the first completion token: True; last supervised tokens ['Ġwith', 'Ġ**', '60', 'Ġmar', 'bles', '**.'] (ids [342, 1401, 1549, 2304, 9143, 12530]); target tail 'tal marbles:\n   \\[\n   20 + 40 = 60\n   \\]\n\n#### Baez ends up with **60 marbles**.'
- pythia-1.4b--step120000 / qa: 530 tokens, 512 prompt tokens masked, 18 supervised; first label is the first completion token: True; last supervised tokens ['Ġfrom', 'Ġ305', 'Ġto', 'Ġ30', 'ĠBC', '.'] (ids [432, 26402, 281, 1884, 12895, 15]); target tail 'The Ptolemaic dynasty ruled for 275 years, from 305 to 30 BC.'
- pythia-1.4b--step120000 / qa: 514 tokens, 512 prompt tokens masked, 2 supervised; first label is the first completion token: True; last supervised tokens ['1957', '.'] (ids [35364, 15]); target tail '1957.'
- pythia-1.4b--step120000 / math: 187 tokens, 70 prompt tokens masked, 117 supervised; first label is the first completion token: True; last supervised tokens ['Ġwith', 'Ġ**', '60', 'Ġmar', 'bles', '**.'] (ids [342, 1401, 1549, 2304, 9143, 12530]); target tail 'tal marbles:\n   \\[\n   20 + 40 = 60\n   \\]\n\n#### Baez ends up with **60 marbles**.'
- gemma3-1b / qa: 536 tokens, 512 prompt tokens masked, 24 supervised; first label is the first completion token: True; last supervised tokens ['▁to', '▁', '3', '0', '▁BC', '.'] (ids [531, 236743, 236800, 236771, 19339, 236761]); target tail 'The Ptolemaic dynasty ruled for 275 years, from 305 to 30 BC.'
- gemma3-1b / qa: 517 tokens, 512 prompt tokens masked, 5 supervised; first label is the first completion token: True; last supervised tokens ['1', '9', '5', '7', '.'] (ids [236770, 236819, 236810, 236832, 236761]); target tail '1957.'
- gemma3-1b / math: 203 tokens, 69 prompt tokens masked, 134 supervised; first label is the first completion token: True; last supervised tokens ['▁with', '▁**', '6', '0', '▁marbles', '**.'] (ids [607, 5213, 236825, 236771, 147437, 84750]); target tail 'tal marbles:\n   \\[\n   20 + 40 = 60\n   \\]\n\n#### Baez ends up with **60 marbles**.'
- gemma3-4b / qa: 536 tokens, 512 prompt tokens masked, 24 supervised; first label is the first completion token: True; last supervised tokens ['▁to', '▁', '3', '0', '▁BC', '.'] (ids [531, 236743, 236800, 236771, 19339, 236761]); target tail 'The Ptolemaic dynasty ruled for 275 years, from 305 to 30 BC.'
- gemma3-4b / qa: 517 tokens, 512 prompt tokens masked, 5 supervised; first label is the first completion token: True; last supervised tokens ['1', '9', '5', '7', '.'] (ids [236770, 236819, 236810, 236832, 236761]); target tail '1957.'
- gemma3-4b / math: 203 tokens, 69 prompt tokens masked, 134 supervised; first label is the first completion token: True; last supervised tokens ['▁with', '▁**', '6', '0', '▁marbles', '**.'] (ids [607, 5213, 236825, 236771, 147437, 84750]); target tail 'tal marbles:\n   \\[\n   20 + 40 = 60\n   \\]\n\n#### Baez ends up with **60 marbles**.'
