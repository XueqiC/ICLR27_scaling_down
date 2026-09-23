# P3b end-marker control: paired readouts on the 384 fresh items

## pythia-1.4b--step120000: student EleutherAI/pythia-410m@step120000, control run results/v12-distill/EleutherAI--pythia-410m--step120000/gpt-5.6-luna_full_600_p3-eos-s3-pythia-1.4b--step120000_lora

| Readout | S3 student (no marker) | Control (marker) | Control minus S3 [95%] |
|---|---:|---:|---:|
| Token-weighted loss (nats) | 4.860 | 4.853 | -0.007 [-0.060, +0.047] |
| Exact match, 32 tokens | 0.065 | 0.083 | +0.018 [-0.010, +0.049] |
| Exact match, 96 tokens (span) | 0.076 | 0.076 | +0.000 [-0.034, +0.031] |
| Token F1, 96 tokens (span) | 0.117 | 0.106 | -0.011 [-0.045, +0.023] |
| Reached the 96-token cap | 0.628 | 0.005 | -0.622 [-0.672, -0.573] |
| Generated tokens (mean) | 63.8 | 4.6 | -59.122 [-63.214, -54.807] |

- Control minus quant_only (quant:channel_b4): loss -0.453 [-0.574, -0.329], exact match 96 -0.070 [-0.107, -0.034] (quant_only: loss 5.306, EM96 0.146, cap 0.128)
- Control minus pristine (pristine:student): loss -0.375 [-0.474, -0.274], exact match 96 -0.104 [-0.143, -0.065] (pristine: loss 5.228, EM96 0.180, cap 0.219)
- Probe losses after training (math / code / qa): S3 student {'math': 1.5731036937435736, 'code': 1.5841454718326597, 'qa': 4.4326283269961975}; control {'math': 1.5998378549394923, 'code': 1.6420846208699786, 'qa': 4.403160646387833}

