# V77 Pythia weight identity

Audited 23 cached step revisions, including all 22 revisions with raw pruning/quantization measurements. Streamed SHA-256 over 30 unique blobs (103,920,787,466 bytes).

Pythia-2.8B step16000 and step143000 select different serialized blobs. The step64000 and step143000 snapshots select the same blob. All three selections are cross-checked against the v72 freeze provenance.

**Critical weight-state finding:** step16000, step64000 and step143000 have identical learned parameter values modulo the sign of zero. Among 388 parameter tensors, 387 are byte-identical between step16000 and step143000. The only difference is `gpt_neox.layers.8.input_layernorm.bias[201]`: step16000 stores `+0.0` (`0x0000`), step143000 stores `-0.0` (`0x8000`). Step16000 additionally serializes 96 non-parameter buffers. **Distinct file hashes therefore do not establish two independent Pythia training states for v72.** This audit does not modify the existing measurements or fits.

## Coincident revision groups

- `pythia-2.8b@step143000`, `pythia-2.8b@step64000`

No other complete selected weight sets coincide among successfully verified revisions.

## Selected weight blobs

| Revision | Measured panel state | Format/file | Bytes | Actual SHA-256 |
|---|---|---|---:|---|
| pythia-1.4b@step112000 | yes | model-00001-of-00002.safetensors | 4,987,196,512 | `ececf53cd7e0bd3ae9c30606ce985f58f1530e4be60e6bdf05b8bf60b1320e75` |
| pythia-1.4b@step112000 | yes | model-00002-of-00002.safetensors | 671,429,088 | `2a8aebed981d6673820e3b6cbe4d923c0fb2b92445b0d4b25521285d27e86c66` |
| pythia-1.4b@step143000 | yes | model-00001-of-00002.safetensors | 4,987,196,512 | `2066abb3d75f8938ecac701b0982ed24fdbabaa3924967b583affa8fafdc62eb` |
| pythia-1.4b@step143000 | yes | model-00002-of-00002.safetensors | 671,429,088 | `d18df8a16798bb6d069c665c79a7589d3434b33fe2f18bf2637cbe2b35690565` |
| pythia-1.4b@step16000 | yes | model-00001-of-00002.safetensors | 4,987,196,512 | `63df2fa511d358ee6b01a6426a3287f23232b2803a2f49a52cb8e84b2f08fbce` |
| pythia-1.4b@step16000 | yes | model-00002-of-00002.safetensors | 671,429,088 | `03e7353100336a983c098882d6d4e0dba208f3c2dfc362ddc75e470e392a7a1d` |
| pythia-1.4b@step64000 | yes | model-00001-of-00002.safetensors | 4,987,196,512 | `69d98ab6078779832830495e34f20db9a3b95f5bcfeead2d06dede30d923c2b9` |
| pythia-1.4b@step64000 | yes | model-00002-of-00002.safetensors | 671,429,088 | `ced120c7eeb8260f62061a6e4ed9e21ebab05b6bbf240ae1af2c686cac15ee16` |
| pythia-1.4b@step96000 | yes | model-00001-of-00002.safetensors | 4,987,196,512 | `70dbf4f9ed7d0652ea3260f9a46fcb46de48faa11f60a36b6e6d6804f070beb8` |
| pythia-1.4b@step96000 | yes | model-00002-of-00002.safetensors | 671,429,088 | `0378430878c0c7c02b213821aaac94da2d81086cde6dc4999cbec86cc377ccae` |
| pythia-160m@step143000 | yes | model.safetensors | 649,308,728 | `d829d1a5cf66032491679d64c5b18e85b82d37833a99c346905668b8553084d5` |
| pythia-160m@step16000 | yes | model.safetensors | 649,308,728 | `b9cabaa6801bede34659808394d5b148bc5553eb47191b4c3385106afb34367c` |
| pythia-160m@step64000 | yes | model.safetensors | 649,308,728 | `0e054244779c34938b39db958396bf840c0ab22dac4fc172d6cd96d270ce5859` |
| pythia-160m@step96000 | yes | model.safetensors | 649,308,728 | `c0a0cfe72389a4c9fe8d910b4c427bf7b30107b2574d88f9d31907dae3fab5eb` |
| pythia-1b@step112000 | yes | model.safetensors | 4,047,149,576 | `b7a8307519cae97ac4dd6111231addcac7f3a65860e7ee121df281942495d858` |
| pythia-1b@step32000 | yes | model.safetensors | 4,047,149,576 | `57bfd83c56e38fca4d2bef4100e2ccd094bdf4ee97f6c42cc825f81645e2a80f` |
| pythia-1b@step96000 | yes | model.safetensors | 4,047,149,576 | `d0f23956b1eb0c9cfc82f3bc6bc2ca8f0470bf2445c83e61792b6dc072edbccc` |
| pythia-2.8b@step143000 | yes | model.safetensors | 5,550,463,728 | `462f2b960062159c3779eb1cfe5829783ea72da66b020bcbabf86bc6dd33063c` |
| pythia-2.8b@step16000 | yes | model.safetensors | 5,684,693,096 | `ab496f1c3fd79e3c749a9d5414136a2c8e4224f94eecb261970315cdb0f813fe` |
| pythia-2.8b@step64000 | cache check only | model.safetensors | 5,550,463,728 | `462f2b960062159c3779eb1cfe5829783ea72da66b020bcbabf86bc6dd33063c` |
| pythia-410m@step143000 | yes | model.safetensors | 1,621,370,224 | `1c88b0bf18293fae0e09c1d978b14b1adb08cb6a83154463133982fe460fcea4` |
| pythia-410m@step16000 | yes | model.safetensors | 1,621,370,224 | `1d3b5176e12886e00546119571cba4c48ad3d66c920e632216431f33cd882bed` |
| pythia-410m@step48000 | yes | model.safetensors | 1,621,370,224 | `09aa8bebb9260605cca30049a4d57aa462e0abdcb6498e13c41706a9d2c0c532` |
| pythia-410m@step64000 | yes | model.safetensors | 1,621,370,224 | `d5c2f8ba697b84dca4c35e77c8e26da3264e0f70b7a350134afd39804b90b8f4` |
| pythia-410m@step96000 | yes | model.safetensors | 1,621,370,224 | `c940e8184c07037cbef288a9ade2d7f0d06b3a3384019557abca905ce8678696` |
| pythia-6.9b@step112000 | yes | pytorch_model-00001-of-00002.bin | 9,911,682,774 | `8c0241797f86b84de9bf7b696098a24bfa9e5943604598114e40fbe210ce4d91` |
| pythia-6.9b@step112000 | yes | pytorch_model-00002-of-00002.bin | 3,937,306,520 | `2463dab7218d98d15d267b0c693d59446b3f6f57153a4559b90dcab05d65a32a` |
| pythia-6.9b@step32000 | yes | pytorch_model-00001-of-00002.bin | 9,911,682,774 | `1ac5ce5069c216dbf1e5609f01ac84f7f37c1c6fe251249efd2ac7f2e258b360` |
| pythia-6.9b@step32000 | yes | pytorch_model-00002-of-00002.bin | 3,937,306,520 | `57ef239aa76f85941d0cd33f6b679354a2dfcc9e2ca1f131cb41f38fc99d7a80` |
| pythia-6.9b@step80000 | yes | pytorch_model-00001-of-00002.bin | 9,911,682,774 | `dc940c3bc923748a91327ee12711f918ff68714294b26b04e06975754af990d9` |
| pythia-6.9b@step80000 | yes | pytorch_model-00002-of-00002.bin | 3,937,306,520 | `ccd93b0d64d8a36b641c1be231dbd45dcdbb179ef4fa12fd005ab92d5197f766` |

The step16000 snapshot also contains two safetensors shards. They are not the selected weights: the loader chooses its monolithic `model.safetensors` first. Pythia-6.9B uses two `.bin` shards per revision; both are included in the identity signature.

## Provenance and limits

- Identity is of serialized weight blobs, not a tensor-value equivalence test across different serializations. The supplemental Pythia-2.8B check compares all learned tensors and identifies signed-zero-only differences.
- Step labels are cache revision labels; coincident blobs are not independent weight states.
- Only v72 supplies frozen blob-selection provenance; other panels are checked against their current cached revisions. PyTorch bin shards are hashed without deserialization.
- JSON includes resolved snapshot commits, symlink targets, actual blob paths, byte lengths, LFS address comparisons, measurement evidence and v72 comparisons.
