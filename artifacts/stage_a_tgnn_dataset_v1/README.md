# Stage A Temporal GNN Dataset v1

Verdict: **STAGE A TGNN DATASET V1 CONSTRUCTED — READY FOR NARROW INDEPENDENT DATASET GATE**

This dataset was constructed only from the accepted, frozen production evidence. It does not rebuild the accepted RF dataset, train or tune a model, evaluate test targets, or run the independent TGNN acceptance gate.

## Graph scope

The accepted G3 `IntegratedGroundGraphRecord` artifacts define a combined satellite-ground heterogeneous graph. Each run is one ordered sequence of all 11 valid timesteps. Satellite nodes use inherited G3 operational semantics: failed satellites are absent. Selected ground-station nodes remain present under G5 persistent failure overlay semantics and expose `operational_indicator=0` when failed.

## Counts

- Train: 70 designs / 350 run sequences
- Validation: 15 designs / 75 run sequences
- Sealed test: 15 designs / 75 run sequence indexes
- Five realizations per design; zero design and run overlap
- Sequence length: {11: 500}
- Node counts by timestep: {19: 22, 20: 11, 22: 11, 23: 11, 25: 11, 26: 22, 27: 77, 28: 55, 29: 99, 30: 55, 31: 22, 32: 66, 33: 66, 34: 77, 35: 55, 36: 33, 37: 44, 38: 33, 39: 33, 40: 44, 41: 33, 42: 66, 43: 66, 44: 154, 45: 253, 46: 143, 47: 110, 48: 231, 49: 187, 50: 275, 51: 187, 52: 220, 53: 143, 54: 88, 55: 77, 56: 110, 57: 132, 58: 132, 59: 154, 60: 176, 61: 33, 62: 66, 63: 77, 64: 55, 65: 55, 66: 55, 67: 110, 68: 154, 69: 55, 70: 121, 71: 132, 72: 44, 73: 44, 74: 55, 75: 143, 76: 88, 77: 33, 78: 44, 79: 66, 80: 99, 81: 22, 82: 22, 83: 22, 84: 11, 89: 22, 90: 44, 91: 44}
- Physical edge counts by timestep: {0: 27, 1: 70, 2: 47, 3: 44, 4: 64, 5: 87, 6: 66, 7: 76, 8: 66, 9: 70, 10: 94, 11: 103, 12: 85, 13: 86, 14: 65, 15: 63, 16: 63, 17: 33, 18: 41, 19: 48, 20: 48, 21: 52, 22: 67, 23: 65, 24: 76, 25: 78, 26: 75, 27: 79, 28: 53, 29: 63, 30: 70, 31: 56, 32: 57, 33: 75, 34: 56, 35: 49, 36: 49, 37: 50, 38: 59, 39: 70, 40: 50, 41: 45, 42: 57, 43: 56, 44: 50, 45: 43, 46: 38, 47: 53, 48: 39, 49: 49, 50: 39, 51: 33, 52: 44, 53: 27, 54: 56, 55: 50, 56: 44, 57: 51, 58: 49, 59: 39, 60: 40, 61: 26, 62: 31, 63: 54, 64: 41, 65: 40, 66: 44, 67: 33, 68: 33, 69: 32, 70: 55, 71: 38, 72: 56, 73: 30, 74: 43, 75: 53, 76: 40, 77: 39, 78: 37, 79: 39, 80: 44, 81: 39, 82: 45, 83: 49, 84: 46, 85: 39, 86: 57, 87: 42, 88: 51, 89: 38, 90: 42, 91: 48, 92: 39, 93: 37, 94: 46, 95: 43, 96: 44, 97: 50, 98: 47, 99: 40, 100: 30, 101: 27, 102: 16, 103: 17, 104: 11, 105: 16, 106: 9, 107: 8, 108: 10, 109: 10, 110: 9, 111: 10, 112: 12, 113: 10, 114: 3, 115: 11, 116: 8, 117: 8, 118: 8, 119: 9, 120: 7, 121: 10, 122: 11, 123: 12, 124: 9, 125: 3, 126: 4, 134: 5, 135: 30, 137: 5, 138: 10, 140: 5}
- Serialized directed edge counts by timestep: {0: 27, 2: 70, 4: 47, 6: 44, 8: 64, 10: 87, 12: 66, 14: 76, 16: 66, 18: 70, 20: 94, 22: 103, 24: 85, 26: 86, 28: 65, 30: 63, 32: 63, 34: 33, 36: 41, 38: 48, 40: 48, 42: 52, 44: 67, 46: 65, 48: 76, 50: 78, 52: 75, 54: 79, 56: 53, 58: 63, 60: 70, 62: 56, 64: 57, 66: 75, 68: 56, 70: 49, 72: 49, 74: 50, 76: 59, 78: 70, 80: 50, 82: 45, 84: 57, 86: 56, 88: 50, 90: 43, 92: 38, 94: 53, 96: 39, 98: 49, 100: 39, 102: 33, 104: 44, 106: 27, 108: 56, 110: 50, 112: 44, 114: 51, 116: 49, 118: 39, 120: 40, 122: 26, 124: 31, 126: 54, 128: 41, 130: 40, 132: 44, 134: 33, 136: 33, 138: 32, 140: 55, 142: 38, 144: 56, 146: 30, 148: 43, 150: 53, 152: 40, 154: 39, 156: 37, 158: 39, 160: 44, 162: 39, 164: 45, 166: 49, 168: 46, 170: 39, 172: 57, 174: 42, 176: 51, 178: 38, 180: 42, 182: 48, 184: 39, 186: 37, 188: 46, 190: 43, 192: 44, 194: 50, 196: 47, 198: 40, 200: 30, 202: 27, 204: 16, 206: 17, 208: 11, 210: 16, 212: 9, 214: 8, 216: 10, 218: 10, 220: 9, 222: 10, 224: 12, 226: 10, 228: 3, 230: 11, 232: 8, 234: 8, 236: 8, 238: 9, 240: 7, 242: 10, 244: 11, 246: 12, 248: 9, 250: 3, 252: 4, 268: 5, 270: 30, 274: 5, 276: 10, 280: 5}

## Features

- Node features: 14 float64 values; see `tgnn_node_feature_schema.json`.
- Edge attributes: 10 float64 values; see `tgnn_edge_schema.json`.
- Targets are run-level index labels only: `partition_any` maps exactly to `overall_threshold_breach_any`; `gcc_frac_min` maps exactly to `space_gcc_fraction_original_min`.
- No target, target derivative, future value, run summary, identifier, path, hash, seed, split, replay, or acceptance field is a learned graph feature.

## Storage

Each `sequences/run_NNN.tgnn` is a deterministic binary artifact with a canonical JSON header and non-pickle NumPy arrays. Variable node counts are represented by snapshot offsets and identity indexes; no padding or masking is used. Source undirected physical edges are serialized in both directions for PyTorch Geometric COO compatibility.

## Provenance and seal

The indexes and `tgnn_provenance_manifest.jsonl` retain frozen source paths and exact SHA-256 identities. The sealed test index contains no classification or regression target fields, no outcome summaries, and test target files were not read. The source freeze manifest SHA-256 is `b2d1fbd9510d3d828fe051b4da04f6747088ae00d78d65e0a9851dc85559844e` and the contract hash is `fc13a33a0a1af435189990e54b6c68efa56bbc207c0cae6e27e12bf031605930`.

See `tgnn_construction_report.json` for structural checks and `tgnn_leakage_exclusion_report.json` for excluded fields. The independent TGNN dataset acceptance gate was not performed.
