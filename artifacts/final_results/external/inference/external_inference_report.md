# SATNET Phase 4B Frozen Real-Data External Model Inference

External evaluation on real-world Starlink orbital observations transformed through the frozen SATNET feature and network-construction methodology.

## Status

REAL-DATA EXTERNAL MODEL INFERENCE COMPLETE — EXTERNAL VALIDATION RESULTS FROZEN

Preflight: PASS; evaluations: 20/20; failures: 0; runtime_seconds: 4.593000
External bundle SHA-256: `9abede2f83adbfc96e8267805e3253d6dbcd6f5240d675729d5677aa053b7e28`
Adapter SHA-256: `83ebc1e4d31b6aa8ff9b1574e3a39b4b9716d1b182ebe2fd5e7e961ed4ef4e23`
tle_2025.parquet SHA-256: `9e9339cdbfc536cb91a1b5a26c898f193fdb683202f1cb1ceb7cee8bda81f64e`
Authorization SHA-256: `13e8f0bd7960fcbdae0e283d6f7e1c61545f90bb01b38a864767ed547d30ad3a`

## Frozen model/checkpoint hashes

```json
[
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_42\\final_model.joblib",
    "seed": 42,
    "sha256": "cefea7b3ce8a0f3dc59f436f83ed08933a26ecdb4a091802c1a81302823ca7e6",
    "task": "rf_space_classification"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_123\\final_model.joblib",
    "seed": 123,
    "sha256": "973b7ea709cac3b8ea3948af98f16703cd2e8361fd8e437a9862010a47ce1701",
    "task": "rf_space_classification"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_456\\final_model.joblib",
    "seed": 456,
    "sha256": "6b667f6ecd30cc33331430350b4d5f2fcce3960129b02ed3279f8da0b505ef03",
    "task": "rf_space_classification"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_789\\final_model.joblib",
    "seed": 789,
    "sha256": "0ff4d68b1475af3069f40fcc421c3f475733a72b401040b582943b14df006e22",
    "task": "rf_space_classification"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_2026\\final_model.joblib",
    "seed": 2026,
    "sha256": "7d2994fd3209cd3aa36c3f173b743ada3ccb31b5f4f4c9ba4d247258fc6b7f59",
    "task": "rf_space_classification"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_42\\best_validation_checkpoint.pt",
    "seed": 42,
    "sha256": "7d5262bf58c05848d6230e044a1ed46ee3d184d04aa1a1fa2c42eb93f787f3e0",
    "task": "tgnn_space_classification"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_123\\best_validation_checkpoint.pt",
    "seed": 123,
    "sha256": "d07d57579676d97ae6092d234e1e9b3290f50fe56befe984b429571dc364206c",
    "task": "tgnn_space_classification"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_456\\best_validation_checkpoint.pt",
    "seed": 456,
    "sha256": "6e8e19c7bdb1109ea4e1435a79cdb5695a662c8662878eb1802719581a4bdcac",
    "task": "tgnn_space_classification"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_789\\best_validation_checkpoint.pt",
    "seed": 789,
    "sha256": "bade1c470a9cb164d070d777886809457a3f516b100e4823169c46e42cc5c734",
    "task": "tgnn_space_classification"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_2026\\best_validation_checkpoint.pt",
    "seed": 2026,
    "sha256": "a3ebff6d93d1c638aa353f139d382fbf68a7d6427214073dba2d4e524fcf79d0",
    "task": "tgnn_space_classification"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_42\\final_model.joblib",
    "seed": 42,
    "sha256": "bfacf778e6d9623d508571b50621868d0130e60b78ebecf31088d284269c40d8",
    "task": "rf_space_regression"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_123\\final_model.joblib",
    "seed": 123,
    "sha256": "22cb470105ca8f51b413ea58cf95c592fccb205357cd00a62da7897b703de217",
    "task": "rf_space_regression"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_456\\final_model.joblib",
    "seed": 456,
    "sha256": "14d4100886131303a33aae3d6e77443d7a94ffbc723c5f44ef8b53d7d6dd0ef2",
    "task": "rf_space_regression"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_789\\final_model.joblib",
    "seed": 789,
    "sha256": "00c5f250ccd4fa86746002211dd391ea5c6df0117fe42ccbc3c4c07ef2fe4f73",
    "task": "rf_space_regression"
  },
  {
    "kind": "final_model.joblib",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_2026\\final_model.joblib",
    "seed": 2026,
    "sha256": "aaec456c006d330c10d296a3f6a6631b28ad8da356306ea5bff1b5f447459991",
    "task": "rf_space_regression"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_42\\best_validation_checkpoint.pt",
    "seed": 42,
    "sha256": "22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a",
    "task": "tgnn_space_regression"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_123\\best_validation_checkpoint.pt",
    "seed": 123,
    "sha256": "308f4bf179212df640162c8ba45e5b53a2221a6cf3dd8e2fafef5425f41bf8b0",
    "task": "tgnn_space_regression"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_456\\best_validation_checkpoint.pt",
    "seed": 456,
    "sha256": "e42b224adc78f35c9b1fad69aa89139fb72d84044445d2b391d9ca2a257ab199",
    "task": "tgnn_space_regression"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_789\\best_validation_checkpoint.pt",
    "seed": 789,
    "sha256": "e231b66d8f4f4ef120a64e5a326a83611b02c5c8a1c2c91547186a57d0250fb3",
    "task": "tgnn_space_regression"
  },
  {
    "kind": "best_validation_checkpoint.pt",
    "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_2026\\best_validation_checkpoint.pt",
    "seed": 2026,
    "sha256": "11834cae573a65a9b8e22838cae3d0b8118d0f27da1fcdd3e804d3a057690a9e",
    "task": "tgnn_space_regression"
  }
]
```

## Prediction and metric hashes

```json
[
  {
    "metrics_sha256": "c6f6a1308bd89a2985dc7c67332333f727848a4cf3bdd0d23e5a38fa33e935e6",
    "predictions_sha256": "85881716a744e1f32170dc5302dcff818664a04c83ea665cb0bca25ef23eef3e",
    "seed": 42,
    "task": "rf_space_classification"
  },
  {
    "metrics_sha256": "b228728e423c387b42814f61a26f219c4b99bedca84319363987536379fb7006",
    "predictions_sha256": "4ab867f7d77895bef3d3d9deca832bc7b2a7c8015155ebdc5678f4837aa519ea",
    "seed": 123,
    "task": "rf_space_classification"
  },
  {
    "metrics_sha256": "3357531edc8e741d00a9f85036867e9b7b3d09b81ed1559c6470df0a5e50b427",
    "predictions_sha256": "4e18780386fad94d598222067969aa5cf47458d307bf8e76de0fd364c11df4f9",
    "seed": 456,
    "task": "rf_space_classification"
  },
  {
    "metrics_sha256": "27f91a882a5edd215f672933c9c4d68e14e73a4920d9d8066334fbc6bec49de9",
    "predictions_sha256": "0a2ca201d228d5b1a34e3ce66f57ad2f059f76aa305d85ea5aae2009be0ba5bc",
    "seed": 789,
    "task": "rf_space_classification"
  },
  {
    "metrics_sha256": "a403d4ba36abc97a9f640eff57c617ec6a94e5631650e1bb9380a23d764e97f3",
    "predictions_sha256": "fbab1dc0e304cefb61c91a78957216000424b99d1b1f218f0bb6f0204eb41e3b",
    "seed": 2026,
    "task": "rf_space_classification"
  },
  {
    "metrics_sha256": "6a08d0c8fe6ddc74e4b21809dfea3bd83a5bab9b8c09078ecbb682e3f186b2e3",
    "predictions_sha256": "c80503036561cef6cc990263d9d894707b54a9fb9f13bf3e68735c94266d5086",
    "seed": 42,
    "task": "tgnn_space_classification"
  },
  {
    "metrics_sha256": "3f26671f49bce2a0472b9b14e6e94e8b6b128f3f71603f14e00f8fe1b61fe523",
    "predictions_sha256": "e06bcae0aa176f3b6b6cac01e92ed46a6b3b2cfe1d6384d7b50c986bcf886d49",
    "seed": 123,
    "task": "tgnn_space_classification"
  },
  {
    "metrics_sha256": "755c8e2df8d9b79294f2c1212d4903942a7033f155fda68cd8cb9f4dd0c48d92",
    "predictions_sha256": "f26f256d15bea4ee1981e2807ebf7847979f6fb183a60e1357c1dd2155232590",
    "seed": 456,
    "task": "tgnn_space_classification"
  },
  {
    "metrics_sha256": "8829c95e5cba2cc89585c2d35d8ba4d603319d4caec01133e97b6d67ca24c4ee",
    "predictions_sha256": "462f7cf5623d42813d93291acc9beb445ddd87e68d029819461228496bddecb3",
    "seed": 789,
    "task": "tgnn_space_classification"
  },
  {
    "metrics_sha256": "96c5ff1516a95199eb851132d5bfbac3f2e02879f92f844ddebf4ef6ebef3807",
    "predictions_sha256": "a46da7e95f88ed7b9098aa36654c97d8c3196b1e29e712e295332e0122ca5e0a",
    "seed": 2026,
    "task": "tgnn_space_classification"
  },
  {
    "metrics_sha256": "b36fc250f3c2a9fcbbfa85ecaac684689df37630cf481f4c1dbd71dfb55c5b5b",
    "predictions_sha256": "c3a4b0ef5b087a390be81848ad8abce2bb93dd99e7e8fe60f2a098eb51d2710e",
    "seed": 42,
    "task": "rf_space_regression"
  },
  {
    "metrics_sha256": "669a3dc7cc94b625b7bf8ed66b685d1fc54930489c4fb15f92acff24e5ca08f2",
    "predictions_sha256": "3caf4da38e7fe127c81f959aede50ede5ada1804fa824e49556621901dce4d5d",
    "seed": 123,
    "task": "rf_space_regression"
  },
  {
    "metrics_sha256": "a904e60db6dcccfcfb315434b027251c95795d2f86aede2f319ab6f2768d7d04",
    "predictions_sha256": "b8fad9fc3e913d5ecd2bbfa3d2f30a51be53186d1c52eba5e23fa0096b5b3957",
    "seed": 456,
    "task": "rf_space_regression"
  },
  {
    "metrics_sha256": "d1f7ff836b3d3d54294bf24bd4dde44b987fed69157146f8d3ff904658fbff4b",
    "predictions_sha256": "065ff5c28ae848e7ba5cfe42a336ab3743c46bac4f1121d4fc163320144ad878",
    "seed": 789,
    "task": "rf_space_regression"
  },
  {
    "metrics_sha256": "52e2cd3e869600c6507d0418c564b3612170fd7dce29f15188b58375e9f456c4",
    "predictions_sha256": "ed33ff10f4bfea6f68c783a2f2538eb46456b1982209dfc8a42c245913e66214",
    "seed": 2026,
    "task": "rf_space_regression"
  },
  {
    "metrics_sha256": "ca2f332f2aec13cffdd34be2a134a9d16cfa63d139204d7a3052d4ee8f8706ca",
    "predictions_sha256": "c32a5f0d5ac63631144f11bb4a14e3f453b86ee25adeb80e0cdfa6a48d73b3cd",
    "seed": 42,
    "task": "tgnn_space_regression"
  },
  {
    "metrics_sha256": "2723ecdb528ed1abdf205051cb16f1f9d18353f73b289d4b52ae412d4dd18a33",
    "predictions_sha256": "2f6521a94e8acd979de039b650a0a2ba38859f96de6c2bc415cedd811d4f9080",
    "seed": 123,
    "task": "tgnn_space_regression"
  },
  {
    "metrics_sha256": "2992ce83453a1733c169ffd1469fbc5bfb12bdbee15fb0074edc3d99d6de8a57",
    "predictions_sha256": "948c9b82831994a2f459e34c1fe2433df416b26c7c843e094c2ff785ed708d49",
    "seed": 456,
    "task": "tgnn_space_regression"
  },
  {
    "metrics_sha256": "bc7a980ed1cbe4dc1a4504f4ebeae2f1d275e9d5525eb52ca99527ce1879a452",
    "predictions_sha256": "003125ff8d86ea911081c2ca53965a3bba5209d7d4216ddcec884ac30804d86c",
    "seed": 789,
    "task": "tgnn_space_regression"
  },
  {
    "metrics_sha256": "eda217362cf1f96ea3273a098bae5259458ae97873b5e97219518ce6125bdb7a",
    "predictions_sha256": "badb4bfb7bb435298227e308be86da88e968043baca827b04e64670dddf24cf5",
    "seed": 2026,
    "task": "tgnn_space_regression"
  }
]
```

## Classification

The external classification set contains 300 positives and 0 negatives. Results are descriptive only and do not estimate two-class discrimination. Specificity, two-class balanced accuracy, ROC-AUC, and negative-class precision/recall/F1 are not estimable or meaningful.
{
  "RF": {
    "false_negative_rate_per_seed": {
      "42": 0.0,
      "123": 0.0,
      "456": 0.0,
      "789": 0.0,
      "2026": 0.0
    },
    "max_sensitivity": 1.0,
    "mean_sensitivity": 1.0,
    "min_sensitivity": 1.0,
    "sensitivity_per_seed": {
      "42": 1.0,
      "123": 1.0,
      "456": 1.0,
      "789": 1.0,
      "2026": 1.0
    },
    "standard_deviation_sensitivity": 0.0
  },
  "TGNN": {
    "false_negative_rate_per_seed": {
      "42": 0.0,
      "123": 0.0,
      "456": 0.0,
      "789": 0.0,
      "2026": 0.0
    },
    "max_sensitivity": 1.0,
    "mean_sensitivity": 1.0,
    "min_sensitivity": 1.0,
    "sensitivity_per_seed": {
      "42": 1.0,
      "123": 1.0,
      "456": 1.0,
      "789": 1.0,
      "2026": 1.0
    },
    "standard_deviation_sensitivity": 0.0
  }
}

## Regression

{
  "RF": {
    "mae_per_seed": {
      "42": 0.1273978715832696,
      "123": 0.12765187652567322,
      "456": 0.12744006491892657,
      "789": 0.1273781063462055,
      "2026": 0.12745201391140437
    },
    "max_mae": 0.12765187652567322,
    "mean_mae": 0.12746398665709585,
    "min_mae": 0.1273781063462055,
    "population_standard_deviation_mae": 9.773927531563766e-05,
    "seed_42_mae": 0.1273978715832696
  },
  "TGNN": {
    "mae_per_seed": {
      "42": 0.06290224140586045,
      "123": 0.0627940192249717,
      "456": 0.06345601852596121,
      "789": 0.05722958686430618,
      "2026": 0.06799850607345659
    },
    "max_mae": 0.06799850607345659,
    "mean_mae": 0.06287607441891123,
    "min_mae": 0.05722958686430618,
    "population_standard_deviation_mae": 0.0034195286427564386,
    "seed_42_mae": 0.06290224140586045
  }
}

## Synthetic baselines and generalization

{
  "synthetic_train_mean": {
    "mae": 0.12919839166666672,
    "maximum_absolute_error": 0.27147214569161005,
    "median_absolute_error": 0.1360554790249434,
    "prediction_max": 0.30272214569161005,
    "prediction_mean": 0.3027221456916101,
    "prediction_median": 0.30272214569161005,
    "prediction_min": 0.30272214569161005,
    "prediction_standard_deviation": 5.551115123125783e-17,
    "predictions_above_one": 0,
    "predictions_below_zero": 0,
    "predictor_value": 0.30272214569161005,
    "r2": -1.7635843811974077,
    "rmse": 0.1418458992371677,
    "target_mean": 0.18940945767195766,
    "target_median": 0.16666666666666666,
    "target_standard_deviation": 0.0853258100703513
  },
  "synthetic_train_median": {
    "mae": 0.07018472222222222,
    "maximum_absolute_error": 0.3666666666666667,
    "median_absolute_error": 0.0380952380952381,
    "prediction_max": 0.13333333333333333,
    "prediction_mean": 0.13333333333333333,
    "prediction_median": 0.13333333333333333,
    "prediction_min": 0.13333333333333333,
    "prediction_standard_deviation": 0.0,
    "predictions_above_one": 0,
    "predictions_below_zero": 0,
    "predictor_value": 0.13333333333333333,
    "r2": -0.43191187019878696,
    "rmse": 0.1021030145735302,
    "target_mean": 0.18940945767195766,
    "target_median": 0.16666666666666666,
    "target_standard_deviation": 0.0853258100703513
  }
}
{
  "RF": [
    {
      "absolute_mae_change": 0.08016675909636975,
      "external_real_data_driven_mae": 0.1273978715832696,
      "relative_mae_change": 1.697329469395942,
      "seed": 42,
      "synthetic_heldout_test_mae": 0.04723111248689984
    },
    {
      "absolute_mae_change": 0.08054076158457922,
      "external_real_data_driven_mae": 0.12765187652567322,
      "relative_mae_change": 1.7095914984241916,
      "seed": 123,
      "synthetic_heldout_test_mae": 0.047111114941094004
    },
    {
      "absolute_mae_change": 0.08020874333796271,
      "external_real_data_driven_mae": 0.12744006491892657,
      "relative_mae_change": 1.6982108620540928,
      "seed": 456,
      "synthetic_heldout_test_mae": 0.04723132158096386
    },
    {
      "absolute_mae_change": 0.08002216540388618,
      "external_real_data_driven_mae": 0.1273781063462055,
      "relative_mae_change": 1.6898020356380439,
      "seed": 789,
      "synthetic_heldout_test_mae": 0.047355940942319326
    },
    {
      "absolute_mae_change": 0.08034291944079042,
      "external_real_data_driven_mae": 0.12745201391140437,
      "relative_mae_change": 1.7054651621653927,
      "seed": 2026,
      "synthetic_heldout_test_mae": 0.04710909447061395
    }
  ],
  "TGNN": [
    {
      "absolute_mae_change": 0.031275557565575554,
      "external_real_data_driven_mae": 0.06290224140586045,
      "relative_mae_change": 0.9888977840205272,
      "seed": 42,
      "synthetic_heldout_test_mae": 0.0316266838402849
    },
    {
      "absolute_mae_change": 0.031467535489311936,
      "external_real_data_driven_mae": 0.0627940192249717,
      "relative_mae_change": 1.0045026360073603,
      "seed": 123,
      "synthetic_heldout_test_mae": 0.031326483735659764
    },
    {
      "absolute_mae_change": 0.03313114480458869,
      "external_real_data_driven_mae": 0.06345601852596121,
      "relative_mae_change": 1.0925402397055444,
      "seed": 456,
      "synthetic_heldout_test_mae": 0.030324873721372516
    },
    {
      "absolute_mae_change": 0.02981478960365845,
      "external_real_data_driven_mae": 0.05722958686430618,
      "relative_mae_change": 1.0875436838066923,
      "seed": 789,
      "synthetic_heldout_test_mae": 0.02741479726064773
    },
    {
      "absolute_mae_change": 0.0357178063267627,
      "external_real_data_driven_mae": 0.06799850607345659,
      "relative_mae_change": 1.1064755908960997,
      "seed": 2026,
      "synthetic_heldout_test_mae": 0.03228069974669389
    }
  ]
}

## Seed-42 paired RF versus TGNN

{
  "bootstrap_mean_delta": 0.06462429794351755,
  "bootstrap_sd": 0.004447060475846749,
  "bootstrap_seed": 20260820,
  "confidence_interval": "95% percentile",
  "observed_delta_mae_rf_minus_tgnn": 0.06449563017740914,
  "observed_rf_mae": 0.1273978715832696,
  "observed_tgnn_mae": 0.06290224140586045,
  "paired": true,
  "percentile_2_5": 0.05592972863883819,
  "percentile_97_5": 0.0734203385275529,
  "replicates": 2000,
  "sampling_unit": "episode_id",
  "valid_replicates": 2000
}
The synthetic model-family conclusion is **preserved** for seed 42 (RF MAE `0.1273978715832696`, TGNN MAE `0.06290224140586045`).

## Distribution shift carried forward

External edge failure probability mean: 0.7536; synthetic TRAIN mean: 0.1162.
External altitude mean: 519.602 km; synthetic TRAIN mean: 739.108 km.
External inclination mean: 53.169 degrees; synthetic TRAIN mean: 62.967 degrees.
This is interpreted as out-of-distribution generalization. The experiment does not independently validate proprietary Starlink ISL routing or service telemetry.

## Controls

No training, retraining, fine-tuning, hyperparameter search, threshold tuning, model selection, episode resampling/redesign, adapter modification, or external-data modification occurred.
