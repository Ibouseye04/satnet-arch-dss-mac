# Held-Out Synthetic Test v1

{
  "authorization_sha256": "c10439f100e347dbeb9027163ddc0f8b23e44612c3e659f7e70f2cdeff6f9534",
  "baselines": {
    "rf_integrated_classification": {
      "metrics": {
        "majority_class_classifier": {
          "accuracy": 0.9926666666666667,
          "balanced_accuracy": 0.5,
          "confusion_matrix": [
            [
              0,
              11
            ],
            [
              0,
              1489
            ]
          ],
          "f1_by_class": {
            "0": 0.0,
            "1": 0.9963198394111743
          },
          "macro_f1": 0.49815991970558715,
          "negative_class_count": 11,
          "positive_class_count": 1489,
          "pr_auc": 0.9926666666666667,
          "precision_by_class": {
            "0": 0.0,
            "1": 0.9926666666666667
          },
          "recall_by_class": {
            "0": 0.0,
            "1": 1.0
          },
          "roc_auc": null,
          "sensitivity": 1.0,
          "specificity": 0.0,
          "train_class_counts": {
            "0": 58,
            "1": 6942
          },
          "train_positive_prevalence": 0.9917142857142857,
          "weighted_f1": 0.9890134939221591
        },
        "stratified_random_classifier": {
          "accuracy": 0.9813333333333333,
          "balanced_accuracy": 0.49429147078576224,
          "confusion_matrix": [
            [
              0,
              11
            ],
            [
              17,
              1472
            ]
          ],
          "f1_by_class": {
            "0": 0.0,
            "1": 0.990578734858681
          },
          "macro_f1": 0.4952893674293405,
          "negative_class_count": 11,
          "positive_class_count": 1489,
          "pr_auc": 0.9925835625938081,
          "precision_by_class": {
            "0": 0.0,
            "1": 0.9925826028320971
          },
          "recall_by_class": {
            "0": 0.0,
            "1": 0.9885829415715245
          },
          "roc_auc": 0.49429147078576224,
          "seed": 42,
          "sensitivity": 0.9885829415715245,
          "specificity": 0.0,
          "train_class_counts": {
            "0": 58,
            "1": 6942
          },
          "train_positive_prevalence": 0.9917142857142857,
          "weighted_f1": 0.9833144908030507
        }
      },
      "target": "overall_threshold_breach_any",
      "task": "rf_integrated_classification",
      "test_rows": 1500,
      "train_rows": 7000
    },
    "rf_integrated_regression_mean": {
      "metrics": {
        "training_mean_predictor": {
          "mae": 0.20976788727925513,
          "maximum_absolute_error": 0.7060026801175014,
          "median_absolute_error": 0.17763368351886213,
          "prediction_mean": 0.20490641079158942,
          "prediction_median": 0.20490641079158933,
          "prediction_standard_deviation": 8.326672684688674e-17,
          "predictions_above_one": 0,
          "predictions_below_zero": 0,
          "r2": -2.470945139587677e-05,
          "rmse": 0.24879433930955894,
          "target_mean": 0.20366970418470415,
          "target_median": 0.0787878787878787,
          "target_standard_deviation": 0.24879126558070405,
          "train_mean": 0.20490641079158933
        },
        "training_median_predictor": {
          "mae": 0.17574801587301586,
          "maximum_absolute_error": 0.8336363636363636,
          "median_absolute_error": 0.06626262626262625,
          "prediction_mean": 0.07727272727272719,
          "prediction_median": 0.0772727272727272,
          "prediction_standard_deviation": 1.3877787807814457e-17,
          "predictions_above_one": 0,
          "predictions_below_zero": 0,
          "r2": -0.25810898031108476,
          "rmse": 0.27905786067003246,
          "target_mean": 0.20366970418470415,
          "target_median": 0.0787878787878787,
          "target_standard_deviation": 0.24879126558070405,
          "train_median": 0.0772727272727272
        }
      },
      "target": "failure_adjusted_overall_service_fraction_mean",
      "task": "rf_integrated_regression_mean",
      "test_rows": 1500,
      "train_rows": 7000
    },
    "rf_integrated_regression_min": {
      "metrics": {
        "training_mean_predictor": {
          "mae": 0.1808758527966742,
          "maximum_absolute_error": 0.7332126275510205,
          "median_absolute_error": 0.14678737244897958,
          "prediction_mean": 0.1467873724489796,
          "prediction_median": 0.14678737244897958,
          "prediction_standard_deviation": 2.7755575615628914e-17,
          "predictions_above_one": 0,
          "predictions_below_zero": 0,
          "r2": -6.409681968078118e-06,
          "rmse": 0.21970572420038798,
          "target_mean": 0.14623113756613756,
          "target_median": 0.0,
          "target_standard_deviation": 0.21970502008186352,
          "train_mean": 0.14678737244897958
        },
        "training_median_predictor": {
          "mae": 0.14623113756613756,
          "maximum_absolute_error": 0.88,
          "median_absolute_error": 0.0,
          "prediction_mean": 0.0,
          "prediction_median": 0.0,
          "prediction_standard_deviation": 0.0,
          "predictions_above_one": 0,
          "predictions_below_zero": 0,
          "r2": -0.44299595056767016,
          "rmse": 0.26392014217004867,
          "target_mean": 0.14623113756613756,
          "target_median": 0.0,
          "target_standard_deviation": 0.21970502008186352,
          "train_median": 0.0
        }
      },
      "target": "failure_adjusted_overall_service_fraction_min",
      "task": "rf_integrated_regression_min",
      "test_rows": 1500,
      "train_rows": 7000
    },
    "rf_space_classification": {
      "metrics": {
        "majority_class_classifier": {
          "accuracy": 0.8286666666666667,
          "balanced_accuracy": 0.5,
          "confusion_matrix": [
            [
              0,
              257
            ],
            [
              0,
              1243
            ]
          ],
          "f1_by_class": {
            "0": 0.0,
            "1": 0.9063069631790011
          },
          "macro_f1": 0.45315348158950053,
          "negative_class_count": 257,
          "positive_class_count": 1243,
          "pr_auc": 0.8286666666666667,
          "precision_by_class": {
            "0": 0.0,
            "1": 0.8286666666666667
          },
          "recall_by_class": {
            "0": 0.0,
            "1": 1.0
          },
          "roc_auc": null,
          "sensitivity": 1.0,
          "specificity": 0.0,
          "train_class_counts": {
            "0": 1181,
            "1": 5819
          },
          "train_positive_prevalence": 0.8312857142857143,
          "weighted_f1": 0.7510263701543322
        },
        "stratified_random_classifier": {
          "accuracy": 0.708,
          "balanced_accuracy": 0.49046645651445764,
          "confusion_matrix": [
            [
              41,
              216
            ],
            [
              222,
              1021
            ]
          ],
          "f1_by_class": {
            "0": 0.1576923076923077,
            "1": 0.8233870967741935
          },
          "macro_f1": 0.4905397022332506,
          "negative_class_count": 257,
          "positive_class_count": 1243,
          "pr_auc": 0.8259702794826453,
          "precision_by_class": {
            "0": 0.155893536121673,
            "1": 0.8253839935327405
          },
          "recall_by_class": {
            "0": 0.15953307392996108,
            "1": 0.8213998390989542
          },
          "roc_auc": 0.49046645651445764,
          "seed": 42,
          "sensitivity": 0.8213998390989542,
          "specificity": 0.15953307392996108,
          "train_class_counts": {
            "0": 1181,
            "1": 5819
          },
          "train_positive_prevalence": 0.8312857142857143,
          "weighted_f1": 0.7093313895781638
        }
      },
      "target": "space_threshold_breach_any",
      "task": "rf_space_classification",
      "test_rows": 1500,
      "train_rows": 7000
    },
    "rf_space_regression": {
      "metrics": {
        "training_mean_predictor": {
          "mae": 0.2723685158919123,
          "maximum_absolute_error": 0.69727785430839,
          "median_absolute_error": 0.21938881235827667,
          "prediction_mean": 0.30272214569161005,
          "prediction_median": 0.30272214569161,
          "prediction_standard_deviation": 5.551115123125783e-17,
          "predictions_above_one": 0,
          "predictions_below_zero": 0,
          "r2": -4.9667800536834505e-06,
          "rmse": 0.3221212382192703,
          "target_mean": 0.3020042592592592,
          "target_median": 0.125,
          "target_standard_deviation": 0.32212043826957976,
          "train_mean": 0.30272214569161
        },
        "training_median_predictor": {
          "mae": 0.22145830687830687,
          "maximum_absolute_error": 0.8666666666666667,
          "median_absolute_error": 0.06666666666666671,
          "prediction_mean": 0.1333333333333333,
          "prediction_median": 0.1333333333333333,
          "prediction_standard_deviation": 0.0,
          "predictions_above_one": 0,
          "predictions_below_zero": 0,
          "r2": -0.2741851284795438,
          "rmse": 0.3636089355388496,
          "target_mean": 0.3020042592592592,
          "target_median": 0.125,
          "target_standard_deviation": 0.32212043826957976,
          "train_median": 0.1333333333333333
        }
      },
      "target": "space_gcc_fraction_original_min",
      "task": "rf_space_regression",
      "test_rows": 1500,
      "train_rows": 7000
    }
  },
  "bootstrap": {
    "space_classification": {
      "bootstrap_mean_delta": -0.0035944428605818087,
      "bootstrap_seed": 20260812,
      "bootstrap_standard_deviation": 0.018276561477150278,
      "cluster_unit": "design_id",
      "comparison": "rf_space_classification/seed_42 vs tgnn_space_classification/seed_42",
      "designs": 300,
      "observed_delta": -0.0034152342612794317,
      "observed_rf_metric": 0.9397779315137533,
      "observed_tgnn_metric": 0.9431931657750328,
      "paired": true,
      "percentile_2_5": -0.04079517635686995,
      "percentile_97_5": 0.030551570656432322,
      "realizations_per_design": 5,
      "rejected_replicates": 0,
      "statistic": "delta_balanced_accuracy = RF - TGNN",
      "valid_replicates": 2000
    },
    "space_regression": {
      "bootstrap_mean_delta": 0.015600269400416564,
      "bootstrap_seed": 20260812,
      "bootstrap_standard_deviation": 0.002788411111375053,
      "cluster_unit": "design_id",
      "comparison": "rf_space_regression/seed_42 vs tgnn_space_regression/seed_42",
      "designs": 300,
      "observed_delta": 0.015604428646614939,
      "observed_rf_metric": 0.04723111248689984,
      "observed_tgnn_metric": 0.0316266838402849,
      "paired": true,
      "percentile_2_5": 0.01042093875804641,
      "percentile_97_5": 0.02109229034530408,
      "realizations_per_design": 5,
      "rejected_replicates": 0,
      "statistic": "delta_MAE = RF - TGNN",
      "valid_replicates": 2000
    }
  },
  "evaluations_completed": 35,
  "evaluations_expected": 35,
  "failed": 0,
  "heldout_test_driver_sha256": "7edda4e9c4f0ac391e7e0098c3d51a90c443ba1c083dfffbc5e5500e173b3b01",
  "model_artifacts": [
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_42\\final_model.joblib",
      "seed": 42,
      "sha256": "cefea7b3ce8a0f3dc59f436f83ed08933a26ecdb4a091802c1a81302823ca7e6",
      "task": "rf_space_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_123\\final_model.joblib",
      "seed": 123,
      "sha256": "973b7ea709cac3b8ea3948af98f16703cd2e8361fd8e437a9862010a47ce1701",
      "task": "rf_space_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_456\\final_model.joblib",
      "seed": 456,
      "sha256": "6b667f6ecd30cc33331430350b4d5f2fcce3960129b02ed3279f8da0b505ef03",
      "task": "rf_space_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_789\\final_model.joblib",
      "seed": 789,
      "sha256": "0ff4d68b1475af3069f40fcc421c3f475733a72b401040b582943b14df006e22",
      "task": "rf_space_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_2026\\final_model.joblib",
      "seed": 2026,
      "sha256": "7d2994fd3209cd3aa36c3f173b743ada3ccb31b5f4f4c9ba4d247258fc6b7f59",
      "task": "rf_space_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_42\\final_model.joblib",
      "seed": 42,
      "sha256": "bfacf778e6d9623d508571b50621868d0130e60b78ebecf31088d284269c40d8",
      "task": "rf_space_regression"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_123\\final_model.joblib",
      "seed": 123,
      "sha256": "22cb470105ca8f51b413ea58cf95c592fccb205357cd00a62da7897b703de217",
      "task": "rf_space_regression"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_456\\final_model.joblib",
      "seed": 456,
      "sha256": "14d4100886131303a33aae3d6e77443d7a94ffbc723c5f44ef8b53d7d6dd0ef2",
      "task": "rf_space_regression"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_789\\final_model.joblib",
      "seed": 789,
      "sha256": "00c5f250ccd4fa86746002211dd391ea5c6df0117fe42ccbc3c4c07ef2fe4f73",
      "task": "rf_space_regression"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_2026\\final_model.joblib",
      "seed": 2026,
      "sha256": "aaec456c006d330c10d296a3f6a6631b28ad8da356306ea5bff1b5f447459991",
      "task": "rf_space_regression"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_42\\final_model.joblib",
      "seed": 42,
      "sha256": "2fe450bae4dbfbea44e4996d7ad391a195ad88210511176f2d2865e5e668ecf3",
      "task": "rf_integrated_regression_mean"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_123\\final_model.joblib",
      "seed": 123,
      "sha256": "28dc0d2d52777b2932672a2974af349f06a4be0f7f84a6287c9337acda9bf629",
      "task": "rf_integrated_regression_mean"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_456\\final_model.joblib",
      "seed": 456,
      "sha256": "dd3bee1691c52d800508c79a50b918544a33081ecda68803e3d4a31cdcae007a",
      "task": "rf_integrated_regression_mean"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_789\\final_model.joblib",
      "seed": 789,
      "sha256": "bbd378972fa5541d6fbc77f0cc345f17eca97b8bc118ac9416d55581eb82cc04",
      "task": "rf_integrated_regression_mean"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_2026\\final_model.joblib",
      "seed": 2026,
      "sha256": "65361ce5ade303cdd2f20bd68ebac1bcb0d5b166a6d4cb8aac8bdb4e73f2bb2c",
      "task": "rf_integrated_regression_mean"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_42\\final_model.joblib",
      "seed": 42,
      "sha256": "19b933b6f44bceb9cae5a565725923d714e65c6eb469ee3aab4e2a8feac3161d",
      "task": "rf_integrated_regression_min"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_123\\final_model.joblib",
      "seed": 123,
      "sha256": "8071728c33d62b4769fe07a4fed72426b1d14fbf472fb784d4b613584af89a67",
      "task": "rf_integrated_regression_min"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_456\\final_model.joblib",
      "seed": 456,
      "sha256": "5d368f68da529cd3bbce83f4941955bfc7a64a38278cd56f4e54f5c3edd6f38e",
      "task": "rf_integrated_regression_min"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_789\\final_model.joblib",
      "seed": 789,
      "sha256": "1921527c84935e83ae284f4c02acf78652d6d684827cc14fae38f4523292f542",
      "task": "rf_integrated_regression_min"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_2026\\final_model.joblib",
      "seed": 2026,
      "sha256": "493ebcbefcef198ecd4bd033ef0366a9b063ead3779e457e42f8be2764adc0b7",
      "task": "rf_integrated_regression_min"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_42\\final_model.joblib",
      "seed": 42,
      "sha256": "473010f4e75e8bd2f70e46771526acc7ea5599456f91b97a0e39f937740ecacb",
      "task": "rf_integrated_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_123\\final_model.joblib",
      "seed": 123,
      "sha256": "897fda7c3e21904119ce4e3ce0c0474392d170fbef3856da600f3e6478630e94",
      "task": "rf_integrated_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_456\\final_model.joblib",
      "seed": 456,
      "sha256": "1a9d24ce74cd185de8d5f0c6668b916b8f60a2cd3807623990d621801ae466d3",
      "task": "rf_integrated_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_789\\final_model.joblib",
      "seed": 789,
      "sha256": "fa932084a1bc1ed9e01ebce306d4e7cc5c75f5d44e073d5e2e39727ed01441be",
      "task": "rf_integrated_classification"
    },
    {
      "kind": "final_model.joblib",
      "model_family": "RF",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_2026\\final_model.joblib",
      "seed": 2026,
      "sha256": "0bda0ae28129c14f52472e68115af4a4b6d3b8046a8596a9e8bc815745bd2128",
      "task": "rf_integrated_classification"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_42\\best_validation_checkpoint.pt",
      "seed": 42,
      "sha256": "7d5262bf58c05848d6230e044a1ed46ee3d184d04aa1a1fa2c42eb93f787f3e0",
      "task": "tgnn_space_classification"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_123\\best_validation_checkpoint.pt",
      "seed": 123,
      "sha256": "d07d57579676d97ae6092d234e1e9b3290f50fe56befe984b429571dc364206c",
      "task": "tgnn_space_classification"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_456\\best_validation_checkpoint.pt",
      "seed": 456,
      "sha256": "6e8e19c7bdb1109ea4e1435a79cdb5695a662c8662878eb1802719581a4bdcac",
      "task": "tgnn_space_classification"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_789\\best_validation_checkpoint.pt",
      "seed": 789,
      "sha256": "bade1c470a9cb164d070d777886809457a3f516b100e4823169c46e42cc5c734",
      "task": "tgnn_space_classification"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_2026\\best_validation_checkpoint.pt",
      "seed": 2026,
      "sha256": "a3ebff6d93d1c638aa353f139d382fbf68a7d6427214073dba2d4e524fcf79d0",
      "task": "tgnn_space_classification"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_42\\best_validation_checkpoint.pt",
      "seed": 42,
      "sha256": "22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a",
      "task": "tgnn_space_regression"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_123\\best_validation_checkpoint.pt",
      "seed": 123,
      "sha256": "308f4bf179212df640162c8ba45e5b53a2221a6cf3dd8e2fafef5425f41bf8b0",
      "task": "tgnn_space_regression"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_456\\best_validation_checkpoint.pt",
      "seed": 456,
      "sha256": "e42b224adc78f35c9b1fad69aa89139fb72d84044445d2b391d9ca2a257ab199",
      "task": "tgnn_space_regression"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_789\\best_validation_checkpoint.pt",
      "seed": 789,
      "sha256": "e231b66d8f4f4ef120a64e5a326a83611b02c5c8a1c2c91547186a57d0250fb3",
      "task": "tgnn_space_regression"
    },
    {
      "kind": "best_validation_checkpoint.pt",
      "model_family": "TGNN",
      "path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_2026\\best_validation_checkpoint.pt",
      "seed": 2026,
      "sha256": "11834cae573a65a9b8e22838cae3d0b8118d0f27da1fcdd3e804d3a057690a9e",
      "task": "tgnn_space_regression"
    }
  ],
  "no_test_based_selection_or_tuning": true,
  "no_training_performed": true,
  "primary_results": {
    "rf_integrated_classification": {
      "accuracy": 0.9573333333333334,
      "balanced_accuracy": 0.7529153183955064,
      "confusion_matrix": [
        [
          6,
          5
        ],
        [
          59,
          1430
        ]
      ],
      "f1_by_class": {
        "0": 0.15789473684210525,
        "1": 0.9781121751025992
      },
      "macro_f1": 0.5680034559723522,
      "model_sha256": "473010f4e75e8bd2f70e46771526acc7ea5599456f91b97a0e39f937740ecacb",
      "negative_class_count": 11,
      "positive_class_count": 1489,
      "pr_auc": 0.9996689988352089,
      "precision_by_class": {
        "0": 0.09230769230769231,
        "1": 0.9965156794425087
      },
      "recall_by_class": {
        "0": 0.5454545454545454,
        "1": 0.9603760913364674
      },
      "roc_auc": 0.9568349716099884,
      "seed": 42,
      "sensitivity": 0.9603760913364674,
      "specificity": 0.5454545454545454,
      "task": "rf_integrated_classification",
      "weighted_f1": 0.9720972472220222
    },
    "rf_integrated_regression_mean": {
      "mae": 0.04400465550398545,
      "maximum_absolute_error": 0.40590377003695394,
      "median_absolute_error": 0.02271680294248648,
      "model_sha256": "2fe450bae4dbfbea44e4996d7ad391a195ad88210511176f2d2865e5e668ecf3",
      "prediction_mean": 0.20625460787811092,
      "prediction_median": 0.07274362631432647,
      "prediction_standard_deviation": 0.24031445958927145,
      "predictions_above_one": 0,
      "predictions_below_zero": 0,
      "r2": 0.9143246747152756,
      "rmse": 0.07282206841335918,
      "seed": 42,
      "target_mean": 0.20366970418470415,
      "target_median": 0.0787878787878787,
      "target_standard_deviation": 0.24879126558070405,
      "task": "rf_integrated_regression_mean"
    },
    "rf_integrated_regression_min": {
      "mae": 0.04336136180801915,
      "maximum_absolute_error": 0.43908781278530756,
      "median_absolute_error": 0.017274423894251045,
      "model_sha256": "19b933b6f44bceb9cae5a565725923d714e65c6eb469ee3aab4e2a8feac3161d",
      "prediction_mean": 0.14738603945662448,
      "prediction_median": 0.018821848792954718,
      "prediction_standard_deviation": 0.2067349301514923,
      "predictions_above_one": 0,
      "predictions_below_zero": 0,
      "r2": 0.8757446275229711,
      "rmse": 0.07744574610858401,
      "seed": 42,
      "target_mean": 0.14623113756613756,
      "target_median": 0.0,
      "target_standard_deviation": 0.21970502008186352,
      "task": "rf_integrated_regression_min"
    },
    "rf_space_classification": {
      "accuracy": 0.936,
      "balanced_accuracy": 0.9397779315137533,
      "confusion_matrix": [
        [
          243,
          14
        ],
        [
          82,
          1161
        ]
      ],
      "f1_by_class": {
        "0": 0.8350515463917526,
        "1": 0.9602977667493796
      },
      "macro_f1": 0.8976746565705661,
      "model_sha256": "cefea7b3ce8a0f3dc59f436f83ed08933a26ecdb4a091802c1a81302823ca7e6",
      "negative_class_count": 257,
      "positive_class_count": 1243,
      "pr_auc": 0.9961268911176098,
      "precision_by_class": {
        "0": 0.7476923076923077,
        "1": 0.9880851063829788
      },
      "recall_by_class": {
        "0": 0.9455252918287937,
        "1": 0.9340305711987128
      },
      "roc_auc": 0.980629267086345,
      "seed": 42,
      "sensitivity": 0.9340305711987128,
      "specificity": 0.9455252918287937,
      "task": "rf_space_classification",
      "weighted_f1": 0.9388389143281062
    },
    "rf_space_regression": {
      "mae": 0.04723111248689984,
      "maximum_absolute_error": 0.5232577911494485,
      "median_absolute_error": 0.018480806440913156,
      "model_sha256": "bfacf778e6d9623d508571b50621868d0130e60b78ebecf31088d284269c40d8",
      "prediction_mean": 0.3002514496317086,
      "prediction_median": 0.12764772805852387,
      "prediction_standard_deviation": 0.309580312936362,
      "predictions_above_one": 0,
      "predictions_below_zero": 0,
      "r2": 0.9244722630325218,
      "rmse": 0.08852613781352532,
      "seed": 42,
      "target_mean": 0.3020042592592592,
      "target_median": 0.125,
      "target_standard_deviation": 0.32212043826957976,
      "task": "rf_space_regression"
    },
    "tgnn_space_classification": {
      "accuracy": 0.9493333333333334,
      "balanced_accuracy": 0.9431931657750328,
      "confusion_matrix": [
        [
          240,
          17
        ],
        [
          59,
          1184
        ]
      ],
      "f1_by_class": {
        "0": 0.8633093525179856,
        "1": 0.9689034369885434
      },
      "macro_f1": 0.9161063947532645,
      "model_sha256": "7d5262bf58c05848d6230e044a1ed46ee3d184d04aa1a1fa2c42eb93f787f3e0",
      "negative_class_count": 257,
      "positive_class_count": 1243,
      "pr_auc": 0.9977745993746233,
      "precision_by_class": {
        "0": 0.802675585284281,
        "1": 0.9858451290591174
      },
      "recall_by_class": {
        "0": 0.933852140077821,
        "1": 0.9525341914722446
      },
      "roc_auc": 0.9888339682768249,
      "seed": 42,
      "sensitivity": 0.9525341914722446,
      "specificity": 0.933852140077821,
      "task": "tgnn_space_classification",
      "weighted_f1": 0.9508116505159211
    },
    "tgnn_space_regression": {
      "mae": 0.0316266838402849,
      "maximum_absolute_error": 0.39754767417907716,
      "median_absolute_error": 0.017642470342772346,
      "model_sha256": "22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a",
      "prediction_mean": 0.30105758607387545,
      "prediction_median": 0.12312136590480804,
      "prediction_standard_deviation": 0.31762401998589396,
      "predictions_above_one": 2,
      "predictions_below_zero": 0,
      "r2": 0.9719050731794814,
      "rmse": 0.05399235043596987,
      "seed": 42,
      "target_mean": 0.3020042592592593,
      "target_median": 0.125,
      "target_standard_deviation": 0.32212043826957976,
      "task": "tgnn_space_regression"
    }
  },
  "primary_seed": 42,
  "runtime_seconds": 254.0983491000079,
  "schema_version": "satnet.heldout_test_summary.v1",
  "seed_summary": {
    "rf_integrated_classification": {
      "max": 0.7529153183955064,
      "mean": 0.6989742963550889,
      "min": 0.6630136149948105,
      "population_standard_deviation": 0.04404266006774804,
      "primary_metric": "balanced_accuracy",
      "seed_42": 0.7529153183955064,
      "seed_values": {
        "123": 0.7529153183955064,
        "2026": 0.6630136149948105,
        "42": 0.7529153183955064,
        "456": 0.6630136149948105,
        "789": 0.6630136149948105
      }
    },
    "rf_integrated_regression_mean": {
      "max": 0.04403343977618728,
      "mean": 0.043924207188281675,
      "min": 0.043718141534587494,
      "population_standard_deviation": 0.00011180554001390065,
      "primary_metric": "mae",
      "seed_42": 0.04400465550398545,
      "seed_values": {
        "123": 0.04403343977618728,
        "2026": 0.043905000471379345,
        "42": 0.04400465550398545,
        "456": 0.04395979865526877,
        "789": 0.043718141534587494
      }
    },
    "rf_integrated_regression_min": {
      "max": 0.04367786177248616,
      "mean": 0.043457834015578276,
      "min": 0.04330969958667068,
      "population_standard_deviation": 0.00013092020680515053,
      "primary_metric": "mae",
      "seed_42": 0.04336136180801915,
      "seed_values": {
        "123": 0.04367786177248616,
        "2026": 0.04330969958667068,
        "42": 0.04336136180801915,
        "456": 0.043416571926806646,
        "789": 0.04352367498390875
      }
    },
    "rf_space_classification": {
      "max": 0.9491690431396365,
      "mean": 0.9464190126185237,
      "min": 0.9397779315137533,
      "population_standard_deviation": 0.003462278974384028,
      "primary_metric": "balanced_accuracy",
      "seed_42": 0.9397779315137533,
      "seed_values": {
        "123": 0.9471577800664265,
        "2026": 0.9491690431396365,
        "42": 0.9397779315137533,
        "456": 0.9468212652331657,
        "789": 0.9491690431396365
      }
    },
    "rf_space_regression": {
      "max": 0.047355940942319326,
      "mean": 0.04720771688437819,
      "min": 0.04710909447061395,
      "population_standard_deviation": 9.179675725618659e-05,
      "primary_metric": "mae",
      "seed_42": 0.04723111248689984,
      "seed_values": {
        "123": 0.047111114941094004,
        "2026": 0.04710909447061395,
        "42": 0.04723111248689984,
        "456": 0.04723132158096386,
        "789": 0.047355940942319326
      }
    },
    "tgnn_space_classification": {
      "max": 0.960695067475137,
      "mean": 0.944984676836197,
      "min": 0.9343279564001991,
      "population_standard_deviation": 0.00939178857541793,
      "primary_metric": "balanced_accuracy",
      "seed_42": 0.9431931657750328,
      "seed_values": {
        "123": 0.9343279564001991,
        "2026": 0.960695067475137,
        "42": 0.9431931657750328,
        "456": 0.9373487639731914,
        "789": 0.9493584305574251
      }
    },
    "tgnn_space_regression": {
      "max": 0.03228069974669389,
      "mean": 0.03059470766093176,
      "min": 0.02741479726064773,
      "population_standard_deviation": 0.0017104198829715333,
      "primary_metric": "mae",
      "seed_42": 0.0316266838402849,
      "seed_values": {
        "123": 0.031326483735659764,
        "2026": 0.03228069974669389,
        "42": 0.0316266838402849,
        "456": 0.030324873721372516,
        "789": 0.02741479726064773
      }
    }
  },
  "status": "complete",
  "test_accessed": true,
  "test_class_counts_seed_42": {
    "rf_integrated_classification": {
      "negative": 11,
      "positive": 1489
    },
    "rf_space_classification": {
      "negative": 257,
      "positive": 1243
    },
    "tgnn_space_classification": {
      "negative": 257,
      "positive": 1243
    }
  },
  "test_designs": 300,
  "test_rows_per_model": 1500
}

No training, retraining, checkpoint update, model selection, threshold tuning, or TEST-derived refit occurred.
