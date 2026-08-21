# Robustness Provenance Reconciliation

{
  "authoritative_paths": {
    "execution_worktree": "C:\\Users\\johns\\external\\satnet-10k-training-worktree-v1",
    "final_robustness_root": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness",
    "qualified_python": "C:\\Users\\johns\\venvs\\satnet-10k-qualification\\Scripts\\python.exe"
  },
  "frozen_bundle_check": {
    "expected": "12d8db5f7b96f3b2e920911c0c877525153e2a23fab3b3d21321af5b79f3ab33",
    "files": 85,
    "observed": "12d8db5f7b96f3b2e920911c0c877525153e2a23fab3b3d21321af5b79f3ab33"
  },
  "generated_at": "2026-08-20T11:50:56.054789+00:00",
  "identities": {
    "dataset_bundle_sha": "38dacd66432bfa660410ff9cab7f181a53151120e8102dc40df57016f78bbfc3",
    "frozen_robustness_bundle_sha": "12d8db5f7b96f3b2e920911c0c877525153e2a23fab3b3d21321af5b79f3ab33",
    "implementation_sha": "e55dba59e83864d2dd11fa47482bd4ec2bdd797a",
    "production_sha": "d0515088cf3fca06a6aa2d47059269089dcb10a7",
    "rf_selection_sha": "a1b7076197f123ecf62b0b79ddd092bdb66f0d7d8b94d6c113dc201744163911",
    "tgnn_selection_sha": "c85dfdcd3b6a62d741675ac522fda76bd3de7dd4c311a82bc1b12e079dab2bfb",
    "training_plan_bundle_sha": "e14ec5e5b2221aa2c9aa187a2baacc517e676f9dc748612c206eb1dc43788a0a"
  },
  "reconciliation_driver_sha256": "a70bc46be01da0ff11fbd11efaeafbcb04c4f14b7a27888f635c095137ffbff3",
  "retraining_required": false,
  "rf": {
    "driver_note": "The robustness driver constructed each RF estimator using params = {**spec, 'random_state': seed, 'n_jobs': -1}, but serialized configuration: spec. The persisted fitted estimator is authoritative for per-run RF randomness; the manifest configuration is not authoritative for random_state.",
    "mismatches": 0,
    "models_checked": 25,
    "random_state_matches": 25,
    "results": [
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": "sqrt",
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_42\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_42\\final_model.joblib",
        "model_sha256": "cefea7b3ce8a0f3dc59f436f83ed08933a26ecdb4a091802c1a81302823ca7e6",
        "persisted_random_state": 42,
        "seed": 42,
        "task": "rf_space_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": "sqrt",
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_123\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_123\\final_model.joblib",
        "model_sha256": "973b7ea709cac3b8ea3948af98f16703cd2e8361fd8e437a9862010a47ce1701",
        "persisted_random_state": 123,
        "seed": 123,
        "task": "rf_space_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": "sqrt",
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_456\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_456\\final_model.joblib",
        "model_sha256": "6b667f6ecd30cc33331430350b4d5f2fcce3960129b02ed3279f8da0b505ef03",
        "persisted_random_state": 456,
        "seed": 456,
        "task": "rf_space_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": "sqrt",
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_789\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_789\\final_model.joblib",
        "model_sha256": "0ff4d68b1475af3069f40fcc421c3f475733a72b401040b582943b14df006e22",
        "persisted_random_state": 789,
        "seed": 789,
        "task": "rf_space_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": "sqrt",
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_2026\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_classification\\seed_2026\\final_model.joblib",
        "model_sha256": "7d2994fd3209cd3aa36c3f173b743ada3ccb31b5f4f4c9ba4d247258fc6b7f59",
        "persisted_random_state": 2026,
        "seed": 2026,
        "task": "rf_space_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 600,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_42\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_42\\final_model.joblib",
        "model_sha256": "bfacf778e6d9623d508571b50621868d0130e60b78ebecf31088d284269c40d8",
        "persisted_random_state": 42,
        "seed": 42,
        "task": "rf_space_regression"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 600,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_123\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_123\\final_model.joblib",
        "model_sha256": "22cb470105ca8f51b413ea58cf95c592fccb205357cd00a62da7897b703de217",
        "persisted_random_state": 123,
        "seed": 123,
        "task": "rf_space_regression"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 600,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_456\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_456\\final_model.joblib",
        "model_sha256": "14d4100886131303a33aae3d6e77443d7a94ffbc723c5f44ef8b53d7d6dd0ef2",
        "persisted_random_state": 456,
        "seed": 456,
        "task": "rf_space_regression"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 600,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_789\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_789\\final_model.joblib",
        "model_sha256": "00c5f250ccd4fa86746002211dd391ea5c6df0117fe42ccbc3c4c07ef2fe4f73",
        "persisted_random_state": 789,
        "seed": 789,
        "task": "rf_space_regression"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 600,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_2026\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_space_regression\\seed_2026\\final_model.joblib",
        "model_sha256": "aaec456c006d330c10d296a3f6a6631b28ad8da356306ea5bff1b5f447459991",
        "persisted_random_state": 2026,
        "seed": 2026,
        "task": "rf_space_regression"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": null,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_42\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_42\\final_model.joblib",
        "model_sha256": "2fe450bae4dbfbea44e4996d7ad391a195ad88210511176f2d2865e5e668ecf3",
        "persisted_random_state": 42,
        "seed": 42,
        "task": "rf_integrated_regression_mean"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": null,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_123\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_123\\final_model.joblib",
        "model_sha256": "28dc0d2d52777b2932672a2974af349f06a4be0f7f84a6287c9337acda9bf629",
        "persisted_random_state": 123,
        "seed": 123,
        "task": "rf_integrated_regression_mean"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": null,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_456\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_456\\final_model.joblib",
        "model_sha256": "dd3bee1691c52d800508c79a50b918544a33081ecda68803e3d4a31cdcae007a",
        "persisted_random_state": 456,
        "seed": 456,
        "task": "rf_integrated_regression_mean"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": null,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_789\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_789\\final_model.joblib",
        "model_sha256": "bbd378972fa5541d6fbc77f0cc345f17eca97b8bc118ac9416d55581eb82cc04",
        "persisted_random_state": 789,
        "seed": 789,
        "task": "rf_integrated_regression_mean"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": null,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_2026\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_mean\\seed_2026\\final_model.joblib",
        "model_sha256": "65361ce5ade303cdd2f20bd68ebac1bcb0d5b166a6d4cb8aac8bdb4e73f2bb2c",
        "persisted_random_state": 2026,
        "seed": 2026,
        "task": "rf_integrated_regression_mean"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_42\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_42\\final_model.joblib",
        "model_sha256": "19b933b6f44bceb9cae5a565725923d714e65c6eb469ee3aab4e2a8feac3161d",
        "persisted_random_state": 42,
        "seed": 42,
        "task": "rf_integrated_regression_min"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_123\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_123\\final_model.joblib",
        "model_sha256": "8071728c33d62b4769fe07a4fed72426b1d14fbf472fb784d4b613584af89a67",
        "persisted_random_state": 123,
        "seed": 123,
        "task": "rf_integrated_regression_min"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_456\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_456\\final_model.joblib",
        "model_sha256": "5d368f68da529cd3bbce83f4941955bfc7a64a38278cd56f4e54f5c3edd6f38e",
        "persisted_random_state": 456,
        "seed": 456,
        "task": "rf_integrated_regression_min"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_789\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_789\\final_model.joblib",
        "model_sha256": "1921527c84935e83ae284f4c02acf78652d6d684827cc14fae38f4523292f542",
        "persisted_random_state": 789,
        "seed": 789,
        "task": "rf_integrated_regression_min"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "max_depth": 20,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_2026\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_regression_min\\seed_2026\\final_model.joblib",
        "model_sha256": "493ebcbefcef198ecd4bd033ef0366a9b063ead3779e457e42f8be2764adc0b7",
        "persisted_random_state": 2026,
        "seed": 2026,
        "task": "rf_integrated_regression_min"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_42\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_42\\final_model.joblib",
        "model_sha256": "473010f4e75e8bd2f70e46771526acc7ea5599456f91b97a0e39f937740ecacb",
        "persisted_random_state": 42,
        "seed": 42,
        "task": "rf_integrated_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_123\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_123\\final_model.joblib",
        "model_sha256": "897fda7c3e21904119ce4e3ce0c0474392d170fbef3856da600f3e6478630e94",
        "persisted_random_state": 123,
        "seed": 123,
        "task": "rf_integrated_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_456\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_456\\final_model.joblib",
        "model_sha256": "1a9d24ce74cd185de8d5f0c6668b916b8f60a2cd3807623990d621801ae466d3",
        "persisted_random_state": 456,
        "seed": 456,
        "task": "rf_integrated_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_789\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_789\\final_model.joblib",
        "model_sha256": "fa932084a1bc1ed9e01ebce306d4e7cc5c75f5d44e073d5e2e39727ed01441be",
        "persisted_random_state": 789,
        "seed": 789,
        "task": "rf_integrated_classification"
      },
      {
        "checks": {
          "manifest_seed_matches": true,
          "model_sha256_matches": true,
          "persisted_random_state_matches": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_configuration": {
          "bootstrap": true,
          "class_weight": "balanced",
          "max_depth": 10,
          "max_features": 1.0,
          "min_samples_leaf": 5,
          "n_estimators": 300,
          "n_jobs": -1,
          "random_state": 42
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_2026\\final_manifest.json",
        "model_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\rf_integrated_classification\\seed_2026\\final_model.joblib",
        "model_sha256": "0bda0ae28129c14f52472e68115af4a4b6d3b8046a8596a9e8bc815745bd2128",
        "persisted_random_state": 2026,
        "seed": 2026,
        "task": "rf_integrated_classification"
      }
    ]
  },
  "robustness_bundle_modified": false,
  "schema_version": "satnet.robustness_provenance_reconciliation.v1",
  "scientific_training_invalidated": false,
  "test_accessed": false,
  "tgnn": {
    "checkpoint_selected_epoch_matches": 10,
    "driver_note": "The robustness driver serialized epochs_run and selected_epoch as best_epoch and set stopped_early from best_epoch < max_epochs. selected_epoch/checkpoint epoch are authoritative; actual_epochs_run and actual_stopped_early are reconstructed from progress.jsonl.",
    "mismatches": 0,
    "progress_logs_valid": 10,
    "results": [
      {
        "actual_epochs_run": 24,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_42\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "7d5262bf58c05848d6230e044a1ed46ee3d184d04aa1a1fa2c42eb93f787f3e0",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_42\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_42\\progress.jsonl",
        "progress_record_count": 24,
        "reported_epochs_run": 14,
        "reported_stopped_early": true,
        "seed": 42,
        "selected_epoch": 14,
        "task": "tgnn_space_classification"
      },
      {
        "actual_epochs_run": 51,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_123\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "d07d57579676d97ae6092d234e1e9b3290f50fe56befe984b429571dc364206c",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_123\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_123\\progress.jsonl",
        "progress_record_count": 51,
        "reported_epochs_run": 41,
        "reported_stopped_early": true,
        "seed": 123,
        "selected_epoch": 41,
        "task": "tgnn_space_classification"
      },
      {
        "actual_epochs_run": 28,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_456\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "6e8e19c7bdb1109ea4e1435a79cdb5695a662c8662878eb1802719581a4bdcac",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_456\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_456\\progress.jsonl",
        "progress_record_count": 28,
        "reported_epochs_run": 18,
        "reported_stopped_early": true,
        "seed": 456,
        "selected_epoch": 18,
        "task": "tgnn_space_classification"
      },
      {
        "actual_epochs_run": 32,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_789\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "bade1c470a9cb164d070d777886809457a3f516b100e4823169c46e42cc5c734",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_789\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_789\\progress.jsonl",
        "progress_record_count": 32,
        "reported_epochs_run": 22,
        "reported_stopped_early": true,
        "seed": 789,
        "selected_epoch": 22,
        "task": "tgnn_space_classification"
      },
      {
        "actual_epochs_run": 32,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_2026\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "a3ebff6d93d1c638aa353f139d382fbf68a7d6427214073dba2d4e524fcf79d0",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_2026\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_classification\\seed_2026\\progress.jsonl",
        "progress_record_count": 32,
        "reported_epochs_run": 22,
        "reported_stopped_early": true,
        "seed": 2026,
        "selected_epoch": 22,
        "task": "tgnn_space_classification"
      },
      {
        "actual_epochs_run": 38,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_42\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_42\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_42\\progress.jsonl",
        "progress_record_count": 38,
        "reported_epochs_run": 28,
        "reported_stopped_early": true,
        "seed": 42,
        "selected_epoch": 28,
        "task": "tgnn_space_regression"
      },
      {
        "actual_epochs_run": 44,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_123\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "308f4bf179212df640162c8ba45e5b53a2221a6cf3dd8e2fafef5425f41bf8b0",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_123\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_123\\progress.jsonl",
        "progress_record_count": 44,
        "reported_epochs_run": 34,
        "reported_stopped_early": true,
        "seed": 123,
        "selected_epoch": 34,
        "task": "tgnn_space_regression"
      },
      {
        "actual_epochs_run": 62,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_456\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "e42b224adc78f35c9b1fad69aa89139fb72d84044445d2b391d9ca2a257ab199",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_456\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_456\\progress.jsonl",
        "progress_record_count": 62,
        "reported_epochs_run": 52,
        "reported_stopped_early": true,
        "seed": 456,
        "selected_epoch": 52,
        "task": "tgnn_space_regression"
      },
      {
        "actual_epochs_run": 40,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_789\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "e231b66d8f4f4ef120a64e5a326a83611b02c5c8a1c2c91547186a57d0250fb3",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_789\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_789\\progress.jsonl",
        "progress_record_count": 40,
        "reported_epochs_run": 30,
        "reported_stopped_early": true,
        "seed": 789,
        "selected_epoch": 30,
        "task": "tgnn_space_regression"
      },
      {
        "actual_epochs_run": 40,
        "actual_stopped_early": true,
        "checkpoint_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_2026\\best_validation_checkpoint.pt",
        "checkpoint_sha256": "11834cae573a65a9b8e22838cae3d0b8118d0f27da1fcdd3e804d3a057690a9e",
        "checks": {
          "checkpoint_selected_epoch_matches_manifest": true,
          "checkpoint_sha256_matches": true,
          "manifest_seed_matches": true,
          "progress_valid": true,
          "selected_epoch_within_actual_run": true,
          "status_completed": true,
          "test_accessed_false": true
        },
        "manifest_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_2026\\final_manifest.json",
        "max_epochs": 100,
        "progress_path": "C:\\Users\\johns\\external\\satnet-10k-model-training-v1\\final_robustness\\tgnn_space_regression\\seed_2026\\progress.jsonl",
        "progress_record_count": 40,
        "reported_epochs_run": 30,
        "reported_stopped_early": true,
        "seed": 2026,
        "selected_epoch": 30,
        "task": "tgnn_space_regression"
      }
    ],
    "runs_checked": 10
  }
}

## TGNN Actual Epochs

| task | seed | selected_epoch | reported_epochs_run | actual_epochs_run | max_epochs | reported_stopped_early | actual_stopped_early | checkpoint_sha256 |
|---|---:|---:|---:|---:|---:|---|---|---|
| tgnn_space_classification | 42 | 14 | 14 | 24 | 100 | True | True | 7d5262bf58c05848d6230e044a1ed46ee3d184d04aa1a1fa2c42eb93f787f3e0 |
| tgnn_space_classification | 123 | 41 | 41 | 51 | 100 | True | True | d07d57579676d97ae6092d234e1e9b3290f50fe56befe984b429571dc364206c |
| tgnn_space_classification | 456 | 18 | 18 | 28 | 100 | True | True | 6e8e19c7bdb1109ea4e1435a79cdb5695a662c8662878eb1802719581a4bdcac |
| tgnn_space_classification | 789 | 22 | 22 | 32 | 100 | True | True | bade1c470a9cb164d070d777886809457a3f516b100e4823169c46e42cc5c734 |
| tgnn_space_classification | 2026 | 22 | 22 | 32 | 100 | True | True | a3ebff6d93d1c638aa353f139d382fbf68a7d6427214073dba2d4e524fcf79d0 |
| tgnn_space_regression | 42 | 28 | 28 | 38 | 100 | True | True | 22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a |
| tgnn_space_regression | 123 | 34 | 34 | 44 | 100 | True | True | 308f4bf179212df640162c8ba45e5b53a2221a6cf3dd8e2fafef5425f41bf8b0 |
| tgnn_space_regression | 456 | 52 | 52 | 62 | 100 | True | True | e42b224adc78f35c9b1fad69aa89139fb72d84044445d2b391d9ca2a257ab199 |
| tgnn_space_regression | 789 | 30 | 30 | 40 | 100 | True | True | e231b66d8f4f4ef120a64e5a326a83611b02c5c8a1c2c91547186a57d0250fb3 |
| tgnn_space_regression | 2026 | 30 | 30 | 40 | 100 | True | True | 11834cae573a65a9b8e22838cae3d0b8118d0f27da1fcdd3e804d3a057690a9e |
