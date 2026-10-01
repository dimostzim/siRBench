# TabPFN-3.5 benchmark policy

Locked before the first outer-test or HeLa predictions in this experiment. The earlier v2/v3.5 pilot remains a separate, validation-only study.

Report two separate predictors: **TabPFN-3.5 (frozen)** and **TabPFN-3.5 (fine-tuned)**. Both use the corrected dataset, existing five target/guide-grouped folds and five unselected random partitions, with seeds 0, 1 and 2: 30 fits per variant. Grouped evaluation remains primary. Each fit predicts its outer test and the complete 1,047-row HeLa cohort. The aligned 896-row HeLa result is a subset of those saved predictions.

Both variants receive exactly the combined ridge baseline's 176 inputs: 76 guide-position one-hot indicators and the 100 named features from the frozen feature manifest. These are guide/central-duplex descriptors; they do not encode the full 57-nt flanks. Source, cell line, target/group identifiers and labels are excluded from predictors. No HeLa record enters fitting or selection. Validation rows select weights but are never added to the inference context.

Use official TabPFN 9.0.0 at commit `9393b12a46bfc32369a89a53f6c578d22e41faba` and the official September 9, 2026 v3.5 checkpoint, SHA256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`. Frozen weights remain fixed. Fine-tuning updates the released regression model with AdamW, learning rate 1e-5, weight decay 0.01, gradient clipping at 1, released CRPS + MSE loss, 10% linear warmup and cosine decay. Use automatic CUDA mixed precision and activation checkpointing.

Fine-tune for at most 100 epochs, with patience 20, complete validation every epoch and no minimum improvement threshold. Minimize validation MSE, mathematically equivalent to maximizing continuous validation R² on the same fixed rows. The initial foundation weights are eligible, as in the official fine-tuner: if no updated checkpoint improves validation, retain the initial weights and state this explicitly in results. Record all epoch losses, validation values, selected epoch and sampled parameter changes to distinguish optimization from a frozen fit.

Each epoch includes all original training rows in one context/query episode, with an 80/20 split drawn only inside training. Keep the released two-estimator training setting, varying preprocessing seeds across epochs, and use eight estimators for validation and final prediction. Do not subsample the inference context. The smaller training ensemble is a training resource setting, not two input features or two train/test splits.

The official fine-tuner disables GPU preprocessing. Therefore use CPU preprocessing for **both** benchmark variants; retain the released fingerprint feature and other preprocessing defaults. The previous validation-only pilot used default GPU preprocessing and remains separately identified. Predict each complete evaluation cohort in one call in frozen row order, given the previously documented small floating-point/hash sensitivity to query batching. Do not clip, calibrate, ensemble across training seeds, or refit using validation rows.

Save the restored selected weights and a fitted training-context artifact separately. Verify checkpoint and fitted-state reloads against validation predictions before accepting a run. Failed runs restart from foundation weights in a fresh attempt directory because upstream automatic resume does not preserve the full selection history. Preserve failed attempts and log all reruns.

The largest training partition (2,498 rows) passed a two-epoch resource and checkpoint probe on node 4: peak allocated 10.87 GB, peak reserved 12.01 GB, saved-weight reload prediction difference zero. This probe is not part of the benchmark. Production fits are distributed across nodes 4, 5 and 6 with host/GPU recorded; all artifacts are returned to node 4 and verified before temporary worker data are removed.

Frozen seeds describe stochastic preprocessing/ensemble construction; fine-tuned seeds additionally describe gradient optimization. Report both rows regardless of relative performance. Existing six-method predictions and reports remain unchanged; append a separate sealed 60-row run index and apply the existing paired group-bootstrap analysis.

Counter clarification from the implementation audit: the wrapper's
`optimizer_steps` field records the native global-step counter (training-step
attempts). Automatic mixed precision can skip an update on gradient overflow.
The audit therefore records successful optimizer-step counters from each
selected native checkpoint separately. This clarifies reporting and does not
change any training or selection setting.
