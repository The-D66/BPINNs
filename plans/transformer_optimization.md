# Transformer Rolling Prediction Optimization Plan

## Objective
Improve the long-term stability, accuracy, and physical consistency of the `ConvFormer` auto-regressive model for solving Saint-Venant equations. This plan addresses the issues identified in the code review, specifically regarding numerical dissipation, exposure bias, and spatial coupling.

## Phase 1: Physics & Loss Function Improvements
**Goal**: Reduce numerical dissipation caused by the implicit Backward Euler scheme in the PDE loss.

*   **Target File**: `src/algorithms/Seq2SeqTrainer.py`
*   **Changes**:
    *   Refactor `compute_pde_residual` method.
    *   Implement **Crank-Nicolson (Trapezoidal)** time-stepping scheme for the residual calculation.
    *   Instead of calculating spatial derivatives ($h_x, u_x$) only at $t+1$, calculate them for both the previous step ($t$, from input) and the predicted step ($t+1$).
    *   Use the average of these spatial derivatives in the momentum and continuity equations.
    *   Formula: $\text{Res} \approx \frac{U^{n+1} - U^n}{\Delta t} + \frac{1}{2} [F(U^{n+1})_x + F(U^n)_x]$

## Phase 2: Training Strategy Enhancements (Multi-step Loss)
**Goal**: Mitigate "Exposure Bias" (where the model relies too much on perfect ground truth history) and train the model to correct or tolerate its own errors.

*   **Target File**: `src/algorithms/Seq2SeqTrainer.py`
*   **Changes**:
    *   Modify `train_step` to perform an autoregressive loop for $k$ steps (e.g., start with $k=2$, gradually increase or fix at small number).
    *   **Step 1**: Predict $U_{t+1}$ using ground truth history. Calculate Loss.
    *   **Step 2**: Append predicted $U_{t+1}$ to history (sliding window). Predict $U_{t+2}$. Calculate Loss.
    *   Accumulate gradients from all steps.
    *   Update the training loop in `train()` to support this multi-step logic.

## Phase 3: Architecture Refinement (Spatial Coupling)
**Goal**: Increase the receptive field of the spatial encoder to ensure boundary information propagates effectively to the domain center during rollout.

*   **Target File**: `src/networks/ConvFormer.py`
*   **Changes**:
    *   Increase `kernel_size` in `SpatialEncoderBlock` (e.g., from 5 to 7 or 9) to widen the receptive field.
    *   (Optional) Add Dilated Convolutions if simply increasing kernel size is too expensive.
    *   Verify `PositionalEncoding` is correctly applied and scaled for the temporal dimension.

## Phase 4: Validation & Metrics
**Goal**: Use a metric that actually reflects deployment performance (long-term stability) rather than just single-step prediction accuracy.

*   **Target Files**: 
    *   `src/algorithms/Seq2SeqTrainer.py`
    *   `src/verify_rollout.py` (Utilize existing logic)
*   **Changes**:
    *   In `Seq2SeqTrainer.train()`, ensure that the validation step calls `validate_model` (from `verify_rollout.py`) which performs a full rollout (e.g., 100+ steps).
    *   Use this **Rollout Error** as the criterion for saving `model_best.weights.h5`, instead of the single-step validation loss.

## Phase 5: Data Pipeline Safety
**Goal**: Prevent data leakage where testing statistics might pollute normalization parameters.

*   **Target File**: `src/setup/window_loader.py`
*   **Changes**:
    *   Modify `__compute_stats` and `normalize` methods.
    *   Ensure mean and std are calculated **only** using the indices corresponding to the training set.
    *   Pass a `mode` or explicit indices to `__compute_stats` to exclude test data from normalization statistics.
