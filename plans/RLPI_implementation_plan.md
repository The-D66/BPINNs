# RLPI Implementation Plan

Based on `plans/RLPI.md`, this document outlines the step-by-step implementation plan for the Reinforcement Learning Physics-Informed (RLPI) framework.

## Phase 1: Core Architecture & Components

### 1.1 Policy Network (RL Agent)
- [x] Create `src/networks/PolicyNN.py`.
    - **Goal**: Implement the network that outputs viscosity $\mu(x,t)$.
    - **Input**: State $S = [x, t, u, h, |\partial u/\partial x|, |\partial h/\partial x|]$.
    - **Output**: $\mu$ (scalar).
    - **Constraints**: Use `Softplus` activation * scale to ensure $\mu \ge 0$.
    - **Acceptance Criteria**: Class `PolicyNN` exists, takes state tensor, outputs positive $\mu$.

### 1.2 Equation Adaptation
- [x] Modify `src/equations/SaintVenant.py`.
    - **Goal**: Allow `comp_residual` to accept a dynamic viscosity field $\mu(x,t)$ instead of a constant scalar.
    - **Changes**: Update `comp_residual(inputs, out_sol, out_par, tape)` to optionally accept `mu_field`. Use this field in the momentum equation: $\partial_x (\mu \partial_x (hu))$ or simplified viscous term.
    - **Acceptance Criteria**: `comp_residual` correctly computes residuals with a spatially/temporally varying viscosity.

### 1.3 Coupled Model Container
- [x] Modify `src/networks/BayesNN.py` or create `src/networks/CoupledNN.py`.
    - **Goal**: Manage both the Solver Network (standard PINN) and the new Policy Network.
    - **Logic**:
        - Should contain instances of `CoreNN` (Solver) and `PolicyNN` (Agent).
        - `forward` method needs to coordinate inputs: Solver predicts $(u, h)$, gradients computed, fed to Policy to get $\mu$.
    - **Acceptance Criteria**: Model can perform a forward pass returning solution, gradients, and viscosity.

## Phase 2: Loss & Optimization Logic

### 2.1 Loss Function Update
- [x] Modify `src/networks/LossNN.py`.
    - **Goal**: Incorporate RL-specific losses.
    - **Solver Loss**: Standard PDE residuals (using current $\mu$).
    - **RL Loss**: PDE residuals + L1 Penalty on $\mu$ ($\lambda \|\mu\|_1$).
    - **Acceptance Criteria**: `loss_total` or new method returns appropriate loss components for both phases.

### 2.2 Alternating Optimization Algorithm
- [x] Create `src/algorithms/RLPI.py` (inheriting from `Algorithm`).
    - **Goal**: Implement the specific training loop defined in `plans/RLPI.md`.
    - **Steps**:
        1.  **Warm-up**: Train Solver with fixed $\mu=0.01$ or $\mu=0$.
        2.  **Joint Loop**:
            - **Phase 1 (Solver)**: Fix RL weights, update Solver (5-10 steps).
            - **Phase 2 (RL)**: Fix Solver weights, update RL (1 step). Minimize residual + sparsity.
    - **Acceptance Criteria**: Training loop runs, alternates updates, and respects warm-up epochs.

## Phase 3: Configuration & Integration

### 3.1 Configuration Handling
- [x] Update `src/setup/param.py` and `src/setup/args.py`.
    - **Goal**: Add parameters for RLPI (e.g., `lambda_reg`, `mu_max`, `rl_layers`, `rl_neurons`).
    - **Acceptance Criteria**: Can parse RL-specific arguments from JSON or CLI.

### 3.2 Integration with Main
- [x] Update `src/algorithms/Trainer.py` and `src/main.py`.
    - **Goal**: Recognize `RLPI` as a valid method/algorithm.
    - **Acceptance Criteria**: `python src/main.py --method RLPI` initializes and runs the RLPI algorithm.

## Phase 4: Verification

### 4.1 Unit Testing
- [x] Verify Policy Network output range.
- [x] Verify differentiation through the coupled system (can we compute $d(Loss)/d(RL_{weights})$?).

### 4.2 Experimentation
- [x] Create `config/best_models/RLPI_sv_test.json`.
- [x] Run a test case on SaintVenant1D.
- [x] Verify if $\mu$ concentrates around shocks (non-zero only where needed).
