# SMT-LIB v2 Feasibility Analysis for mHEALTH and HMC Datasets

## 1. Overview

This report analyzes whether the **mHEALTH** activity recognition dataset and the **HMC Sleep Staging** dataset can support formal analysis using **SMT-LIB v2** (Satisfiability Modulo Theories Library, version 2). SMT solvers (e.g., Z3, CVC5) accept SMT-LIB v2 as their input language and can verify logical properties over mathematical theories including linear arithmetic, bit-vectors, and arrays.

Typical SMT-based analyses applicable to biosignal datasets include:
- **Temporal property verification** — checking whether label transition sequences satisfy domain-defined logical rules
- **Formal classifier verification** — proving robustness or decision-boundary properties of learned models
- **Constraint-based anomaly detection** — encoding expected physiological behavior as SMT constraints and identifying violations

---

## 2. Dataset Overview

### 2.1 mHEALTH Dataset

| Property | Value |
|----------|-------|
| File format | Tab-separated `.log`, one file per subject |
| Subjects | 10 (`mHealth_subject1.log` – `mHealth_subject10.log`) |
| Sampling rate | 50 Hz |
| Signal channels | 13 (ECG; left-ankle accelerometer x/y/z; left-ankle gyroscope x/y/z; right-arm accelerometer x/y/z; right-arm gyroscope x/y/z) |
| Label type | Discrete activity class (0 = null, 1–12 = 12 activities) |
| Window size (loader) | 100 samples = 2 s |
| Total windows (approx.) | ~3,000 per subject after null removal |

**Modality structure:**

```
Chest:       ECG (1 ch)
Left ankle:  Accelerometer (3 ch) + Gyroscope (3 ch)
Right arm:   Accelerometer (3 ch) + Gyroscope (3 ch)
```

Activity labels cover standing, sitting, lying, walking, climbing stairs, waist bend, arm elevation, knee bending, cycling, jogging, running, and jumping — forming a finite set of discrete states with physically meaningful transition constraints.

---

### 2.2 HMC Sleep Staging Dataset

| Property | Value |
|----------|-------|
| File format | EDF (European Data Format), two files per subject |
| Subjects | ~151 (`SN001.edf` + `SN001_sleepscoring.edf` per subject) |
| Sampling rate | 256 Hz (raw), resampled to 100 Hz in loader |
| Signal channels | 8 (EEG F4-M1, EEG C4-M1, EEG O2-M1, EEG C3-M2, EMG chin, EOG E1-M2, EOG E2-M2, ECG) |
| Label type | Sleep stage (W=0, N1=1, N2=2, N3=3, REM=4) |
| Epoch size (loader) | 3,000 samples = 30 s (non-overlapping) |
| Total epochs (approx.) | ~800–1,200 per subject (full night) |

**Modality structure:**

```
EEG:  F4-M1, C4-M1, O2-M1, C3-M2  (4 ch, frontal / central / occipital)
EOG:  E1-M2, E2-M2                  (2 ch, eye movement)
EMG:  chin                           (1 ch, muscle tone)
ECG:  lead II                        (1 ch, cardiac)
```

Sleep stages follow **AASM (American Academy of Sleep Medicine)** rules, which are explicit logical constraints on permissible stage sequences — making this dataset inherently well-suited for formal rule encoding.

---

## 3. SMT-LIB v2 Compatibility Analysis

### 3.1 Why Raw Signals Cannot Be Directly Encoded

Both datasets contain continuous real-valued time-series signals sampled at high frequency. Direct encoding of raw signals into SMT-LIB v2 faces three fundamental obstacles:

| Issue | Detail |
|-------|--------|
| **Scale** | A 30-minute mHEALTH segment at 50 Hz × 13 channels = 1,170,000 variables; a full-night HMC recording exceeds 14 million samples |
| **Numeric theory mismatch** | Floating-point theory (QF_FP) is decidable but practically intractable at this scale; rational arithmetic (QF_LRA) requires quantization of all values |
| **Semantic gap** | SMT excels at logical / combinatorial reasoning, not statistical pattern matching over noisy sensor data |

### 3.2 mHEALTH Compatibility

**Strengths:**

- Labels are discrete integers (12 classes), mapping directly to SMT integer or enumeration variables.
- Activity transitions form a finite-state machine; temporal properties such as *"Running always follows Jogging rather than Standing"* can be expressed in Linear Temporal Logic (LTL) and encoded as SMT assertions.
- Window-level feature vectors (mean, variance, frequency-band energy per channel) are compact enough (~13 × 5 = 65 scalar features per window) to encode as bounded real variables.
- With ~3,000 windows per subject, bounded model checking over a full subject recording is feasible.

**Limitations:**

- Activity segments are separated by long null periods (Activity = 0), which must be handled explicitly in the state encoding.
- No established domain rule set exists (unlike AASM for sleep), so temporal constraints must be designed manually or mined from data.

**Feasible SMT analyses:**

1. Verify that a trained linear/tree-based classifier satisfies monotonicity constraints (e.g., higher accelerometer magnitude → higher predicted activity intensity).
2. Check whether activity label sequences in the dataset violate physically implausible transitions (e.g., direct jump from Lying to Running without intermediate states).
3. Generate adversarial feature perturbations that flip the classifier's output, bounded by a physiologically meaningful ε-ball.

---

### 3.3 HMC Compatibility

**Strengths:**

- The **30-second epoch** is a natural discrete abstraction unit — each epoch maps to a single label, reducing a full-night recording from ~2.5 M samples to ~1,000 labeled tokens.
- **AASM sleep staging rules** provide a rich, well-documented set of formal constraints that translate almost directly into SMT assertions, for example:

  ```
  ; REM cannot follow Wake without prior NREM
  (assert (forall ((i Int))
    (=> (= stage[i] REM)
        (exists ((j Int)) (and (< j i) (>= stage[j] N1))))))
  ```

- The 5-class label space (W, N1, N2, N3, REM) is small and well-defined, enabling exhaustive state enumeration.
- Per-epoch EEG features (delta/theta/alpha/sigma/beta band power) are standard and well-characterized, suitable as quantized SMT variables.

**Limitations:**

- Full-night recordings with ~1,000 epochs per subject make unbounded quantifier encodings expensive; bounded model checking over a sliding window (e.g., 10 epochs) is recommended.
- Inter-subject variability in sleep architecture complicates the definition of universal constraints; subject-specific constraints may be needed.

**Feasible SMT analyses:**

1. Formal verification that an automated sleep scorer's output never violates AASM transition rules (e.g., W→N3 without N1/N2 is forbidden).
2. Counterexample generation: find the minimal feature perturbation to an N2 epoch that causes a classifier to predict REM.
3. Property-guided data validation: automatically flag epochs in the dataset whose predicted label sequence is logically inconsistent with known physiological constraints.

---

## 4. Required Preprocessing Pipeline

Neither dataset can be fed into an SMT solver without a preprocessing stage. The recommended pipeline is:

```
Raw Signals (float, high-frequency)
        │
        ▼
  Feature Extraction
  (per window / per epoch)
  e.g., band power, RMS, zero-crossing rate
        │
        ▼
  Quantization / Discretization
  (map floats → bounded integers or fixed-precision rationals)
        │
        ▼
  SMT Variable Declaration
  (QF_LIA for integers, QF_LRA for rationals)
        │
        ▼
  Constraint Encoding (.smt2)
  (temporal rules, classifier encoding, domain constraints)
        │
        ▼
  SMT Solver (Z3 / CVC5)
        │
        ▼
  SAT / UNSAT + Model / Counterexample
```

---

## 5. Recommended SMT Theories

| Theory | Use case |
|--------|----------|
| `QF_LIA` (Quantifier-Free Linear Integer Arithmetic) | Label sequence constraints, transition rules, epoch indices |
| `QF_LRA` (Quantifier-Free Linear Real Arithmetic) | Quantized feature constraints, linear classifier verification |
| `QF_BV` (Bit-Vectors) | Fixed-precision signal encoding if bitwidth is fixed |
| `LIA` with quantifiers | Universal/existential temporal properties over full sequences |

For most practical verification tasks on these datasets, **QF_LIA + QF_LRA** with bounded model checking (fixed horizon of N epochs/windows) offers the best balance of expressiveness and solver performance.

---

## 6. Comparative Summary

| Criterion | mHEALTH | HMC |
|-----------|---------|-----|
| Label discreteness | ✅ 12 integer classes | ✅ 5 integer classes |
| Natural discrete unit | ⚠️ Sliding window (manual) | ✅ 30 s epoch (built-in) |
| Domain rule set | ❌ None established | ✅ AASM rules |
| Signal scale (per subject) | ⚠️ ~160K samples | ❌ ~2.5M samples (raw) |
| Epoch-level scale | ✅ ~3K windows | ✅ ~1K epochs |
| Transition structure | ⚠️ Implicit | ✅ Explicit (AASM) |
| Overall SMT readiness | **Medium** | **High** (at epoch level) |

---

## 7. Conclusion

Both datasets are compatible with SMT-LIB v2 analysis **at the abstracted feature/epoch level**, but neither is directly usable in raw signal form due to scale and numeric representation constraints.

- **HMC** is the stronger candidate owing to its built-in 30-second epoch structure and the availability of the AASM rule set, which provides a ready-made formal specification that can be encoded almost verbatim as SMT assertions.
- **mHEALTH** is suitable for classifier verification and adversarial perturbation analysis, but requires manual definition of temporal transition constraints and a window-level feature extraction step.

The most productive near-term application of SMT-LIB v2 to these datasets is **formal post-hoc verification of automated classifiers**: given a trained model and a test recording, verify whether the model's predicted label sequence is logically consistent with domain rules, and if not, produce a minimal counterexample.
