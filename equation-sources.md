# Equation sources: Code ↔ main.pdf (main.tex)

Where the **r_eff** (and related) equations live in code and where they are derived/stated in the project PDF (main.tex → main.pdf).

---

## 1. Shale

### Code

| What | File | Lines |
|------|------|--------|
| **r_eff** formula | `Traffic_Benchmarks.py` | **331–341** (`_apply_architecture_model`, `architecture_type == "shale"`) |
| Default **h** | `Traffic_Benchmarks.py` | **169–170** (`ArchitectureParams.shale_h`, `shale_epoch`) |

**Equation in code:**  
`effective_rate = rate / (h + 1)`  
Latency: `alpha * h + beta` (α=1.2, β=2.5).  
Bandwidth tax: `bw_tax = h / (h + 1)`.

### main.pdf / main.tex

| Equation / concept | Section in main.tex | Location (approx.) |
|-------------------|--------------------|--------------------|
| **Throughput floor 1/(h+1)** | **§ Shale → VLB Spray Mechanism** | Quote: *“worst-case throughput floor of $\frac{1}{h+1}$ of the raw link capacity”* (around line 274). |
| **Path length L(h) = h+1** | Same subsection + **Proposed VLB Theorem** | “total path length of $L(h) = h + 1$ hops”; Theorem: “$L(h) = h + 1$” (lines 271, 336). |
| **Capacity model (no explicit formula in text)** | **§ Shale → Capacity Model** | “implements the VLB penalty directly in its capacity calculations” (line 295). |
| **Throughput guarantee Th ∝ 1/(h+1)** | **§ Shale → Numerical Components → Throughput & Bandwidth Functions** | “throughput guarantee ($Th$) scales inversely with … ($h_{spray} + 1$)”; “$Th \approx 0.33$” for h=2, “$0.20$” for h=4 (lines 401–405). |
| **Simulation match to 1/(h+1)** | **§ Shale → Simulation Findings** | “matching the $1/(4+1)$ capacity model”; “throughput capacity continues to follow the $1/(h+1)$ trend” (line 441). |
| **Latency τ(h) = αh + β** | **§ Shale → VLB Saturation Finding** | Eq: $\tau(h) = \alpha \cdot h + \beta \approx 1.2h + 2.5$ (lines 356–357). |
| **Theoretical capacity 1/(h+1)** | **§ Shale → Table (Benchmark Efficiency Summary)** | “Theoretical Capacity & 0.33 & $1/(h+1)$” (line 346). |

**Summary:** The **r_eff = rate / (h+1)** in code comes from the **Shale** section of main.tex: **VLB Spray Mechanism** (throughput floor 1/(h+1)), **Throughput & Bandwidth Functions** (Th ∝ 1/(h+1)), **Simulation Findings** (1/(h+1) trend), and the **Capacity Model** paragraph (VLB penalty in capacity calculations).

---

## 2. Opera

### Code

| What | File | Lines |
|------|------|--------|
| **r_eff** formula | `Traffic_Benchmarks.py` | **342–355** (`_apply_architecture_model`, `architecture_type == "opera"`) |
| Default **α, δ, T_cycle** | `Traffic_Benchmarks.py` | **173–175** (`ArchitectureParams.opera_alpha`, `opera_delta`, `opera_t_cycle`) |

**Equations in code:**  
`reconfig_eff = 1 - (delta / t_cycle)`  
`bulk_rate = alpha_split * rate * reconfig_eff`  
`short_rate = (1 - alpha_split) * rate / max(1, hops)`  
`effective_rate = bulk_rate + short_rate`  
Latency: `hops * delta`.  
Bandwidth tax: `(1 - alpha_split) * (hops - 1) / max(1, hops)`.

### main.pdf / main.tex

| Equation / concept | Section in main.tex | Location (approx.) |
|-------------------|--------------------|--------------------|
| **92/8 hybrid, duty cycle** | **§ Opera → Numerical Components & Bottleneck Determination** | “Cycle Time ($T_{cycle}$) and Reconfiguration Delay ($\delta$)”; “92/8 hybrid split” (lines 542–543, 574). |
| **Bandwidth tax, L hops** | **§ Opera → Throughput & Bandwidth Functions** | $B(S) = S \cdot L$, Tax $= L - 1$; “$L=1$ for bulk … $L \approx 2-4$ for latency-sensitive” (lines 546–550). |
| **Duty cycle loss δ/T_cycle** | **§ Opera → Proposed Opera Theorem** | $\mathcal{P} \ge (1-\alpha)(L_{\text{exp}}-1) + \delta/T_{\text{cycle}}$ (lines 577–580). |
| **Effective throughput (bulk & expander)** | **§ Opera → Table (Benchmark Efficiency Summary)** | “Effective Throughput (Bulk) & 0.828 & $\alpha \times (1 - \delta/T_{\text{cycle}})$”; “(Expander) & 0.033 & $(1-\alpha) \times (1 - \delta/T_{\text{cycle}}) / \bar{\ell}$” (lines 590–591). |

**Summary:** The **r_eff = α·rate·(1−δ/T_cycle) + (1−α)·rate/hops** in code is the implementation of the **Opera** section: **Numerical Components & Bottleneck Determination**, **Throughput & Bandwidth Functions** (L=1 vs L≈2–4), **Proposed Opera Theorem** (duty cycle loss), and the **Opera Benchmark Efficiency Summary** table (bulk and expander effective throughput formulas).

---

## 3. Sirius

### Code

| What | File | Lines |
|------|------|--------|
| **r_eff** (hop-based) | `Traffic_Benchmarks.py` | **357–375** (`_apply_architecture_model`, `architecture_type == "sirius"`) |
| **Load penalty** (ρ > 0.85) | `Traffic_Benchmarks.py` | **295–301** (in `calculate_topology_capacity`, after the per-flow loop) |
| Default **η, T_slot, δ** | `Traffic_Benchmarks.py` | **177–180** (`ArchitectureParams.sirius_eta`, `sirius_t_slot`, `sirius_delta`) |

**Equations in code:**  
- hops ≤ 1: `effective_rate = rate * eta`  
- hops ≤ 2: `effective_rate = (rate * eta) / 2`  
- hops ≥ 3: `effective_rate = (rate * eta) / (hops ** 2)`  
- If load > 0.85: `penalty = exp(-10 * (load - 0.85))`, `total_capacity *= penalty`.

### main.pdf / main.tex

| Equation / concept | Section in main.tex | Location (approx.) |
|-------------------|--------------------|--------------------|
| **η, δ, T_slot** | **§ Sirius → Architectural Components & Numerical Parameters → Timing Parameters** | “efficiency … bounded by … reconfiguration time ($\delta$) … timeslot duration ($T_{slot}$)”; δ=3.84 ns, T_slot=100 ns (lines 721–726). |
| **Hop-based penalty C_hops(ℓ)** | **§ Sirius → Numerical Results & Capacity Penalties** | $C_{\text{hops}}(\ell)$: $\eta$ for $\ell\le 1$, $\eta/2$ for $1<\ell\le 2$, $\eta/\ell^2$ for $\ell>2$ (lines 744–750). |
| **Load penalty C_load(ρ)** | Same subsection | $C_{load}(\rho) = \exp(-10(\rho - 0.85))$ for $\rho > 0.85$ (lines 751–754). |
| **“50% bandwidth tax” on 2-hop** | Same subsection + Table | “2-hop) consumes twice the logical bandwidth”; “50\% bandwidth tax on indirect paths”; Table: “Sprayed Path Capacity & 0.5 η” (lines 737, 776, 786). |
| **η = (T_slot − δ)/T_slot** | **§ Sirius → Table (Benchmark Efficiency Summary)** | “$\eta = (T_{slot}-\delta)/T_{slot}$ (\texttt{sirius\_t\_slot}, \texttt{sirius\_delta})” (line 784). |
| **Saturation load 0.85** | Same subsection + Table | “maintains high utilization up to a load of 0.85”; “Saturation Load & 0.85” (lines 776, 787). |

**Summary:** The **r_eff** and load penalty in code implement **§ Sirius → Numerical Results & Capacity Penalties**: hop-based penalty **C_hops(ℓ)** (η, η/2, η/ℓ²) and high-load penalty **C_load(ρ)** = exp(-10(ρ−0.85)), with parameters and η definition from **Architectural Components & Numerical Parameters** and the **Sirius Benchmark Efficiency Summary** table.

---

## Quick reference: code only

| Algorithm | r_eff in code (file: line range) | Extra (e.g. load penalty) |
|-----------|----------------------------------|---------------------------|
| **Shale** | `Traffic_Benchmarks.py`: **331–341** | — |
| **Opera** | `Traffic_Benchmarks.py`: **342–355** | — |
| **Sirius** | `Traffic_Benchmarks.py`: **357–375** | **295–301** (load > 0.85) |

All three use the same caller: `calculate_topology_capacity` → per-flow `_apply_architecture_model(rate, hops, architecture_type, params)`; `rate` is the waterfilling-derived rate (demand-scaled) computed earlier in the same file.
