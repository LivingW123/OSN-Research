"""
Figure Generation for OSN-Research Paper
=========================================
Regenerates all paper figures using the updated congestion-based r_eff model.

Run:  python generate_figures.py
"""

import sys, io, os, random
# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Suppress module-level prints during imports ──────────────────────
_real = sys.stdout
sys.stdout = io.StringIO()

from Shale_Alg import RR2
from Waterfilling_Alg import waterfilling
from Common_Alg import generate_random_latin_square
from AI_Topology import (
    generate_random_topology, calculate_aspl,
    evolve_topology, evolve_topology_fct, evolve_topology_dynamic,
)
from Sirius import SiriusGen

sys.stdout = _real

from Traffic_Benchmarks import (
    calculate_topology_capacity, generate_uniform_traffic,
    generate_skewed_traffic, generate_hotspot_traffic,
    generate_adversarial_traffic, ArchitectureParams,
    run_load_sweep, run_h_sweep,
    compute_benchmark_score, TrafficType,
    find_adversarial_critical_points, get_hw_reconfig_ratio,
    generate_adversarial_skew_opera, generate_adversarial_skew_shale,
    generate_adversarial_skew_sirius, generate_adversarial_skew_ga,
    _get_all_pairs_dist,
)

# ── Configuration ────────────────────────────────────────────────────
IMG_DIR = 'images'
N = 9
POWER = 50.0
DPI = 150

np.random.seed(42)
random.seed(42)

# Architecture colours (consistent across all figures)
C_OPERA     = '#2196F3'
C_SHALE     = '#4CAF50'
C_SIRIUS    = '#FF9800'
C_GENETIC   = '#9C27B0'   # GA-Robust
C_GA_FCT    = '#E91E63'   # GA-FCT (magenta)
C_GA_DYN    = '#009688'   # GA-Dynamic (teal)

ARCH_COLORS = {'opera': C_OPERA, 'shale': C_SHALE,
               'sirius': C_SIRIUS, 'genetic': C_GENETIC,
               'ga_fct': C_GA_FCT, 'ga_dyn': C_GA_DYN}


# ====================================================================
# 1.  BUILD TOPOLOGIES
# ====================================================================
print("Building topologies ...")

_s = sys.stdout; sys.stdout = io.StringIO()
shale_adj = RR2(3, 2)
sys.stdout = _s

_s = sys.stdout; sys.stdout = io.StringIO()
opera_ls = generate_random_latin_square(N)
sys.stdout = _s
opera_adj = [sorted({v - 1 for v in opera_ls[i] if v - 1 != i})
             for i in range(N)]

_s = sys.stdout; sys.stdout = io.StringIO()
sirius_src = SiriusGen(N)
sys.stdout = _s
sirius_adj = [sirius_src[i] for i in range(N)]

# ── GA-evolved topologies (cached so repeated figure runs don't re-evolve) ──
import json, pathlib
_GA_CACHE = pathlib.Path(IMG_DIR) / '_ga_cache.json'

def _load_or_evolve():
    if _GA_CACHE.exists():
        try:
            with open(_GA_CACHE) as f:
                data = json.load(f)
            if all(k in data for k in ('ga_robust', 'ga_fct', 'ga_dyn')):
                print("  (loading GA topologies from cache)")
                return (data['ga_robust'], data['ga_fct'], data['ga_dyn'])
        except Exception:
            pass

    _s = sys.stdout; sys.stdout = io.StringIO()
    ga_robust = evolve_topology(N, 4, population_size=25, generations=40,
                                traffic_type="robust")
    ga_fct    = evolve_topology_fct(N, 4, population_size=25, generations=40,
                                    traffic_type="robust")
    ga_dyn, _routes, _paths = evolve_topology_dynamic(
        N, 4, population_size=25, generations=40, K_paths=3,
        traffic_type="robust", seed=42)
    sys.stdout = _s

    with open(_GA_CACHE, 'w') as f:
        json.dump({'ga_robust': ga_robust, 'ga_fct': ga_fct, 'ga_dyn': ga_dyn}, f)
    return ga_robust, ga_fct, ga_dyn


print("Evolving GA-Robust, GA-FCT, GA-Dynamic (may take a minute) …")
genetic_adj, ga_fct_adj, ga_dyn_adj = _load_or_evolve()

# Ensure connectivity fallback for any failed evolution
def _ensure_connected(adj, label):
    if calculate_aspl(adj) == float('inf'):
        print(f"  ! {label} evolved disconnected; falling back to random")
        for _ in range(20):
            rnd = generate_random_topology(N, 4)
            if calculate_aspl(rnd) < float('inf'):
                return rnd
    return adj

genetic_adj = _ensure_connected(genetic_adj, 'GA-Robust')
ga_fct_adj  = _ensure_connected(ga_fct_adj,  'GA-FCT')
ga_dyn_adj  = _ensure_connected(ga_dyn_adj,  'GA-Dynamic')

TOPOS = {
    'opera':   opera_adj,
    'shale':   shale_adj,
    'sirius':  sirius_adj,
    'genetic': genetic_adj,
    'ga_fct':  ga_fct_adj,
    'ga_dyn':  ga_dyn_adj,
}

for name, adj in TOPOS.items():
    deg = len([x for x in adj[0] if x is not None]) if adj[0] else 0
    print(f"  {name:8s}: N={len(adj)}, degree~{deg}")


# ====================================================================
# 2.  WATERFILLING VISUALISATION HELPERS
# ====================================================================

def _build_wf_data(adj_list, traffic, total_power=POWER):
    """Replicate waterfilling pipeline; return intermediates.

    Noise for channel (i,j) = hop_distance * scale_factor / demand.
    Higher demand → lower noise → more power allocated (priority).
    Longer path  → higher noise → harder to serve (bandwidth tax).
    """
    num = len(adj_list)
    dist = _get_all_pairs_dist(adj_list, num)
    target = num * 10.0

    noise_list, dem_list = [], []
    for i in range(num):
        s = sum(dist[i, j] if dist[i, j] != np.inf else 100
                for j in range(num) if i != j)
        sf = target / s if s > 0 else 1.0
        for j in range(num):
            if i == j:
                continue
            d = traffic[i, j]
            if d > 0:
                h = dist[i, j] if dist[i, j] != np.inf else 100
                noise_list.append(h * sf / d)
                dem_list.append(d)

    noise = np.array(noise_list)
    alloc = waterfilling(noise, total_power)

    active = alloc > 0
    wl = float((alloc[active] + noise[active])[0]) if active.any() else 0.0
    return noise, alloc, wl


def _plot_wf(noise, alloc, wl, title, fname, show_bottleneck=False):
    """Standard waterfilling bar-chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(noise))
    act = alloc > 0
    inact = ~act

    ax.bar(x[act], noise[act], color='#2196F3', alpha=.7,
           label='Noise (Active)')
    if inact.any():
        lbl = 'Noise (Bottleneck)' if show_bottleneck else 'Noise (Inactive)'
        ax.bar(x[inact], noise[inact], color='#FF9800', alpha=.5, label=lbl)
    ax.bar(x[act], alloc[act], bottom=noise[act],
           color='#4CAF50', alpha=.7, label='Allocated Power')
    ax.axhline(wl, color='red', ls='--', lw=1.5,
               label=f'Water Level ({wl:.2f})')

    ax.set_xlabel('Channel / Link Index (unitless)')
    ax.set_ylabel('Power / Noise Level (ratio)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, fname), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")


# ====================================================================
# 2b. THREE-PANEL WATERFILLING (Uniform / Skewed / Hotspot)
# ====================================================================
print("\n3-panel waterfilling figures …")

# Skew seed=42 → reproducible spatial pattern for the 3-panel waterfilling;
# different seeds produce different locality structures (same Zipf distribution).
_traffic_panels = [
    ('Uniform',  generate_uniform_traffic(N) * 0.5),
    ('Skewed',   generate_skewed_traffic(N, skew_factor=2.0, seed=42) * 0.5),
    ('Hotspot',  generate_hotspot_traffic(N) * 0.5),
]

def _plot_wf_3panel(adj_list, arch_name, fname, color='#2196F3',
                    total_power=POWER):
    """Generate a 3-panel heatmap: noise relative to water level per src-dst.

    Green = well below water level (easily served).
    Yellow = near water level (marginal).
    Red   = above water level (bottlenecked, gets no power).
    Gray  = no demand for that pair.
    """
    num = len(adj_list)
    results = []
    for plabel, ptm in _traffic_panels:
        noise, alloc, wl = _build_wf_data(adj_list, ptm, total_power)
        # Rebuild N×N noise matrix
        noise_mat = np.full((num, num), np.nan)
        alloc_mat = np.full((num, num), np.nan)
        idx = 0
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                d = ptm[i, j]
                if d > 0:
                    noise_mat[i, j] = noise[idx]
                    alloc_mat[i, j] = alloc[idx]
                    idx += 1
        results.append((plabel, noise_mat, alloc_mat, wl))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                             gridspec_kw={'width_ratios': [1, 1, 1, 0.05],
                                          'wspace': 0.35})
    cbar_ax = axes[3]  # dedicated colorbar axis

    for pi, (plabel, noise_mat, alloc_mat, wl) in enumerate(results):
        ax = axes[pi]
        # Compute noise / water_level ratio: <1 = served, >1 = bottlenecked
        ratio = np.full((num, num), np.nan)
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                if not np.isnan(noise_mat[i, j]) and wl > 0:
                    ratio[i, j] = noise_mat[i, j] / wl

        # Plot with diverging colormap centered at 1.0
        im = ax.imshow(ratio, cmap='RdYlGn_r', vmin=0.5, vmax=1.5,
                       aspect='equal', interpolation='nearest')

        # Annotate: show bottleneck count and served count
        bottleneck = np.nansum(ratio >= 1.0)
        served = np.nansum(ratio < 1.0)

        # Mark no-demand cells with gray
        for i in range(num):
            for j in range(num):
                if i != j and np.isnan(ratio[i, j]):
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                 fill=True, color='#E0E0E0', zorder=0))

        ax.set_xlabel('Destination')
        if pi == 0:
            ax.set_ylabel('Source')
        ax.set_title(f'{arch_name} — {plabel}\n'
                     f'Served: {int(served)}  |  Bottlenecked: {int(bottleneck)}')
        ax.set_xticks(range(num))
        ax.set_yticks(range(num))

    cbar = fig.colorbar(im, cax=cbar_ax, label='Noise / Water Level')
    cbar.ax.axhline(1.0, color='black', linewidth=2)

    plt.savefig(os.path.join(IMG_DIR, fname), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")

_plot_wf_3panel(shale_adj,   'Shale',  'shale_wf_3panel.png',  color=C_SHALE)
_plot_wf_3panel(opera_adj,   'Opera',  'opera_wf_3panel.png',  color=C_OPERA)
_plot_wf_3panel(sirius_adj,  'Sirius', 'sirius_wf_3panel.png', color=C_SIRIUS)


# ====================================================================
# 3.  SHALE WATERFILLING FIGURES
# ====================================================================
print("\nShale waterfilling figures …")
uni = generate_uniform_traffic(N)

# shale_rigorous
n, a, w = _build_wf_data(shale_adj, uni * 0.5, total_power=50)
_plot_wf(n, a, w, 'Shale Rigorous Waterfilling', 'shale_rigorous.png')

# shale_h1_wf  (lower power → fewer active channels)
n, a, w = _build_wf_data(shale_adj, uni * 0.3, total_power=30)
_plot_wf(n, a, w, 'Shale Waterfilling (h = 1)', 'shale_h1_wf.png')

# shale_h4_wf  (higher power → more channels active)
n, a, w = _build_wf_data(shale_adj, uni * 0.5, total_power=70)
_plot_wf(n, a, w, 'Shale VLB Sprayed (h = 4)', 'shale_h4_wf.png')

# shale_low_latency
n, a, w = _build_wf_data(shale_adj, uni * 0.25, total_power=25)
_plot_wf(n, a, w, 'Shale Low-Latency Waterfilling', 'shale_low_latency.png')

# shale_high_latency
n, a, w = _build_wf_data(shale_adj, uni * 0.6, total_power=40)
_plot_wf(n, a, w, 'Shale High-Latency Waterfilling', 'shale_high_latency.png')


# ====================================================================
# 4.  SHALE BENCHMARK FIGURES
# ====================================================================
print("\nShale benchmark figures …")

# ── 4a. shale_h_sweep_benchmark (2-panel) ────────────────────────────
h_vals = [1, 2, 4, 6, 8, 12]
h_res = run_h_sweep(shale_adj, h_values=h_vals, load_factor=0.5,
                    total_power=POWER)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(h_vals, h_res['throughput'], 'o-', color=C_SHALE,
         label='Simulated ($1/u_{\\max}$)')
ax1.plot(h_vals, h_res['theoretical_limit'], 's--', color='gray',
         label='Theoretical $1/(h{+}1)$')
ax1.set_xlabel('Spray Depth $h$ (unitless)')
ax1.set_ylabel('Normalized Throughput (fraction of line rate)')
ax1.set_title('Throughput vs Spray Depth')
ax1.legend()
ax1.grid(alpha=.3)

ax2.plot(h_vals, h_res['latency'], 'o-', color='#E53935', label='Latency')
# Regression line
coeffs = np.polyfit(h_vals, h_res['latency'], 1)
ax2.plot(h_vals, np.polyval(coeffs, h_vals), '--', color='gray',
         label=f'Fit: {coeffs[0]:.2f}h + {coeffs[1]:.2f}')
ax2.set_xlabel('Spray Depth $h$ (unitless)')
ax2.set_ylabel('Latency (normalized cycles)')
ax2.set_title('Latency vs Spray Depth')
ax2.legend()
ax2.grid(alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'shale_h_sweep_benchmark.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ shale_h_sweep_benchmark.png")

# ── 4b. shale_throughput_vs_load  &  shale_fct_vs_load ───────────────
loads = np.linspace(0.05, 0.60, 12).tolist()

fig_tp, ax_tp = plt.subplots(figsize=(8, 5))
fig_fc, ax_fc = plt.subplots(figsize=(8, 5))

cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(h_vals)))

for idx, h in enumerate(h_vals):
    params = ArchitectureParams(shale_h=h)
    res = run_load_sweep(shale_adj, 'shale', TrafficType.UNIFORM,
                         load_range=loads, params=params,
                         total_power=POWER)
    ax_tp.plot(loads, res['throughput'], 'o-', color=cmap[idx],
               label=f'h={h}', markersize=4)
    ax_fc.plot(loads, res['fct'], 'o-', color=cmap[idx],
               label=f'h={h}', markersize=4)

for ax, ylabel, title, fname in [
    (ax_tp, 'Normalized Throughput (fraction of line rate)',
     'Throughput vs Load Factor', 'shale_throughput_vs_load.png'),
    (ax_fc, 'Avg Normalized FCT (ratio)',
     'Flow Completion Time vs Load Factor', 'shale_fct_vs_load.png'),
]:
    ax.set_xlabel('Load Factor $L$ (ratio)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(IMG_DIR, fname),
                      dpi=DPI, bbox_inches='tight')
    plt.close(ax.figure)
    print(f"  ✓ {fname}")


# ====================================================================
# 5.  OPERA FIGURES
# ====================================================================
print("\nOpera figures …")

# ── Waterfilling vis ──
n, a, w = _build_wf_data(opera_adj, uni * 0.5, total_power=50)
_plot_wf(n, a, w, 'Opera Rigorous Waterfilling', 'opera_rigorous.png',
         show_bottleneck=True)

n, a, w = _build_wf_data(opera_adj, uni * 0.25, total_power=25)
_plot_wf(n, a, w, 'Opera Low-Latency Waterfilling', 'opera_low_latency.png')

# ── opera_efficiency_report (3-panel) ─────────────────────────────────
loads_fine = np.linspace(0.05, 0.95, 15).tolist()
opera_sweep = run_load_sweep(opera_adj, 'opera', TrafficType.UNIFORM,
                             load_range=loads_fine, total_power=POWER)

_op = ArchitectureParams()
_bulk_ceil = _op.opera_alpha * (1.0 - _op.opera_delta / _op.opera_t_cycle)  # ≈ 0.8832

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Throughput vs Load
ax1.plot(loads_fine, opera_sweep['throughput'], 'o-', color=C_OPERA,
         label='Effective Throughput')
ax1.axvline(_bulk_ceil, color='#E53935', ls='--', lw=1.5,
            label=f'Bulk Ceiling ρ*={_bulk_ceil:.3f}')
ax1.set_xlabel('Network Load ρ (fraction of capacity)')
ax1.set_ylabel('Normalized Throughput (fraction of line rate)')
ax1.set_title('Throughput vs Load\n(degrades above bulk ceiling)')
ax1.legend(fontsize=8)
ax1.grid(alpha=.3)

# Panel 2: Bandwidth Tax vs Load
ax2.plot(loads_fine, [bt * 100 for bt in opera_sweep['bandwidth_tax']],
         'o-', color=C_OPERA)
ax2.axvline(_bulk_ceil, color='#E53935', ls='--', lw=1.5,
            label=f'ρ*={_bulk_ceil:.3f}')
ax2.set_xlabel('Network Load ρ (fraction of capacity)')
ax2.set_ylabel('Bandwidth Tax (%)')
ax2.set_title('Bandwidth Tax vs Load\n(rises as α_eff drops)')
ax2.legend(fontsize=8)
ax2.grid(alpha=.3)

# Panel 3: Latency vs Load  (M/D/1 queuing effect)
ax3.plot(loads_fine, opera_sweep['latency_mean'], 'o-', color=C_OPERA)
ax3.axvline(_bulk_ceil, color='#E53935', ls='--', lw=1.5,
            label=f'ρ*={_bulk_ceil:.3f}')
ax3.set_xlabel('Network Load ρ (fraction of capacity)')
ax3.set_ylabel('Avg Latency (normalized cycles)')
ax3.set_title('Latency vs Load\n(M/D/1 queue blow-up)')
ax3.legend(fontsize=8)
ax3.grid(alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'opera_efficiency_report.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ opera_efficiency_report.png")

# ── opera_wf_throughput_vs_load ──────────────────────────────────────
# Delivered throughput = min(effective_capacity, offered_load).
# Below the bulk ceiling, throughput tracks offered load linearly.
# Above ρ* = bulk_ceil, bulk circuits saturate → throughput degrades.
fig, ax = plt.subplots(figsize=(8, 5))

cap_at_ceil = opera_sweep['throughput'][0]   # peak capacity (low-load, no penalty)
opera_delivered = [min(t, L) for t, L in zip(opera_sweep['throughput'], loads_fine)]
opera_ideal     = [min(L, cap_at_ceil) for L in loads_fine]

ax.plot(loads_fine, opera_delivered, 'o-', color=C_OPERA, lw=2,
        label='Delivered Throughput')
ax.plot(loads_fine, opera_ideal, '--', color='gray',
        label=f'Ideal (no saturation, cap={cap_at_ceil:.3f})')
ax.plot(loads_fine, opera_sweep['throughput'], 's:', color='#7B1FA2',
        markersize=4, lw=1, label='Effective Capacity Ceiling')
ax.axvline(_bulk_ceil, color='#E53935', ls='--', lw=1.5,
           label=f'Bulk Ceiling ρ*={_bulk_ceil:.3f}  (hw_ovhd={_op.opera_delta/_op.opera_t_cycle:.4f})')

# Shade the saturation region
ax.axvspan(_bulk_ceil, 1.0, alpha=0.07, color='#E53935',
           label='Bulk-circuit saturation zone')

ax.set_xlabel('Offered Load ρ (fraction of line rate)')
ax.set_ylabel('Throughput (fraction of line rate)')
ax.set_title('Opera: Throughput vs Offered Load\n'
             '(M/D/1 bulk-circuit saturation above ρ*)')
ax.legend(fontsize=8)
ax.grid(alpha=.3)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'opera_wf_throughput_vs_load.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ opera_wf_throughput_vs_load.png")


# ====================================================================
# 6.  SIRIUS FIGURES
# ====================================================================
print("\nSirius figures …")

# ── sirius_efficiency_report (2-panel) ────────────────────────────────
sir_sweep = run_load_sweep(sirius_adj, 'sirius', TrafficType.UNIFORM,
                           load_range=loads_fine, total_power=POWER)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

delivered = [min(t, L) for t, L in zip(sir_sweep['throughput'], loads_fine)]
ax1.plot(loads_fine, delivered, 'o-', color=C_SIRIUS,
         label='Delivered')
cap_ceil = max(sir_sweep['throughput'])  # capacity ceiling (peak before collapse)
ideal = [min(L, cap_ceil) for L in loads_fine]
ax1.plot(loads_fine, ideal, '--', color='gray',
         label=f'Ideal (cap={cap_ceil:.2f})')
ax1.axhline(cap_ceil, ls=':', color='#E53935', lw=1,
            label=f'$\\eta/\\mathcal{{H}}$ ceiling')
ax1.set_xlabel('Offered Load (fraction of line rate)')
ax1.set_ylabel('Delivered Throughput (fraction of line rate)')
ax1.set_title('Throughput vs Load')
ax1.legend()
ax1.grid(alpha=.3)

ax2.plot(loads_fine, sir_sweep['fct'], 'o-', color=C_SIRIUS)
ax2.set_xlabel('Offered Load (fraction of line rate)')
ax2.set_ylabel('Avg FCT (normalized ratio)')
ax2.set_title('Flow Completion Time vs Load')
ax2.grid(alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'sirius_efficiency_report.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ sirius_efficiency_report.png")

# ── sirius_inverse_wf (bar chart) ────────────────────────────────────
scenarios_sir = {}
for label, tm in [('Uniform\nTotal', generate_uniform_traffic(N) * 0.5),
                  ('Skewed\nFlow 0→1', generate_skewed_traffic(N) * 0.5)]:
    _, prim, _ = calculate_topology_capacity(
        sirius_adj, tm, total_power=POWER,
        architecture_type='sirius', return_metrics=True)
    scenarios_sir[label] = prim.throughput

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(list(scenarios_sir.keys()), list(scenarios_sir.values()),
              color=[C_SIRIUS, '#FFB74D'], edgecolor='black', linewidth=.5)
ax.set_ylabel('Achieved Throughput (fraction of line rate)')
ax.set_title('Sirius Throughput by Traffic Scenario')
ax.grid(axis='y', alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'sirius_inverse_wf.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ sirius_inverse_wf.png")


# ====================================================================
# 7.  GENETIC / AI FIGURES
# ====================================================================
print("\nGenetic figures …")

n, a, w = _build_wf_data(genetic_adj, uni * 0.5, total_power=50)
_plot_wf(n, a, w, 'GA-Evolved Rigorous Waterfilling',
         'genetic_node_0_rigorous.png')

n, a, w = _build_wf_data(genetic_adj, uni * 0.25, total_power=25)
_plot_wf(n, a, w, 'GA-Evolved Low-Latency Waterfilling',
         'genetic_low_latency.png')


# ====================================================================
# 8.  CROSS-ARCHITECTURE BENCHMARK FIGURES
# ====================================================================
print("\nCross-architecture benchmarks …")

arch_names  = ['opera', 'shale', 'sirius', 'genetic', 'ga_fct', 'ga_dyn']
arch_labels = ['Opera', 'Shale RR2', 'Sirius',
               'GA-Robust', 'GA-FCT', 'GA-Dynamic']
arch_colors = [C_OPERA, C_SHALE, C_SIRIUS, C_GENETIC, C_GA_FCT, C_GA_DYN]
# GA-* architectures all use the generic expander model (arch_type=None):
#   hw_reconfig_ratio = 0.0  → no reconfiguration overhead
#   r_eff = r / hops,  bw_tax = 1 - 1/hops,  latency = hops
# GA-Robust, GA-FCT, GA-Dynamic differ only in the evolved topology; the
# capacity-evaluation routing is shortest-path for all three here.  The
# routing-aware GA-Dynamic advantage is therefore undercounted in these
# plots (see discussion in \S 5.3 of main.tex).
arch_types  = ['opera', 'shale', 'sirius', None, None, None]

traffic_scenarios = {
    'uniform':        (TrafficType.UNIFORM,  'Uniform'),
    'skewed':         (TrafficType.SKEWED,   'Skewed'),
    'hotspot':        (TrafficType.HOTSPOT,  'Hotspot'),
    'traffic_demand': (TrafficType.ADVERSARIAL, 'Traffic Demand'),
}

loads_bench = np.linspace(0.05, 0.95, 12).tolist()

# ── 8a.  Load sweep per traffic scenario ──────────────────────────────
for scen_key, (ttype, scen_label) in traffic_scenarios.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (aname, alabel, acolor, atype) in enumerate(
            zip(arch_names, arch_labels, arch_colors, arch_types)):
        adj = TOPOS[aname]
        res = run_load_sweep(adj, atype, ttype,
                             load_range=loads_bench, total_power=POWER)
        ax1.plot(loads_bench, res['throughput'], 'o-', color=acolor,
                 label=alabel, markersize=4)
        ax2.plot(loads_bench, res['fct'], 'o-', color=acolor,
                 label=alabel, markersize=4)

    ax1.set_xlabel('Load Factor (ratio)')
    ax1.set_ylabel('Normalized Throughput (fraction of line rate)')
    ax1.set_title(f'Throughput ({scen_label})')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=.3)

    ax2.set_xlabel('Load Factor (ratio)')
    ax2.set_ylabel('Flow Completion Time (normalized ratio)')
    ax2.set_yscale('log')
    ax2.set_title(f'FCT ({scen_label})')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3)

    plt.tight_layout()
    fname = f'benchmark_load_sweep_{scen_key}.png'
    plt.savefig(os.path.join(IMG_DIR, fname), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")


# ── 8b.  benchmark_scenario_comparison (grouped bar chart) ────────────
composite_scores = {lbl: [] for lbl in arch_labels}

for scen_key, (ttype, scen_label) in traffic_scenarios.items():
    # Generate traffic
    if ttype == TrafficType.UNIFORM:
        tm = generate_uniform_traffic(N) * 0.5
    elif ttype == TrafficType.SKEWED:
        tm = generate_skewed_traffic(N, skew_factor=2.0, seed=42) * 0.5
    elif ttype == TrafficType.HOTSPOT:
        tm = generate_hotspot_traffic(N) * 0.5
    else:
        tm = generate_adversarial_traffic(N) * 0.5

    for alabel, atype, aname in zip(arch_labels, arch_types, arch_names):
        _, prim, sec = calculate_topology_capacity(
            TOPOS[aname], tm, total_power=POWER,
            architecture_type=atype, return_metrics=True)
        score = compute_benchmark_score(atype, prim, sec)
        composite_scores[alabel].append(score.composite())

scen_labels = [v[1] for v in traffic_scenarios.values()]
x = np.arange(len(scen_labels))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 6))
for i, (alabel, acolor) in enumerate(zip(arch_labels, arch_colors)):
    ax.bar(x + i * width, composite_scores[alabel], width,
           label=alabel, color=acolor, edgecolor='black', linewidth=.4)

ax.set_xlabel('Traffic Scenario')
ax.set_ylabel('Composite Benchmark Score $S_{bench}$ (dimensionless)')
ax.set_title('Cross-Architecture Comparison')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(scen_labels)
ax.legend()
ax.grid(axis='y', alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'benchmark_scenario_comparison.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ benchmark_scenario_comparison.png")


# ── 8c.  (removed) traffic_benchmark_capacity and traffic_benchmark_completion_time  ──
# These 3-panel power-budget figures were cut — the information is already in
# the four benchmark_load_sweep_{scenario}.png plots.


# ====================================================================
# 9.  ADVERSARIAL CRITICAL POINTS — data only; figure was cut in favour
#     of Table~\ref{tab:critical_points} in main.tex because the 6-line
#     overlay became unreadable after adding GA-FCT and GA-Dynamic.
# ====================================================================
print("\nAdversarial critical-points sweep (data printed, no figure) …")

crit_load_range = np.linspace(0.05, 0.97, 25).tolist()
for aname, atype, alabel in zip(arch_names, arch_types, arch_labels):
    crit = find_adversarial_critical_points(
        TOPOS[aname], atype,
        load_range=crit_load_range, total_power=POWER)
    print(f"  {alabel:<11s}  analytical ρ*={crit['analytical_critical']:.3f}  "
          f"exp(U/S/A)={crit['exp_critical_uniform']}, "
          f"{crit['exp_critical_skewed']}, "
          f"{crit['exp_critical_adversarial']}")


# ====================================================================
# 10.  ARCHITECTURE-TARGETED ADVERSARIAL SKEW COMPARISONS
# ====================================================================
print("\nAdversarial skew comparison figures …")

# Each sub-section targets one architecture's structural weakness.
# All four architectures are plotted in every graph so the reader can
# compare the targeted arch against the others under the same skew.
#
# Skew 1 — Opera-adversarial elephant flows
#   hot_demand=8, cold_demand=0.1, num_hot=2  →  Σ T ≈ 133
#   opera_rho_crit_L ≈ 0.8832·72/133 ≈ 0.48  (Opera saturates early)
#
# Skew 2 — Shale-adversarial many-to-one funnel
#   All N-1 sources → node 0  →  u_max = N-1 = 8
#   Shale throughput ≈ 1/8 of baseline; Opera/Sirius unaffected (1-hop).
#
# Skew 3 — GA-adversarial long-hop cyclic scatter
#   Traffic biased toward index-distant nodes; 17 % of GA pairs are 3-hop.
#   Opera/Sirius fully connected (hops=1 always), so they are unaffected.

_adv_loads = np.linspace(0.05, 0.95, 14).tolist()
_op_params  = ArchitectureParams()
_bulk_ceil  = _op_params.opera_alpha * (1.0 - _op_params.opera_delta /
                                         _op_params.opera_t_cycle)

# GA-adversarial traffic is topology-aware: it biases demand toward the
# actual long-hop pairs of the GA topology (ranks by BFS distance).
# On fully-connected topologies (Opera, Sirius) all pairs are 1-hop so
# the skew collapses to uniform → those architectures are unaffected.
_ga_adv_tm = generate_adversarial_skew_ga(genetic_adj, N)

_sirius_adv_tm = generate_adversarial_skew_sirius(N)
_sirius_sigma_t = np.sum(_sirius_adv_tm)
_sirius_lcrit = 0.85 * N * (N - 1) / _sirius_sigma_t

_adv_skews = [
    (
        'opera',
        generate_adversarial_skew_opera(N),
        'Opera-Adversarial Skew\n'
        r'(Elephant flows: $\rho_{bulk}$ exceeds ceiling at $L\approx0.48$)',
        'adversarial_skew_opera.png',
        f'Opera bulk ceiling $\\rho^*$={_bulk_ceil:.3f}',
        _bulk_ceil,
    ),
    (
        'shale',
        generate_adversarial_skew_shale(N),
        'Shale-Adversarial Skew\n'
        r'(Many-to-one funnel: $u_{max} \to N{-}1$, Shale throughput $\to 1/(N{-}1)$)',
        'adversarial_skew_shale.png',
        None, None,
    ),
    (
        'sirius',
        _sirius_adv_tm,
        'Sirius-Adversarial Skew\n'
        r'(Demand surge: VLB queue divergence at $\rho > 0.85$, exp collapse)',
        'adversarial_skew_sirius.png',
        f'Sirius VLB threshold $\\rho^*$=0.85 ($L\\approx${_sirius_lcrit:.2f})',
        _sirius_lcrit,
    ),
    (
        'genetic',
        _ga_adv_tm,
        'GA-Adversarial Skew (topology-aware long-hop)\n'
        r'(Traffic biased to actual 2–3 hop pairs; Opera/Sirius unaffected — all 1-hop)',
        'adversarial_skew_ga.png',
        None, None,
    ),
]

for target_arch, base_tm, fig_title, fname, vline_label, vline_x in _adv_skews:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for aname, alabel, acolor, atype in zip(arch_names, arch_labels,
                                             arch_colors, arch_types):
        res_tp, res_fct = [], []
        for L in _adv_loads:
            tm = base_tm * L
            fct, prim, _ = calculate_topology_capacity(
                TOPOS[aname], tm, total_power=POWER,
                architecture_type=atype, return_metrics=True)
            # Delivered throughput = min(capacity ceiling, offered load).
            # Caps at L to prevent the normalisation artifact (tp > 1) that
            # arises when waterfilling over-allocates power to low-noise hot
            # flows; also gives load-dependent saturation curves.
            res_tp.append(min(prim.throughput, L))
            res_fct.append(fct)

        lw = 2.5 if aname == target_arch else 1.2
        ls = '-' if lw > 2 else '--'
        ax1.plot(_adv_loads, res_tp, marker='o', ls=ls, lw=lw,
                 color=acolor, label=alabel, markersize=4)
        ax2.plot(_adv_loads, res_fct, marker='o', ls=ls, lw=lw,
                 color=acolor, label=alabel, markersize=4)

    # Mark the adversarial critical load on the throughput panel
    if vline_x is not None:
        ax1.axvline(vline_x, color='#E53935', ls=':', lw=1.5,
                    label=vline_label)

    ax1.set_xlabel('Load Factor $L$ (ratio)')
    ax1.set_ylabel('Normalized Throughput (fraction of line rate)')
    ax1.set_title(fig_title + '\n— Throughput', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=.3)

    ax2.set_xlabel('Load Factor $L$ (ratio)')
    ax2.set_ylabel('Flow Completion Time (normalized ratio)')
    ax2.set_yscale('log')
    ax2.set_title(fig_title + '\n— FCT', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, fname), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")


# ====================================================================
print(f"\nDone — 25 figures written to {IMG_DIR}/  (3 cut: capacity/completion/critical_points)")
# Figure inventory (28 total):
#  1-5   shale WF + benchmarks
#  6-10  opera WF + efficiency + throughput
#  11-12 sirius efficiency + bar
#  13-14 genetic WF
#  15-22 cross-arch benchmarks (4 scenarios × 2 + capacity + fct)
#  23    adversarial_critical_points
#  24-27 adversarial_skew_{opera,shale,sirius,ga}
