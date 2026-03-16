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

from shale_alg import RR2
from Waterfilling_Alg import waterfilling
from Common_Alg import generate_random_latin_square
from AI_Topology import generate_random_topology, calculate_aspl
from Sirius import SiriusGen

sys.stdout = _real

from Traffic_Benchmarks import (
    calculate_topology_capacity, generate_uniform_traffic,
    generate_skewed_traffic, generate_hotspot_traffic,
    generate_adversarial_traffic, ArchitectureParams,
    run_load_sweep, run_h_sweep,
    compute_benchmark_score, TrafficType,
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
C_OPERA   = '#2196F3'
C_SHALE   = '#4CAF50'
C_SIRIUS  = '#FF9800'
C_GENETIC = '#9C27B0'

ARCH_COLORS = {'opera': C_OPERA, 'shale': C_SHALE,
               'sirius': C_SIRIUS, 'genetic': C_GENETIC}


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

# Genetic: retry until connected
for _ in range(20):
    genetic_adj = generate_random_topology(N, 4)
    if calculate_aspl(genetic_adj) < float('inf'):
        break

TOPOS = {
    'opera':   opera_adj,
    'shale':   shale_adj,
    'sirius':  sirius_adj,
    'genetic': genetic_adj,
}

for name, adj in TOPOS.items():
    deg = len([x for x in adj[0] if x is not None]) if adj[0] else 0
    print(f"  {name:8s}: N={len(adj)}, degree~{deg}")


# ====================================================================
# 2.  WATERFILLING VISUALISATION HELPERS
# ====================================================================

def _build_wf_data(adj_list, traffic, total_power=POWER):
    """Replicate waterfilling pipeline; return intermediates."""
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
                noise_list.append(h * sf)
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

    ax.set_xlabel('Channel / Link Index')
    ax.set_ylabel('Power / Noise Level')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, fname), dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")


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
ax1.set_xlabel('Spray Depth ($h$)')
ax1.set_ylabel('Normalized Throughput')
ax1.set_title('Throughput vs Spray Depth')
ax1.legend()
ax1.grid(alpha=.3)

ax2.plot(h_vals, h_res['latency'], 'o-', color='#E53935', label='Latency')
# Regression line
coeffs = np.polyfit(h_vals, h_res['latency'], 1)
ax2.plot(h_vals, np.polyval(coeffs, h_vals), '--', color='gray',
         label=f'Fit: {coeffs[0]:.2f}h + {coeffs[1]:.2f}')
ax2.set_xlabel('Spray Depth ($h$)')
ax2.set_ylabel('Latency (cycles)')
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
    (ax_tp, 'Normalized Throughput', 'Throughput vs Load Factor',
     'shale_throughput_vs_load.png'),
    (ax_fc, 'Avg Normalized FCT', 'Flow Completion Time vs Load Factor',
     'shale_fct_vs_load.png'),
]:
    ax.set_xlabel('Load Factor ($L$)')
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

# ── opera_efficiency_report (2-panel) ─────────────────────────────────
loads_fine = np.linspace(0.05, 0.95, 15).tolist()
opera_sweep = run_load_sweep(opera_adj, 'opera', TrafficType.UNIFORM,
                             load_range=loads_fine, total_power=POWER)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(loads_fine, [bt * 100 for bt in opera_sweep['bandwidth_tax']],
         'o-', color=C_OPERA)
ax1.set_xlabel('Network Load')
ax1.set_ylabel('Bandwidth Tax (%)')
ax1.set_title('Bandwidth Tax vs Load')
ax1.grid(alpha=.3)

ax2.plot(loads_fine, opera_sweep['latency_mean'], 'o-', color=C_OPERA)
ax2.set_xlabel('Network Load')
ax2.set_ylabel('Avg Latency (cycles)')
ax2.set_title('Latency vs Load')
ax2.grid(alpha=.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'opera_efficiency_report.png'),
            dpi=DPI, bbox_inches='tight')
plt.close()
print("  ✓ opera_efficiency_report.png")

# ── opera_wf_throughput_vs_load ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loads_fine, opera_sweep['throughput'], 'o-', color=C_OPERA,
        label='Delivered Throughput (WF-Limited)')
ax.plot(loads_fine, loads_fine, '--', color='gray', label='Ideal (Zero Loss)')
if opera_sweep['throughput']:
    cap = max(opera_sweep['throughput'])
    ax.axhline(cap, ls=':', color='#E53935',
               label=f'Waterfilling Capacity Limit ({cap:.3f})')
ax.set_xlabel('Offered Load (Rate per node)')
ax.set_ylabel('Throughput (rate per node)')
ax.set_title('Opera Throughput vs Offered Load')
ax.legend(fontsize=8)
ax.grid(alpha=.3)
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

ax1.plot(loads_fine, sir_sweep['throughput'], 'o-', color=C_SIRIUS,
         label='Simulation')
ax1.plot(loads_fine, loads_fine, '--', color='gray', label='Ideal')
ax1.set_xlabel('Offered Load')
ax1.set_ylabel('Throughput')
ax1.set_title('Throughput vs Load')
ax1.legend()
ax1.grid(alpha=.3)

ax2.plot(loads_fine, sir_sweep['fct'], 'o-', color=C_SIRIUS)
ax2.set_xlabel('Offered Load')
ax2.set_ylabel('Avg FCT (slots)')
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
ax.set_ylabel('Achieved Throughput')
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

arch_names  = ['opera', 'shale', 'sirius', 'genetic']
arch_labels = ['Opera', 'Shale RR2', 'Sirius', 'GA-Robust']
arch_colors = [C_OPERA, C_SHALE, C_SIRIUS, C_GENETIC]
arch_types  = ['opera', 'shale', 'sirius', None]  # genetic has no model

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

    ax1.set_xlabel('Load Factor')
    ax1.set_ylabel('Normalized Throughput')
    ax1.set_title(f'Throughput ({scen_label})')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=.3)

    ax2.set_xlabel('Load Factor')
    ax2.set_ylabel('Flow Completion Time')
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
        tm = generate_skewed_traffic(N) * 0.5
    elif ttype == TrafficType.HOTSPOT:
        tm = generate_hotspot_traffic(N) * 0.5
    else:
        tm = generate_adversarial_traffic(N) * 0.5

    for alabel, atype, aname in zip(arch_labels, arch_types, arch_names):
        _, prim, sec = calculate_topology_capacity(
            TOPOS[aname], tm, total_power=POWER,
            architecture_type=atype, return_metrics=True)
        score = compute_benchmark_score(alabel, prim, sec)
        composite_scores[alabel].append(score.composite())

scen_labels = [v[1] for v in traffic_scenarios.values()]
x = np.arange(len(scen_labels))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 6))
for i, (alabel, acolor) in enumerate(zip(arch_labels, arch_colors)):
    ax.bar(x + i * width, composite_scores[alabel], width,
           label=alabel, color=acolor, edgecolor='black', linewidth=.4)

ax.set_xlabel('Scenario')
ax.set_ylabel('Composite Benchmark Score')
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


# ── 8c.  traffic_benchmark_capacity  &  _completion_time (3-panel) ───
power_range = np.linspace(20, 100, 8).tolist()
panel_traffics = [
    ('Uniform', generate_uniform_traffic(N) * 0.5),
    ('Skewed',  generate_skewed_traffic(N) * 0.5),
    ('Hotspot', generate_hotspot_traffic(N) * 0.5),
]

fig_cap, axes_cap = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig_fct, axes_fct = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for pi, (plabel, ptm) in enumerate(panel_traffics):
    for alabel, atype, aname, acolor in zip(
            arch_labels, arch_types, arch_names, arch_colors):
        caps, fcts = [], []
        for pwr in power_range:
            fct, prim, _ = calculate_topology_capacity(
                TOPOS[aname], ptm, total_power=pwr,
                architecture_type=atype, return_metrics=True)
            # Shannon capacity proxy: total effective rate
            caps.append(prim.throughput)
            fcts.append(fct)
        axes_cap[pi].plot(power_range, caps, 'o-', color=acolor,
                          label=alabel, markersize=4)
        axes_fct[pi].plot(power_range, fcts, 'o-', color=acolor,
                          label=alabel, markersize=4)

    for ax_arr, ylabel in [(axes_cap, 'Shannon Capacity'),
                           (axes_fct, 'Total FCT (s)')]:
        ax_arr[pi].set_xlabel('Power Budget (P)')
        ax_arr[pi].set_title(plabel)
        ax_arr[pi].legend(fontsize=7)
        ax_arr[pi].grid(alpha=.3)
    axes_cap[0].set_ylabel('Shannon Capacity')
    axes_fct[0].set_ylabel('Total Flow Completion Time (s)')

for figobj, fname in [(fig_cap, 'traffic_benchmark_capacity.png'),
                       (fig_fct, 'traffic_benchmark_completion_time.png')]:
    figobj.tight_layout()
    figobj.savefig(os.path.join(IMG_DIR, fname), dpi=DPI, bbox_inches='tight')
    plt.close(figobj)
    print(f"  ✓ {fname}")


# ====================================================================
print(f"\nDone — {23} figures written to {IMG_DIR}/")
