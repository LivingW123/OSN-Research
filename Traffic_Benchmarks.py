"""
===================================================================================
TRAFFIC BENCHMARKS MODULE
===================================================================================
Unified traffic generation and topology capacity calculation with full metrics.

Implements benchmarking metrics from the research framework:
- Primary Metrics: Throughput, FCT, Hop Count, Latency (mean, tail)
- Secondary Metrics: Bandwidth Tax, Capacity Efficiency, Saturation Point
- Architecture-Specific Models: Shale VLB, Opera Hybrid, Sirius Static
===================================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import collections


# =================================================================================
# SECTION 1: TRAFFIC MODELS
# =================================================================================

class TrafficType(Enum):
    """Traffic model types matching benchmark scenario definitions."""
    UNIFORM = "uniform"
    SKEWED = "skewed"
    HOTSPOT = "hotspot"
    ADVERSARIAL = "adversarial"


def generate_uniform_traffic(num_nodes: int) -> np.ndarray:
    """
    Uniform Traffic Model: All pairs communicate equally.
    
    Used as baseline benchmark for all architectures.
    Demand matrix: T[i,j] = 1 for all i ≠ j
    """
    traffic = np.ones((num_nodes, num_nodes))
    np.fill_diagonal(traffic, 0)
    return traffic


def generate_hotspot_traffic(num_nodes: int, hotspot_nodes: List[int] = None, 
                            intensity: float = 10.0) -> np.ndarray:
    """
    Hotspot Traffic Model: Certain nodes receive/send much more traffic.
    
    Used to stress specific nodes (e.g., aggregation points).
    From benchmarking framework Section 1.1.
    
    Args:
        num_nodes: Number of network nodes N
        hotspot_nodes: List of node indices that are hotspots
        intensity: Multiplier for traffic to/from hotspot nodes
    """
    if hotspot_nodes is None:
        hotspot_nodes = [0]
    traffic = np.ones((num_nodes, num_nodes))
    for node in hotspot_nodes:
        traffic[:, node] *= intensity
        traffic[node, :] *= intensity
    np.fill_diagonal(traffic, 0)
    return traffic


def generate_skewed_traffic(num_nodes: int, skew_factor: float = 2.0,
                            seed=None) -> np.ndarray:
    """
    Skewed Traffic Model: Zipf/power-law distribution over randomised node ranks.

    Each source node i draws its destinations in a random rank order; demand
    to the rank-r destination is  1 / (r^γ + 1).  A fresh random permutation
    per source is used each call, so different seeds produce spatially varied
    but statistically equivalent skew patterns.

    Skew ratio (max / min demand per source):
        ratio = D[i, rank-1] / D[i, rank-(N-1)]
              = (1+1) / (1/(N-1)^γ + 1)  ≈  (N-1)^γ  for large N.
        N=9, γ=2.0 → ratio ≈ 64×   (D_near ≈ 0.50, D_far ≈ 0.008)
        N=9, γ=1.0 → ratio ≈  8×   (D_near ≈ 0.50, D_far ≈ 0.111)

    Distribution: Zipf with exponent γ, rank indexed from 1.
    Mean demand per source ≈ Σ_{r=1}^{N-1} 1/(r^γ+1) / (N-1).

    Args:
        num_nodes:   Number of network nodes N
        skew_factor: Zipf exponent γ (≥0; 0 ≈ uniform, 1 = classical Zipf,
                     2 = quadratic fall-off).  Default 2.0.
        seed:        RNG seed for node-rank permutation.  None uses the global
                     numpy state (reproducible when caller sets np.random.seed).
                     Pass an explicit integer to isolate the pattern from other
                     random draws.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    traffic = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        dests = np.array([j for j in range(num_nodes) if j != i])
        rng.shuffle(dests)                           # randomise affinity order
        for rank, j in enumerate(dests, start=1):   # rank ∈ [1, N-1]
            traffic[i, j] = 1.0 / (rank ** skew_factor + 1)
    return traffic


def generate_adversarial_traffic(num_nodes: int) -> np.ndarray:
    """
    Adversarial Traffic Model: Maximizes congestion on bottleneck links.

    Each node sends all traffic to the node furthest away.
    Used to stress-test worst-case performance.
    """
    traffic = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        # All traffic goes to opposite node
        j = (i + num_nodes // 2) % num_nodes
        if i != j:
            traffic[i, j] = num_nodes  # Heavy concentrated demand
    return traffic


# ── Architecture-targeted adversarial skew generators ────────────────────────

def generate_adversarial_skew_opera(num_nodes: int,
                                    hot_demand: float = 8.0,
                                    cold_demand: float = 0.1,
                                    num_hot: int = 2) -> np.ndarray:
    """
    Opera-adversarial 'elephant-flow' skew.

    Two destination nodes attract very high demand from ALL sources.
    This drives opera_rho = Σ T / (N·(N-1)) above the bulk ceiling
    (α·(1−δ/T_cycle) ≈ 0.8832) at a LOWER load factor than uniform:

        L_crit ≈ bulk_ceil · N·(N-1) / Σ T
                ≈ 0.8832 · 72 / 132 ≈ 0.48

    Opera saturates at L ≈ 0.48; other architectures remain below their
    own critical loads at the same L.

    Parameters
    ----------
    hot_demand  : demand on pairs (i → hot_dest_j), default 8
    cold_demand : demand on all other pairs, default 0.1
    num_hot     : number of shared hot destinations (nodes 0..num_hot-1)
    """
    traffic = np.full((num_nodes, num_nodes), cold_demand)
    np.fill_diagonal(traffic, 0)
    for i in range(num_nodes):
        for j in range(num_hot):
            if i != j:
                traffic[i, j] = hot_demand
    return traffic


def generate_adversarial_skew_shale(num_nodes: int,
                                    target_node: int = 0,
                                    demand: float = 10.0) -> np.ndarray:
    """
    Shale-adversarial 'many-to-one funnel' skew.

    All N-1 source nodes direct their entire traffic to a single target.
    Under Shale VLB, every spray path must eventually exit through the
    target's ingress link, collapsing the congestion factor:

        u_max → N-1  ⟹  throughput ≈ rate / (N-1)

    Opera and Sirius serve the target directly (1-hop, fully connected).
    GA-Robust may use 1 or 2 hops depending on its evolved topology.
    """
    traffic = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if i != target_node:
            traffic[i, target_node] = demand
    return traffic


def generate_adversarial_skew_sirius(num_nodes: int,
                                     num_burst: int = 5,
                                     d_burst: float = 5.0,
                                     d_background: float = 0.5) -> np.ndarray:
    """
    Sirius-adversarial 'synchronized demand surge' skew.

    A majority of source nodes (the 'burst set' B, |B| = num_burst) transmit
    heavy demand d_burst to ALL destinations simultaneously, while the
    remaining sources carry only background demand d_background.

    Under Sirius VLB, every flow is sprayed through a random intermediate
    node.  When |B| burst sources each flood all N-1 destinations, every
    intermediate node must simultaneously buffer and forward spray traffic
    from |B| heavy sources — the aggregate load pushes the normalised
    offered load ρ = Σ T / (N·(N-1)) well above the VLB queue-divergence
    threshold ρ* = 0.85, triggering the exponential collapse
    C_load(ρ) = exp(−10(ρ − 0.85)).

    For N=9, |B|=5, d_burst=5.0, d_background=0.5:
        Σ_T  = 5·8·5.0 + 4·8·0.5 = 200 + 16 = 216
        ρ(L) = L · 216 / 72 = 3.0 L
        L_crit^Sirius ≈ 0.85 / 3.0 ≈ 0.283

    Opera saturates later (ρ* = 0.883 → L_crit ≈ 0.294) and decays more
    gently (rate 8 vs 10).  Shale and GA-Robust are unaffected: demand is
    spread across all destinations (no funnel), and the topology is fully
    connected (no long-hop bias).

    Parameters
    ----------
    num_nodes    : Number of network nodes N
    num_burst    : Number of burst-source nodes |B| (nodes 0..num_burst-1)
    d_burst      : Demand from each burst source to every destination
    d_background : Demand from each non-burst source to every destination
    """
    traffic = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if i < num_burst:
                    traffic[i, j] = d_burst
                else:
                    traffic[i, j] = d_background
    return traffic


def generate_adversarial_skew_ga(adj_list: List[List[int]],
                                 num_nodes: int,
                                 skew_factor: float = 2.0) -> np.ndarray:
    """
    GA-adversarial 'topology-aware long-hop' skew.

    Traffic is biased toward the actual long-hop destination pairs of the
    given sparse topology: rank 1 = farthest destination (most traffic),
    rank N-1 = nearest destination (least traffic).

    On a GA-evolved 4-regular graph (ASPL ≈ 1.81, 17 % of pairs are 3-hop):
        effective_rate = rate / hops  →  3-hop flows penalised 3×

    On fully-connected topologies (Opera, Sirius — all pairs hops=1) every
    destination is equidistant, so the skew degenerates to uniform demand
    and does NOT adversely affect those architectures.

    Parameters
    ----------
    adj_list    : adjacency list of the topology to target
    num_nodes   : number of nodes N
    skew_factor : Zipf exponent γ (default 2.0)
    """
    dist = _get_all_pairs_dist(adj_list, num_nodes)
    traffic = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        # Sort destinations: farthest first (rank 1 = most demand)
        dests = [(dist[i, j], j)
                 for j in range(num_nodes)
                 if j != i and dist[i, j] != np.inf]
        dests.sort(key=lambda x: -x[0])          # descending hop count
        for rank, (_, j) in enumerate(dests, start=1):
            traffic[i, j] = 1.0 / (rank ** skew_factor + 1)
    return traffic


# =================================================================================
# SECTION 2: METRICS DATA STRUCTURES
# =================================================================================

@dataclass
class PrimaryMetrics:
    """
    Primary Metrics from Benchmarking Framework Section 1.3.
    
    - throughput: Normalized to line rate [0, 1]. In this module, all
      capacity/throughput quantities are expressed as a fraction of a
      per-link line rate R, and we take R = 1.0 in the implementation.
    - fct: Flow Completion Time (normalized)
    - avg_hops: Average path length L(h)
    - latency_mean: Mean latency in cycles
    - latency_p99: 99th percentile latency
    """
    throughput: float = 0.0
    fct: float = 0.0
    avg_hops: float = 0.0
    latency_mean: float = 0.0
    latency_p99: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'throughput': self.throughput,
            'fct': self.fct,
            'avg_hops': self.avg_hops,
            'latency_mean': self.latency_mean,
            'latency_p99': self.latency_p99
        }


@dataclass
class SecondaryMetrics:
    """
    Secondary Metrics from Benchmarking Framework Section 1.3.
    
    - bandwidth_tax: Extra bandwidth used (L - 1)
    - capacity_efficiency: Utilized / Available capacity
    - bottleneck_util: Maximum link utilization
    - saturation_point: Load where throughput saturates
    """
    bandwidth_tax: float = 0.0
    capacity_efficiency: float = 0.0
    bottleneck_util: float = 0.0
    saturation_point: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'bandwidth_tax': self.bandwidth_tax,
            'capacity_efficiency': self.capacity_efficiency,
            'bottleneck_util': self.bottleneck_util,
            'saturation_point': self.saturation_point
        }


@dataclass
class ArchitectureParams:
    """
    Architecture-specific parameters for capacity calculation.
    
    Shale: h (spray depth), epoch length
    Opera: α (hybrid split), δ (reconfig delay), T_cycle
    Sirius: η (efficiency), slot duration, reconfig time
    """
    # Shale VLB Parameters (Section 2.1)
    # Guard-band overhead (~9 %, 4.096 ns of 45.056 ns slot) is already
    # folded into the 1/(2h) throughput bound and is not modelled as a
    # separate duty-cycle loss.  (Shale SIGCOMM'24, Amir et al., §5.)
    shale_h: int = 2                    # Spray depth h ∈ {1,2,4,6,8,12}
    shale_epoch: int = 15               # Epoch length E
    
    # Opera Hybrid Parameters (Section 2.2)
    # opera_alpha: byte-fraction of traffic routed over direct (bulk) paths.
    # Opera NSDI'20 §4.1 classifies flows ≥15 MB as bulk; the Datamining
    # workload (Fig. 1) shows ~92 % of *bytes* fall in this category,
    # giving α ≈ 0.92.  (Opera NSDI'20, Mellette et al., §5.1 / Fig. 7.)
    opera_alpha: float = 0.92           # Bulk fraction (92% bulk)
    opera_delta: float = 2.0            # Reconfiguration delay δ
    opera_t_cycle: float = 50.0         # Cycle time T_cycle

    # Sirius Static Parameters (Section 2.3)
    # Sirius SIGCOMM'20 §6 uses 90 ns transmission slots with a 10 ns
    # guardband (100 ns total), giving an operational overhead of 10 %.
    # The raw hardware switching time is 3.84 ns, but the paper
    # "conservatively" sets the guardband to 10 ns (Ballani et al., §7).
    sirius_delta: float = 0.10          # Operational guardband ratio 10ns / 100ns
    sirius_eta: float = 0.90            # Efficiency 1 - δ/T_slot


# =================================================================================
# SECTION 2b: UNIFIED HARDWARE OVERHEAD
# =================================================================================

def get_hw_reconfig_ratio(architecture_type: str,
                          params: 'ArchitectureParams') -> float:
    """
    Unified dimensionless reconfiguration overhead  δ / T_slot  for comparison.

    Returns the fraction of link time lost to hardware reconfiguration,
    expressed on the same scale for every architecture so the three can be
    compared directly:

        Shale:     0.0000  — oblivious back-to-back timeslot schedule;
                              the ~9 % guard-band (4.096 / 45.056 ns) is
                              already absorbed by the 1/(2h) throughput
                              bound and is not a separate duty-cycle loss.
                              (Shale SIGCOMM'24, Amir et al., §5.)
        Opera:     δ / T_cycle = opera_delta / opera_t_cycle
                              = 2.0 / 50.0 = 0.0400  (4.00 %)
                              Physical: δ ≈ 10 µs, T_cycle ≈ 250 µs.
                              (Opera NSDI'20, Mellette et al., §4.1.)
        Sirius:    δ / T_slot = sirius_delta
                              = 10 ns / 100 ns = 0.1000  (10.00 %)
                              The raw hardware switching time is only
                              3.84 ns, but the paper conservatively sets
                              the operational guardband to 10 ns.
                              (Sirius SIGCOMM'20, Ballani et al., §7.)
        GA-Robust: 0.0000  — static topology, no circuit reconfiguration.
    """
    if architecture_type == "opera":
        return params.opera_delta / params.opera_t_cycle
    elif architecture_type == "sirius":
        return params.sirius_delta   # already stored as δ / T_slot
    else:                            # "shale", "genetic", None
        return 0.0


# =================================================================================
# SECTION 3: CORE CAPACITY CALCULATION
# =================================================================================

def calculate_topology_capacity(adj_list: List[List[int]], 
                               traffic_matrix: np.ndarray, 
                               total_power: float = 50.0, 
                               architecture_type: str = None,
                               params: ArchitectureParams = None,
                               return_metrics: bool = False) -> float:
    """
    Calculate network capacity for a given topology and traffic demand.
    
    Implements the waterfilling algorithm with architecture-specific penalties:
    - Shale: VLB spraying with 1/(h+1) throughput scaling
    - Opera: Hybrid model with bulk/latency split
    - Sirius: Static scheduling with direct/indirect penalties
    
    From Benchmarking Framework Sections 2.1-2.3.
    
    Args:
        adj_list: Adjacency list representation of topology
        traffic_matrix: NxN traffic demand matrix T
        total_power: Power budget for waterfilling
        architecture_type: "shale", "opera", "sirius", or None
        params: Architecture-specific parameters
        return_metrics: If True, return (fct, primary_metrics, secondary_metrics)
    
    Returns:
        Total Flow Completion Time (or tuple with metrics if return_metrics=True)
    """
    from Waterfilling_Alg import waterfilling
    
    if params is None:
        params = ArchitectureParams()
    
    num_nodes = len(adj_list)
    weight_sum_target = num_nodes * 10.0
    
    # 1. Calculate all-pairs shortest paths
    dist_matrix = _get_all_pairs_dist(adj_list, num_nodes)
    
    # 2. Build channels and demands with weight normalization
    channels_noise = []
    demands = []
    flow_metas = []  # (src, dst, hops)
    
    for i in range(num_nodes):
        # Normalize per-node weights to Power Level target
        current_sum = sum(
            dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100
            for j in range(num_nodes) if i != j
        )
        scale_factor = weight_sum_target / current_sum if current_sum > 0 else 1.0
        
        for j in range(num_nodes):
            if i == j:
                continue
            demand = traffic_matrix[i, j]
            if demand > 0:
                hops = dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100
                normalized_noise = hops * scale_factor
                channels_noise.append(normalized_noise)
                demands.append(demand)
                flow_metas.append((i, j, hops))
    
    if not channels_noise:
        if return_metrics:
            return 0, PrimaryMetrics(), SecondaryMetrics()
        return 0

    # 2b. Pre-compute architecture-level quantities used inside the per-flow loop.

    # Opera: normalised offered load ρ (used for load-dependent α_eff / queuing)
    # NOT clipped to 1.0 — high-demand patterns (hotspot, adversarial) produce
    # ρ > 1, which is essential for M/D/1 queue divergence and traffic-dependent
    # FCT.  Opera NSDI'20 §4.1: "bulk flows wait for a direct circuit" — when
    # ρ ≫ 1 the wait grows without bound (queue divergence).
    opera_rho: float = 0.0
    opera_col_loads: np.ndarray = None  # per-destination offered load
    if architecture_type == "opera":
        total_demand = float(np.sum(traffic_matrix))
        opera_rho = total_demand / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
        opera_rho = float(max(opera_rho, 0.0))
        # Per-destination column load: each destination j receives Σ_i D[i,j].
        # Under Opera's round-robin rotor schedule each destination gets 1/N
        # of the total circuit time per epoch.  Destinations with above-average
        # ingress demand see proportionally higher bulk-queue occupancy.
        # Opera NSDI'20 §5.1 / Fig. 7: FCT depends on workload mix because
        # heavy destinations saturate their bulk circuits earlier.
        col_sums = np.sum(traffic_matrix, axis=0)  # per-destination ingress
        mean_col = float(np.mean(col_sums[col_sums > 0])) if np.any(col_sums > 0) else 1.0
        opera_col_loads = col_sums / (mean_col + 1e-9)  # ratio vs mean

    # Shale congestion pre-computation (Option C)
    #     Compute u_max and per-flow spray info BEFORE waterfilling
    #     so the per-flow loop can use congestion-based scaling.
    shale_congestion = None
    if architecture_type == "shale":
        from Shale_Alg import compute_spray_congestion
        flow_demands_list = [
            (src, dst, demands[idx])
            for idx, (src, dst, _) in enumerate(flow_metas)
        ]
        spray_k = max(3, params.shale_h)
        u_max, per_flow_info = compute_spray_congestion(
            adj_list, num_nodes, flow_demands_list,
            k=spray_k, penalty_factor=2.0
        )
        shale_congestion = (u_max, per_flow_info)

    # 3. Waterfilling allocation
    allocations = waterfilling(channels_noise, total_power)

    # 4. Calculate capacity with architecture-specific models
    total_capacity = 0
    total_completion_time = 0
    total_hops = 0
    total_bandwidth_tax = 0
    active_flows = 0
    latencies = []
    inactive_flows = []

    for idx in range(len(channels_noise)):
        p = allocations[idx]
        n = channels_noise[idx]
        demand = demands[idx]
        src, dst, hops = flow_metas[idx]

        if p > 0:
            # Rate from waterfilling allocation (demand-scaled)
            rate = demand * np.log2(1 + p/n)

            # Resolve per-flow congestion parameters for Shale
            cong_factor = None
            spray_hops = None
            if shale_congestion is not None:
                cong_u_max, cong_info = shale_congestion
                cong_factor = cong_u_max
                fi = cong_info.get((src, dst))
                if fi is not None and fi['num_paths'] > 0:
                    spray_hops = fi['avg_hops']

            # Apply architecture-specific capacity model
            # For Opera: pass per-destination load ratio so that flows to
            # heavily-loaded destinations see proportionally higher M/D/1
            # queuing (Opera NSDI'20 §4.1 / Fig. 7).
            dst_load_ratio = 1.0
            if opera_col_loads is not None and dst < len(opera_col_loads):
                dst_load_ratio = float(opera_col_loads[dst])

            effective_rate, bw_tax, latency, fct_wait = _apply_architecture_model(
                rate, hops, architecture_type, params,
                congestion_factor=cong_factor,
                spray_avg_hops=spray_hops,
                load_rho=opera_rho if architecture_type == "opera" else None,
                dest_load_ratio=dst_load_ratio,
            )

            total_capacity += effective_rate
            total_hops += hops
            total_bandwidth_tax += bw_tax
            active_flows += 1
            latencies.append(latency)

            # Flow Completion Time = Volume / Rate  + per-flow bulk-circuit wait (Opera).
            # Opera NSDI'20 §4.1: bulk flows (alpha fraction) wait for a direct
            # circuit before service begins; the M/D/1 wait time diverges as
            # the per-destination bulk load rho_dest -> 1.  The fct_wait term
            # is non-zero only for Opera; other architectures return 0 and
            # the FCT formula reduces to demand/rate as before.
            if effective_rate > 1e-9:
                bulk_wait_contribution = 0.0
                if architecture_type == "opera" and fct_wait > 0.0:
                    # Only the alpha fraction of the flow goes through the bulk
                    # circuit (and thus pays the rotor-schedule wait); the rest
                    # traverses the expander immediately.
                    bulk_wait_contribution = params.opera_alpha * fct_wait
                total_completion_time += demand / effective_rate + bulk_wait_contribution
            else:
                inactive_flows.append(demand)
        else:
            inactive_flows.append(demand)
            
    # Handle inactive flows: use worst active FCT as proxy
    if inactive_flows:
        if active_flows > 0 and total_completion_time > 0:
            # Worst-case: each inactive flow takes as long as the slowest active flow
            worst_active_fct = total_completion_time / active_flows
            for d in inactive_flows:
                total_completion_time += worst_active_fct
        else:
            # No active flows at all — everything is bottlenecked
            for d in inactive_flows:
                total_completion_time += d / 1e-9

    # Apply global capacity penalty (e.g., Sirius load sensitivity > 0.85)
    if architecture_type == "sirius":
        load = np.sum(traffic_matrix) / (num_nodes * (num_nodes - 1))
        if load > 0.85:
            penalty = np.exp(-10 * (load - 0.85))
            total_capacity *= penalty
            total_completion_time /= penalty

    # Opera bulk-ceiling saturation: throughput plateaus above the ceiling.
    # The bulk_ceil = α·(1−δ/T_cycle) ≈ 0.8832 is the maximum load the bulk
    # circuits can sustain.  Above this point, excess traffic overflows to the
    # expander.  On a fully-connected topology (hops=1 for all pairs) the
    # expander has no hop penalty, so throughput *plateaus* rather than
    # collapsing.  Per-destination M/D/1 queue divergence is now handled by
    # dest_load_ratio in _apply_architecture_model, providing traffic-dependent
    # FCT growth (Opera NSDI'20 §4.1 / Fig. 7).
    #
    # Model: penalty = min(1, ceil/ρ) — throughput cannot exceed ceiling × rate.
    #   ρ = 0.90 → penalty = 0.883/0.90 = 0.98
    #   ρ = 1.50 → penalty = 0.883/1.50 = 0.59  (hotspot plateau)
    #   ρ = 3.00 → penalty = 0.883/3.00 = 0.29
    if architecture_type == "opera" and opera_rho > 0:
        bulk_ceil_op = params.opera_alpha * (1.0 - params.opera_delta / params.opera_t_cycle)
        if opera_rho > bulk_ceil_op:
            penalty = bulk_ceil_op / opera_rho
            total_capacity *= penalty
            total_completion_time /= penalty
    
    if return_metrics:
        # Compute full metrics
        primary = PrimaryMetrics(
            throughput=total_capacity / (num_nodes * np.mean(demands)) if demands else 0,
            fct=total_completion_time,
            avg_hops=total_hops / active_flows if active_flows > 0 else 0,
            latency_mean=np.mean(latencies) if latencies else 0,
            latency_p99=np.percentile(latencies, 99) if len(latencies) > 1 else 0
        )
        
        bottleneck = 0.0
        if shale_congestion is not None:
            bottleneck = shale_congestion[0]  # u_max

        secondary = SecondaryMetrics(
            bandwidth_tax=total_bandwidth_tax / active_flows if active_flows > 0 else 0,
            capacity_efficiency=total_capacity / (num_nodes * total_power) if total_power > 0 else 0,
            bottleneck_util=bottleneck
        )
        
        return total_completion_time, primary, secondary
    
    return total_completion_time


def _apply_architecture_model(rate: float, hops: int,
                              architecture_type: str,
                              params: ArchitectureParams,
                              congestion_factor: float = None,
                              spray_avg_hops: float = None,
                              load_rho: float = None,
                              dest_load_ratio: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Apply architecture-specific capacity model.

    For Shale, when congestion_factor (u_max) is provided the throughput
    scales as r / u_max instead of the old r / (h+1).  spray_avg_hops
    drives bandwidth-tax and latency from actual spray paths.

    For Opera, load_rho enables load-dependent queuing latency and the
    effective bulk-fraction alpha_eff, which decreases above the bulk-circuit
    capacity ceiling (alpha * (1 - delta/T_cycle)).  dest_load_ratio scales
    the per-flow queuing by the destination's relative ingress load so that
    flows to hotspot destinations see proportionally higher M/D/1 wait times
    (Opera NSDI'20 §4.1 / Fig. 7).  The returned fct_wait is the M/D/1
    queueing wait at scale T_cycle; callers should add alpha * fct_wait to
    each Opera flow's FCT contribution (Opera NSDI'20 §4.1: bulk flows wait
    for a direct circuit on the rotor schedule before service begins).

    Returns: (effective_rate, bandwidth_tax, latency, fct_wait)
    """
    # Default: architectures other than Opera contribute zero FCT queue wait
    fct_wait = 0.0

    if architecture_type == "shale":
        # Shale VLB Model — congestion-based (Option C)
        # Each cell takes h spray hops + h direct hops = 2h total.
        # Throughput: r / u_eff where u_eff = max(u_max_measured, h+1)
        #   On well-connected graphs u_max ≈ h+1; on sparse graphs u_max > h+1.
        # Latency: τ = γ · (h+1) + β₀  (h+1 is the logical path length)
        gamma = 1.2   # Cycle overhead per spray hop
        beta_0 = 2.5  # Base processing latency
        h = params.shale_h

        # Effective congestion factor: at least h+1 (theoretical floor)
        if congestion_factor is not None and congestion_factor > 0:
            u_eff = max(congestion_factor, h + 1)
        else:
            u_eff = h + 1
        effective_rate = rate / u_eff

        # Shale paper: total path = 2h hops, tax = 1 - 1/(2h)
        total_path = 2 * h
        bw_tax = 1.0 - (1.0 / total_path) if total_path > 1 else 0.0
        # Latency uses the logical path length h+1 (not graph shortest path)
        latency = gamma * (h + 1) + beta_0

    elif architecture_type == "opera":
        # Opera Hybrid Model (Section 2.2)
        # Bulk fraction alpha uses direct 1-hop circuits; (1-alpha) uses expander.
        #
        # Load-dependent effects (requires load_rho):
        #   alpha_eff: decreases above the bulk-circuit capacity ceiling
        #     bulk_ceil = alpha * (1 - δ/T_cycle) ≈ 0.8832
        #     When rho > bulk_ceil, fraction (rho - bulk_ceil)/rho overflows
        #     to the expander, reducing effective alpha.
        #   queuing latency: M/D/1 model for bulk traffic waiting for its slot.
        #     queue_wait ≈ (rho_bulk * T_cycle) / (2 * (1 - rho_bulk))
        #     grows rapidly as rho_bulk → 1.
        alpha_split = params.opera_alpha
        delta = params.opera_delta
        t_cycle = params.opera_t_cycle
        rho = load_rho if load_rho is not None else 0.0

        reconfig_eff = 1.0 - (delta / t_cycle)
        bulk_ceil = alpha_split * reconfig_eff          # ≈ 0.8832

        # Effective alpha: decreases when offered load exceeds bulk capacity
        if rho > bulk_ceil:
            overflow_frac = min(1.0, (rho - bulk_ceil) / (rho + 1e-9))
            alpha_eff = max(alpha_split * (1.0 - overflow_frac * 0.5),
                            0.5 * alpha_split)
        else:
            alpha_eff = alpha_split

        bulk_rate = alpha_eff * rate * reconfig_eff
        short_rate = (1.0 - alpha_eff) * rate / max(1, hops)
        effective_rate = bulk_rate + short_rate

        bw_tax = (1.0 - alpha_eff) * (hops - 1) / max(1, hops)

        # Queuing latency: M/D/1 approximation for bulk circuit wait time.
        # Opera NSDI'20 §4.1: bulk flows wait for a direct circuit on the
        # round-robin rotor schedule.  Wait time is proportional to the
        # offered load on the *destination's* bulk circuits, not just the
        # global ρ.  dest_load_ratio (column-sum / mean-column-sum) scales
        # the per-destination offered load so that hotspot destinations
        # see proportionally higher queue occupancy and thus higher FCT.
        #
        # rho_dest: effective offered load for THIS destination's circuits.
        #   Under uniform traffic dest_load_ratio ≈ 1 → rho_dest ≈ rho_bulk.
        #   Under hotspot with intensity 10: hot dest ratio ≈ 4.5 → rho_dest
        #   diverges at much lower L, causing early FCT growth.
        #
        # The cap at 0.995 (not 0.98) allows queue_wait to reach ~100*delta
        # before clamping, making M/D/1 divergence visible in the FCT curves
        # while keeping numerical stability.
        rho_bulk_global = alpha_split * rho / max(bulk_ceil, 1e-9)
        rho_dest = rho_bulk_global * max(dest_load_ratio, 0.1)
        rho_dest = min(rho_dest, 0.995)
        queue_wait = (rho_dest * delta) / max(1e-6, 2.0 * (1.0 - rho_dest))
        latency = hops * delta + min(queue_wait, delta * 50.0)

        # FCT contribution from bulk-circuit wait (Opera NSDI'20 §4.1).
        # The paper's M/D/1 model operates on the rotor-cycle timescale T_cycle
        # (not delta), because a bulk flow waits for its NEXT direct-circuit
        # slot which arrives once per T_cycle/u per destination.  Below the
        # ceiling the wait is O(T_cycle/2); above the ceiling it diverges as
        # rho_dest -> 1 — exactly the M/D/1 blow-up described in §4.1.
        # We cap at rho_dest = 0.995 (already clamped above) to keep the
        # finite numerical value near 100 * T_cycle, which is enough to show
        # clear divergence in log-scale FCT plots without NaNs.
        fct_wait = (rho_dest * t_cycle) / max(1e-6, 2.0 * (1.0 - rho_dest))

    elif architecture_type == "sirius":
        # Sirius Static Model (Section 2.3)
        # Direct (1-hop): η efficiency
        # Sprayed (2-hop): η/2 (50% tax)
        # Multi-hop: η / hops^2 (quadratic degradation)
        eta = params.sirius_eta

        if hops <= 1:
            effective_rate = rate * eta
            bw_tax = 0
        elif hops <= 2:
            effective_rate = (rate * eta) / 2
            bw_tax = 0.5
        else:
            effective_rate = (rate * eta) / (hops ** 2)
            bw_tax = 1 - (1 / hops)

        latency = hops * (1.0 / eta)  # normalized: each slot costs 1/η time units
        
    else:
        # Generic expander model (GA-evolved topologies)
        # Multi-hop paths pay a bandwidth tax of (hops-1)/hops
        # No duty-cycle loss (static topology, no reconfiguration)
        effective_rate = rate / max(1, hops)
        bw_tax = 1.0 - (1.0 / max(1, hops))
        latency = hops
    
    return effective_rate, bw_tax, latency, fct_wait


def _get_all_pairs_dist(adj_list: List[List[int]], num_nodes: int) -> np.ndarray:
    """
    BFS all-pairs shortest paths.
    Returns NxN distance matrix.
    """
    dist_matrix = np.full((num_nodes, num_nodes), float('inf'))
    
    for start in range(num_nodes):
        dist_matrix[start, start] = 0
        queue = collections.deque([(start, 0)])
        visited = {start}
        
        while queue:
            u, d = queue.popleft()
            dist_matrix[start, u] = d
            
            for v in adj_list[u]:
                if v is not None and v not in visited:
                    visited.add(v)
                    queue.append((v, d + 1))
    
    return dist_matrix


# =================================================================================
# SECTION 4: LOAD SWEEP AND SATURATION ANALYSIS
# =================================================================================

def run_load_sweep(adj_list: List[List[int]], 
                  architecture_type: str,
                  traffic_type: TrafficType = TrafficType.UNIFORM,
                  load_range: List[float] = None,
                  params: ArchitectureParams = None,
                  total_power: float = 50.0) -> Dict:
    """
    Sweep load factor L from light to saturation.
    
    From Benchmarking Framework Section 1.1: Load Factor (L) Sweep.
    
    Returns:
        Dictionary with loads, throughput, fct, latency, and saturation_point
    """
    if load_range is None:
        load_range = np.linspace(0.05, 0.95, 10).tolist()
    if params is None:
        params = ArchitectureParams()
    
    num_nodes = len(adj_list)
    
    # Generate base traffic
    if traffic_type == TrafficType.UNIFORM:
        base_traffic = generate_uniform_traffic(num_nodes)
    elif traffic_type == TrafficType.SKEWED:
        base_traffic = generate_skewed_traffic(num_nodes)
    elif traffic_type == TrafficType.HOTSPOT:
        base_traffic = generate_hotspot_traffic(num_nodes)
    elif traffic_type == TrafficType.ADVERSARIAL:
        base_traffic = generate_adversarial_traffic(num_nodes)
    else:
        base_traffic = generate_uniform_traffic(num_nodes)
    
    results = {
        'loads': load_range,
        'throughput': [],
        'fct': [],
        'latency_mean': [],
        'bandwidth_tax': [],
        'saturation_point': None
    }
    
    for L in load_range:
        traffic = base_traffic * L
        
        fct, primary, secondary = calculate_topology_capacity(
            adj_list, traffic, total_power=total_power,
            architecture_type=architecture_type,
            params=params, return_metrics=True
        )
        
        results['throughput'].append(primary.throughput)
        results['fct'].append(fct)
        results['latency_mean'].append(primary.latency_mean)
        results['bandwidth_tax'].append(secondary.bandwidth_tax)
        
        # Detect saturation: throughput stops increasing
        if len(results['throughput']) >= 2:
            if results['throughput'][-1] < results['throughput'][-2] * 1.02:
                if results['saturation_point'] is None:
                    results['saturation_point'] = L
    
    return results


def run_h_sweep(adj_list: List[List[int]],
               h_values: List[int] = None,
               load_factor: float = 0.5,
               total_power: float = 50.0) -> Dict:
    """
    Shale-specific: Sweep spray depth h.

    From Benchmarking Framework Section 2.1:
    - Throughput scaling: r / u_max  (congestion-based)
    - Average hops: L(h) ≈ h + 1  (from actual spray paths)
    - Latency: τ = α·L_spray + β

    Returns:
        Dictionary with h_values, throughput, latency, u_max,
        avg_spray_hops, and theoretical_limit (old 1/(h+1) for reference)
    """
    if h_values is None:
        h_values = [1, 2, 4, 6, 8, 12]

    num_nodes = len(adj_list)
    traffic = generate_uniform_traffic(num_nodes) * load_factor

    results = {
        'h_values': h_values,
        'throughput': [],
        'avg_hops': [],
        'latency': [],
        'u_max': [],
        'theoretical_limit': []
    }

    for h in h_values:
        params = ArchitectureParams(shale_h=h)

        fct, primary, secondary = calculate_topology_capacity(
            adj_list, traffic, total_power=total_power,
            architecture_type="shale", params=params,
            return_metrics=True
        )

        results['throughput'].append(primary.throughput)
        results['avg_hops'].append(primary.avg_hops)
        results['latency'].append(primary.latency_mean)
        results['u_max'].append(secondary.bottleneck_util)
        results['theoretical_limit'].append(1.0 / (h + 1))

    return results


# =================================================================================
# SECTION 4b: ADVERSARIAL CRITICAL POINTS
# =================================================================================

def find_adversarial_critical_points(
        adj_list: List[List[int]],
        architecture_type: str,
        params: ArchitectureParams = None,
        total_power: float = 50.0,
        load_range: List[float] = None,
        drop_threshold: float = 0.03) -> Dict:
    """
    Identify the load factor at which each architecture becomes adversarial.

    Two estimates are returned for each architecture:

    Experimental (simulation-derived)
        The first load L* where  T(L*) < T(L* - ΔL) - drop_threshold,
        i.e. throughput drops by more than drop_threshold in one step.
        Evaluated under Uniform, Skewed, and Adversarial traffic.

    Analytical (closed-form)
        Sirius:  ρ* = 0.85  — onset of exp(-10(ρ-0.85)) load penalty;
                              queue divergence at VLB intermediate nodes.
        Opera:   ρ* = α·(1-δ/T_c)  — bulk-circuit capacity ceiling;
                              above this, overflow to expander increases tax
                              and queuing latency grows steeply (M/D/1).
        Shale:   ρ* = 1/(h+1)  — VLB throughput ceiling; further load
                              only deepens queue, no extra throughput.
        GA-Robust: ρ* ≈ 1/ASPL — expander capacity ceiling based on
                              average shortest path length.

    Returns dict with keys:
        analytical_critical, reasoning, hw_overhead,
        exp_critical_{uniform,skewed,adversarial},
        sweep_{uniform,skewed,adversarial}  (full run_load_sweep dicts)
    """
    if load_range is None:
        load_range = np.linspace(0.05, 0.97, 25).tolist()
    if params is None:
        params = ArchitectureParams()

    num_nodes = len(adj_list)

    # ── Experimental sweeps ──────────────────────────────────────────
    sweeps = {}
    exp_critical = {}
    for ttype_key, ttype in [('uniform',    TrafficType.UNIFORM),
                              ('skewed',     TrafficType.SKEWED),
                              ('adversarial', TrafficType.ADVERSARIAL)]:
        sw = run_load_sweep(adj_list, architecture_type, ttype,
                            load_range, params, total_power)
        sweeps[ttype_key] = sw
        tp = sw['throughput']
        # Critical point: capacity ceiling < offered load by more than drop_threshold.
        # In the waterfilling model, throughput is approximately constant (the capacity
        # ceiling of the architecture); the network becomes adversarial at the first
        # load L where  L > throughput_ceiling + drop_threshold, i.e. delivered < offered.
        # Also detect an outright throughput drop (Sirius exponential collapse).
        ceiling = max(tp[:max(1, len(tp)//2)])   # estimate ceiling from lower loads
        crit = None
        for k, L in enumerate(load_range):
            if tp[k] < tp[k - 1] - drop_threshold:   # sharp collapse
                crit = L
                break
            if L > ceiling + drop_threshold:          # load exceeds capacity
                crit = L
                break
        exp_critical[ttype_key] = crit

    # ── Analytical estimate ──────────────────────────────────────────
    hw_ratio = get_hw_reconfig_ratio(architecture_type, params)

    if architecture_type == "sirius":
        analytical = 0.85
        reasoning = (
            "Queue divergence at VLB intermediate nodes. "
            "The load penalty C_load(ρ) = exp(-10(ρ-0.85)) activates at ρ=0.85, "
            "causing exponential throughput collapse. "
            f"hw_reconfig_ratio = {hw_ratio:.4f}."
        )
    elif architecture_type == "opera":
        bulk_ceil = params.opera_alpha * (1.0 - params.opera_delta / params.opera_t_cycle)
        analytical = round(bulk_ceil, 4)
        reasoning = (
            f"Bulk-circuit capacity ceiling α·(1-δ/T_c) = {analytical:.4f}. "
            "Above this load, bulk circuits are fully booked; overflow traffic "
            "is forced onto the expander, increasing bandwidth tax and triggering "
            "M/D/1 queuing growth. "
            f"hw_reconfig_ratio = {hw_ratio:.4f}."
        )
    elif architecture_type == "shale":
        analytical = round(1.0 / (params.shale_h + 1), 4)
        reasoning = (
            f"VLB throughput ceiling 1/(h+1) = {analytical:.4f} for h={params.shale_h}. "
            "Beyond this load the network is congestion-limited; additional offered "
            "load only builds queue depth without increasing delivered throughput. "
            f"hw_reconfig_ratio = {hw_ratio:.4f} (zero — no reconfiguration)."
        )
    else:
        # GA-Robust / generic: derive from ASPL
        dist = _get_all_pairs_dist(adj_list, num_nodes)
        finite = dist[(dist != float('inf')) & (dist != 0)]
        aspl = float(np.mean(finite)) if len(finite) > 0 else float('inf')
        analytical = round(1.0 / aspl, 4) if aspl > 0 else 0.0
        reasoning = (
            f"Expander capacity ceiling ≈ 1/ASPL = {analytical:.4f} "
            f"(ASPL = {aspl:.3f} hops). "
            "The topology has no reconfiguration overhead (hw_reconfig_ratio = 0), "
            "so the only ceiling is the multi-hop bandwidth tax."
        )

    return {
        'analytical_critical':        analytical,
        'reasoning':                  reasoning,
        'hw_overhead':                hw_ratio,
        'exp_critical_uniform':       exp_critical['uniform'],
        'exp_critical_skewed':        exp_critical['skewed'],
        'exp_critical_adversarial':   exp_critical['adversarial'],
        'sweep_uniform':              sweeps['uniform'],
        'sweep_skewed':               sweeps['skewed'],
        'sweep_adversarial':          sweeps['adversarial'],
    }


# =================================================================================
# SECTION 5: CROSS-ARCHITECTURE COMPARISON
# =================================================================================

def compare_architectures(architectures: Dict[str, List[List[int]]],
                         traffic_type: TrafficType = TrafficType.UNIFORM,
                         power_range: List[float] = None,
                         load_factor: float = 0.5) -> Dict:
    """
    Compare multiple architectures on unified benchmark dimensions.
    
    From Benchmarking Framework Section 3: Cross-Architecture Comparison.
    
    Args:
        architectures: Dict mapping name -> adjacency list
        traffic_type: Traffic model for benchmark
        power_range: Power budget sweep range
        load_factor: Fixed load factor
    
    Returns:
        Dictionary with per-architecture results for plotting
    """
    if power_range is None:
        power_range = np.linspace(20, 100, 5).tolist()
    
    # Infer num_nodes from first architecture
    num_nodes = len(next(iter(architectures.values())))
    
    # Generate traffic
    if traffic_type == TrafficType.UNIFORM:
        traffic = generate_uniform_traffic(num_nodes)
    elif traffic_type == TrafficType.SKEWED:
        traffic = generate_skewed_traffic(num_nodes)
    elif traffic_type == TrafficType.HOTSPOT:
        traffic = generate_hotspot_traffic(num_nodes)
    else:
        traffic = generate_uniform_traffic(num_nodes)
    
    traffic = traffic * load_factor
    
    results = {}
    
    for arch_name, adj_list in architectures.items():
        # Determine architecture type
        arch_type = None
        if "opera" in arch_name.lower():
            arch_type = "opera"
        elif "shale" in arch_name.lower():
            arch_type = "shale"
        elif "sirius" in arch_name.lower():
            arch_type = "sirius"
        
        arch_results = {
            'power_levels': power_range,
            'fct': [],
            'throughput': [],
            'bandwidth_tax': []
        }
        
        for power in power_range:
            fct, primary, secondary = calculate_topology_capacity(
                adj_list, traffic, total_power=power,
                architecture_type=arch_type, return_metrics=True
            )
            
            arch_results['fct'].append(fct)
            arch_results['throughput'].append(primary.throughput)
            arch_results['bandwidth_tax'].append(secondary.bandwidth_tax)
        
        results[arch_name] = arch_results
    
    return results


# =================================================================================
# SECTION 6: BENCHMARK SCORE COMPUTATION
# =================================================================================

@dataclass
class BenchmarkScore:
    """
    Benchmark Score from Framework Section 1.2.6.

    Multiplicative composite:
      S = T_norm * (1 - B_tax) * (1 - D_loss) * 1/(1 + tau) * 1/FCT_norm
    """
    architecture: str
    throughput_norm: float       # T_norm (§1.2.1)
    fct_norm: float              # FCT_norm (§1.2.2)
    latency_mean: float          # tau_bar (§1.2.3)
    bandwidth_tax: float         # B_tax (§1.2.4)
    duty_cycle_loss: float       # D_loss (§1.2.5)

    def composite(self) -> float:
        """Multiplicative composite score (§1.2.6)."""
        fct_factor = 1.0 / (1.0 + np.log(max(self.fct_norm, 1.0)))
        latency_factor = 1.0 / (1.0 + self.latency_mean)
        return (self.throughput_norm
                * (1.0 - self.bandwidth_tax)
                * (1.0 - self.duty_cycle_loss)
                * latency_factor
                * fct_factor)


def _duty_cycle_loss(architecture: str, params: ArchitectureParams) -> float:
    """Compute D_loss for a given architecture."""
    if architecture == "opera":
        return params.opera_delta / params.opera_t_cycle
    elif architecture == "sirius":
        return params.sirius_delta  # already a ratio δ/T_slot
    else:  # shale, genetic, etc.
        return 0.0


def compute_benchmark_score(architecture: str,
                           primary: PrimaryMetrics,
                           secondary: SecondaryMetrics,
                           params: ArchitectureParams = None) -> BenchmarkScore:
    """
    Compute benchmark score from collected metrics.
    """
    if params is None:
        params = ArchitectureParams()

    return BenchmarkScore(
        architecture=architecture,
        throughput_norm=primary.throughput,
        fct_norm=max(primary.fct, 1.0),
        latency_mean=primary.latency_mean,
        bandwidth_tax=secondary.bandwidth_tax,
        duty_cycle_loss=_duty_cycle_loss(architecture, params)
    )
