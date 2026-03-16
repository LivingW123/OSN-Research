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


def generate_skewed_traffic(num_nodes: int, skew_factor: float = 2.0) -> np.ndarray:
    """
    Skewed Traffic Model: Power-law/Zipf-like distribution.
    
    Traffic intensity decreases with logical distance between nodes.
    Represents realistic datacenter traffic patterns.
    
    Args:
        num_nodes: Number of network nodes N
        skew_factor: Power-law exponent (higher = more skewed)
    """
    traffic = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: 
                continue
            # Closer indices = more traffic (locality pattern)
            traffic[i, j] = 1.0 / (abs(i - j) ** skew_factor + 1)
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
    shale_h: int = 2                    # Spray depth h ∈ {1,2,4,6,8,12}
    shale_epoch: int = 15               # Epoch length E
    
    # Opera Hybrid Parameters (Section 2.2)
    opera_alpha: float = 0.92           # Bulk fraction (92% bulk)
    opera_delta: float = 2.0            # Reconfiguration delay δ
    opera_t_cycle: float = 50.0         # Cycle time T_cycle
    
    # Sirius Static Parameters (Section 2.3)
    sirius_delta: float = 0.0384        # Reconfig ratio δ/T_slot (3.84ns / 100ns)
    sirius_eta: float = 0.9616          # Efficiency 1 - δ/T_slot


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

    # 2b. Shale congestion pre-computation (Option C)
    #     Compute u_max and per-flow spray info BEFORE waterfilling
    #     so the per-flow loop can use congestion-based scaling.
    shale_congestion = None
    if architecture_type == "shale":
        from shale_alg import compute_spray_congestion
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
            effective_rate, bw_tax, latency = _apply_architecture_model(
                rate, hops, architecture_type, params,
                congestion_factor=cong_factor,
                spray_avg_hops=spray_hops
            )
            
            total_capacity += effective_rate
            total_hops += hops
            total_bandwidth_tax += bw_tax
            active_flows += 1
            latencies.append(latency)
            
            # Flow Completion Time = Volume / Rate
            if effective_rate > 1e-9:
                total_completion_time += demand / effective_rate
            else:
                total_completion_time += 1e6
        else:
            total_completion_time += 1e6
            
    # Apply global capacity penalty (e.g., Sirius load sensitivity > 0.85)
    if architecture_type == "sirius":
        load = np.sum(traffic_matrix) / (num_nodes * (num_nodes - 1))
        if load > 0.85:
            penalty = np.exp(-10 * (load - 0.85))
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
                              spray_avg_hops: float = None) -> Tuple[float, float, float]:
    """
    Apply architecture-specific capacity model.

    For Shale, when congestion_factor (u_max) is provided the throughput
    scales as r / u_max instead of the old r / (h+1).  spray_avg_hops
    drives bandwidth-tax and latency from actual spray paths.

    Returns: (effective_rate, bandwidth_tax, latency)
    """
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
        # 92% Bulk (1-hop) + 8% Short (multi-hop via expander)
        alpha_split = params.opera_alpha
        delta = params.opera_delta
        t_cycle = params.opera_t_cycle
        
        reconfig_eff = 1 - (delta / t_cycle)
        bulk_rate = alpha_split * rate * reconfig_eff
        short_rate = (1 - alpha_split) * rate / max(1, hops)
        effective_rate = bulk_rate + short_rate
        
        bw_tax = (1 - alpha_split) * (hops - 1) / max(1, hops)  # Tax for short flows
        latency = hops * delta
        
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
    
    return effective_rate, bw_tax, latency


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
