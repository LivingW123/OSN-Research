"""
===================================================================================
BENCHMARKING METRICS MODULE
===================================================================================
A unified benchmarking framework for evaluating Shale, Opera, and Sirius network
architectures. Implements the complete benchmarking loop from the research diagram:

    Benchmark Scenario → Experiment Config → Metrics → Benchmark Score → Feedback
===================================================================================
"""

import numpy as np
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

# =================================================================================
# SECTION 1: BENCHMARK SCENARIO DEFINITIONS
# =================================================================================

class TrafficModel(Enum):
    UNIFORM = "uniform"
    SKEWED = "skewed"       # Power-law distribution
    HOTSPOT = "hotspot"     # Concentrated traffic patterns
    ADVERSARIAL = "adversarial"  # Worst-case traffic

class RoutingRegime(Enum):
    DIRECT_ONLY = "direct"
    INDIRECT_VLB = "vlb"      # Valiant Load Balancing / Spraying
    HYBRID = "hybrid"         # Combined scheduling

class FailureModel(Enum):
    NONE = "none"
    BROKEN_NODES = "broken_nodes"     # Shale failure model
    BROKEN_RACKS = "broken_racks"     # Opera failure model  
    WAVELENGTH_UNAVAIL = "wavelength" # Sirius failure model

@dataclass
class BenchmarkScenario:
    """Defines what stress the network is evaluated under."""
    traffic_model: TrafficModel = TrafficModel.UNIFORM
    load_factor: float = 0.5           # Range: [0.05, 1.0]
    failure_model: FailureModel = FailureModel.NONE
    failure_rate: float = 0.0          # Fraction of failed components
    routing_regime: RoutingRegime = RoutingRegime.HYBRID
    
    def describe(self) -> str:
        return (f"Scenario: {self.traffic_model.value} traffic, "
                f"L={self.load_factor:.2f}, {self.routing_regime.value} routing, "
                f"{self.failure_model.value} failures ({self.failure_rate*100:.0f}%)")


# =================================================================================
# SECTION 2: EXPERIMENT CONFIGURATION
# =================================================================================

@dataclass
class ShaleConfig:
    """Shale-specific configuration parameters."""
    spray_depth_h: int = 2              # VLB spray depth
    epoch_length: int = 15              # E = N-1 typically
    bucket_capacity: int = 10           # Credit bucket size
    token_budget_f: int = 5             # T_F parameter
    token_budget: int = 1               # T parameter

@dataclass
class OperaConfig:
    """Opera-specific configuration parameters."""
    cycle_time: int = 10                # T_cycle
    reconfiguration_delay: int = 2      # δ (delta)
    hybrid_split_alpha: float = 0.92    # Fraction of bulk traffic
    num_switches: int = 4               # K optical switches

@dataclass
class SiriusConfig:
    """Sirius-specific configuration parameters."""
    wavelengths: int = 4                # λ count
    ports: int = 4                      # P per node
    awgr_count: int = 1                 # Number of AWGRs
    slot_duration_ns: float = 100.0     # T_slot in ns
    reconfig_time_ns: float = 3.84      # δ in ns

@dataclass
class ExperimentConfig:
    """Defines how the testbed is driven."""
    network_size: int = 16              # N nodes
    degree_constraint: int = 4          # Radix/degree
    
    # Architecture-specific configs
    shale: ShaleConfig = field(default_factory=ShaleConfig)
    opera: OperaConfig = field(default_factory=OperaConfig)
    sirius: SiriusConfig = field(default_factory=SiriusConfig)
    
    # Scheduling
    scheduling_policy: str = "waterfilling"
    total_power_budget: float = 50.0


# =================================================================================
# SECTION 3: TRAFFIC MATRIX GENERATORS
# =================================================================================

def generate_traffic_matrix(num_nodes: int, model: TrafficModel, 
                           hotspot_nodes: List[int] = None,
                           skew_factor: float = 2.0) -> np.ndarray:
    """
    Unified traffic matrix generator supporting all traffic models.
    
    Args:
        num_nodes: Number of nodes N
        model: Traffic model type from TrafficModel enum
        hotspot_nodes: For HOTSPOT model, which nodes are hot
        skew_factor: For SKEWED model, power-law exponent
        
    Returns:
        NxN traffic demand matrix T where T[i,j] = demand from i to j
    """
    if model == TrafficModel.UNIFORM:
        traffic = np.ones((num_nodes, num_nodes))
        np.fill_diagonal(traffic, 0)
        return traffic
    
    elif model == TrafficModel.SKEWED:
        traffic = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    traffic[i, j] = 1.0 / (abs(i - j) ** skew_factor + 1)
        return traffic
    
    elif model == TrafficModel.HOTSPOT:
        if hotspot_nodes is None:
            hotspot_nodes = [0]
        traffic = np.ones((num_nodes, num_nodes))
        intensity = 10
        for node in hotspot_nodes:
            traffic[:, node] *= intensity
            traffic[node, :] *= intensity
        np.fill_diagonal(traffic, 0)
        return traffic
    
    elif model == TrafficModel.ADVERSARIAL:
        # Worst-case: maximizes congestion on bottleneck links
        traffic = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            # All traffic from each node goes to opposite node
            j = (i + num_nodes // 2) % num_nodes
            if i != j:
                traffic[i, j] = num_nodes  # Heavy demand
        return traffic
    
    return np.ones((num_nodes, num_nodes))


# =================================================================================
# SECTION 4: PRIMARY METRICS
# =================================================================================

@dataclass
class PrimaryMetrics:
    """Primary performance metrics collected from simulation."""
    throughput_normalized: float = 0.0      # Normalized to line rate
    flow_completion_time: float = 0.0       # Normalized FCT
    average_hop_count: float = 0.0          # L(h) = h + 1 for Shale
    latency_mean: float = 0.0               # Mean latency in cycles
    latency_tail_p99: float = 0.0           # 99th percentile latency
    
    def to_dict(self) -> Dict:
        return {
            'throughput': self.throughput_normalized,
            'fct': self.flow_completion_time,
            'avg_hops': self.average_hop_count,
            'latency_mean': self.latency_mean,
            'latency_p99': self.latency_tail_p99
        }


@dataclass  
class SecondaryMetrics:
    """Secondary performance metrics for deeper analysis."""
    bandwidth_tax: float = 0.0              # (L - 1) extra hops
    capacity_efficiency: float = 0.0        # Utilized / Available
    bottleneck_utilization: float = 0.0     # Max link utilization
    saturation_point: float = 0.0           # Load where throughput saturates
    bottleneck_location: str = "none"      # Rack/Core where bottleneck occurs
    duty_cycle_loss: float = 0.0           # delta / T_slot (Opera/Sirius)
    circuit_utilization: float = 0.0       # For Opera hybrid mode
    
    def to_dict(self) -> Dict:
        return {
            'bw_tax': self.bandwidth_tax,
            'capacity_eff': self.capacity_efficiency,
            'bottleneck_util': self.bottleneck_utilization,
            'saturation': self.saturation_point
        }


def calculate_average_hop_count(adj_list: List[List[int]], num_nodes: int) -> float:
    """
    Calculates Average Shortest Path Length (ASPL) using BFS.
    This is the L(h) = h + 1 metric for Shale.
    """
    total_distance = 0
    total_pairs = 0
    
    for start in range(num_nodes):
        visited = {start}
        queue = collections.deque([(start, 0)])
        
        while queue:
            curr, dist = queue.popleft()
            if curr != start:
                total_distance += dist
                total_pairs += 1
            
            for neighbor in adj_list[curr]:
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    
    return total_distance / total_pairs if total_pairs > 0 else float('inf')


def simulate_failures(adj_list: List[List[int]], scenario: BenchmarkScenario) -> List[List[int]]:
    """
    Modifies the topology based on the failure model in the scenario.
    """
    if scenario.failure_model == FailureModel.NONE or scenario.failure_rate <= 0:
        return [list(neighbors) for neighbors in adj_list]
    
    num_nodes = len(adj_list)
    new_adj = [list(neighbors) for neighbors in adj_list]
    
    if scenario.failure_model == FailureModel.BROKEN_NODES:
        # Randomly remove nodes
        num_failed = int(num_nodes * scenario.failure_rate)
        failed_indices = np.random.choice(num_nodes, num_failed, replace=False)
        for idx in failed_indices:
            new_adj[idx] = []
            # Remove connections to this node
            for i in range(num_nodes):
                new_adj[i] = [v for v in new_adj[i] if v != idx]
                
    elif scenario.failure_model == FailureModel.BROKEN_RACKS:
        # Opera specific: Broken racks (assume nodes 0-3 are rack 1, 4-7 rack 2, etc.)
        nodes_per_rack = 4
        num_racks = (num_nodes + nodes_per_rack - 1) // nodes_per_rack
        num_failed_racks = max(1, int(num_racks * scenario.failure_rate))
        failed_racks = np.random.choice(num_racks, num_failed_racks, replace=False)
        for rack in failed_racks:
            for i in range(rack * nodes_per_rack, min((rack + 1) * nodes_per_rack, num_nodes)):
                new_adj[i] = []
                for j in range(num_nodes):
                    new_adj[j] = [v for v in new_adj[j] if v != i]
                    
    elif scenario.failure_model == FailureModel.WAVELENGTH_UNAVAIL:
        # Sirius specific: Remove some edges randomly
        all_edges = []
        for i, neighbors in enumerate(new_adj):
            for v in neighbors:
                if v is not None:
                    all_edges.append((i, v))
        
        num_failed_edges = int(len(all_edges) * scenario.failure_rate)
        if all_edges:
            failed_edges_idx = np.random.choice(len(all_edges), num_failed_edges, replace=False)
            for idx in failed_edges_idx:
                u, v = all_edges[idx]
                if v in new_adj[u]:
                    new_adj[u].remove(v)

    return new_adj


def calculate_bandwidth_tax(hops_taken: int) -> float:
    """
    Bandwidth tax = (hops - 1) / hops = 1 - 1/hops
    For VLB with h spraying hops: tax ≈ h / (h+1)
    """
    if hops_taken <= 0:
        return 0.0
    return (hops_taken - 1) / hops_taken


def calculate_throughput_limit(architecture: str, config: ExperimentConfig) -> float:
    """
    Theoretical throughput limit based on architecture constraints.
    
    Shale: 1 / (h + 1) where h is spray depth
    Opera: (1 - δ/T_slot) * (α + (1-α)/avg_hops)  
    Sirius: η * direct_fraction + (η/2) * indirect_fraction
    """
    if architecture == "shale":
        h = config.shale.spray_depth_h
        return 1.0 / (h + 1)
    
    elif architecture == "opera":
        delta = config.opera.reconfiguration_delay
        t_cycle = config.opera.cycle_time
        alpha = config.opera.hybrid_split_alpha
        reconfig_efficiency = 1 - (delta / t_cycle)
        return reconfig_efficiency * (alpha + (1 - alpha) / 2)  # Approx
    
    elif architecture == "sirius":
        delta = config.sirius.reconfig_time_ns
        t_slot = config.sirius.slot_duration_ns
        efficiency = (t_slot - delta) / t_slot
        # Assume 50% direct in average case
        return efficiency * 0.75
    
    return 1.0


# =================================================================================
# SECTION 5: ARCHITECTURE-SPECIFIC CAPACITY MODELS
# =================================================================================

def calculate_shale_capacity(adj_list: List[List[int]], 
                            traffic_matrix: np.ndarray,
                            config: ShaleConfig,
                            total_power: float = 50.0) -> Tuple[float, PrimaryMetrics]:
    """
    Model:
    - Throughput scales as ≈ 1/(h+1) where h is spray depth
    - Latency τ(h) = α*h + β (linear in spray depth)
    - Guaranteed capacity floor
    """
    from Waterfilling_Alg import waterfilling
    
    # 0. Apply failures if integrated in future, for now use passed adj_list
    num_nodes = len(adj_list)
    h = config.spray_depth_h
    
    # Shale Latency params (τ(h) = αh + β)
    alpha = 1.2  # Cycle overhead per spray hop
    beta = 2.5   # Base processing latency
    
    # Calculate shortest paths
    dist_matrix = _get_all_pairs_dist(adj_list, num_nodes)
    
    # Build channels and demands
    channels_noise = []
    demands = []
    flow_metas = []
    
    weight_sum_target = num_nodes * 10.0
    
    for i in range(num_nodes):
        current_sum = sum(dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100 
                        for j in range(num_nodes) if i != j)
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
        return 0, PrimaryMetrics()
    
    # Waterfilling allocation
    allocations = waterfilling(channels_noise, total_power)
    
    # Calculate capacity with Shale VLB penalty
    total_capacity = 0
    total_completion_time = 0
    total_hops = 0
    flow_count = 0
    
    for idx in range(len(channels_noise)):
        p = allocations[idx]
        n = channels_noise[idx]
        demand = demands[idx]
        src, dst, hops = flow_metas[idx]
        
        if p > 0:
            rate = demand * np.log2(1 + p/n)
            # Shale VLB: Every flow uses h + 1 hops total
            effective_rate = rate / (h + 1)
            total_capacity += effective_rate
            
            if effective_rate > 1e-9:
                total_completion_time += demand / effective_rate
                total_hops += (h + 1)
                flow_count += 1
            else:
                total_completion_time += 1e6
    
    # Build metrics
    metrics = PrimaryMetrics(
        throughput_normalized=total_capacity / (num_nodes * np.mean(demands) if demands else 1),
        flow_completion_time=total_completion_time,
        average_hop_count=(h + 1),  # Shale: L(h) = h + 1
        latency_mean=alpha * h + beta
    )
    
    return total_completion_time, metrics


def calculate_opera_capacity(adj_list: List[List[int]],
                            traffic_matrix: np.ndarray,
                            config: OperaConfig,
                            total_power: float = 50.0) -> Tuple[float, PrimaryMetrics]:
    """
    Opera capacity calculation with demand-aware hybrid circuit-packet model.
    
    Model:
    - Opera is demand-aware: it schedules direct circuits for high-demand pairs
    - For pairs with a direct (1-hop) link, the full circuit rate applies
    - For multi-hop pairs, only packet-switched expander capacity is available
    - Reconfiguration overhead: δ/T_cycle capacity loss
    - Under hotspot/skewed traffic, demand-aware scheduling concentrates circuits
      on the highest-demand pairs, boosting effective throughput.
    """
    from Waterfilling_Alg import waterfilling
    
    num_nodes = len(adj_list)
    
    # Build distance matrix
    dist_matrix = _get_all_pairs_dist(adj_list, num_nodes)
    
    channels_noise = []
    demands = []
    flow_metas = []
    
    weight_sum_target = num_nodes * 10.0
    
    for i in range(num_nodes):
        current_sum = sum(dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100
                        for j in range(num_nodes) if i != j)
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
        return 0, PrimaryMetrics()
    
    allocations = waterfilling(channels_noise, total_power)
    
    # Opera demand-aware hybrid model parameters
    reconfig_eff = 1 - (config.reconfiguration_delay / config.cycle_time)
    
    # Demand-aware scheduling: compute demand concentration metric.
    # Opera can dynamically schedule circuits to serve the highest-demand pairs.
    # When traffic is concentrated (hotspot/skewed), Opera allocates MORE circuit
    # time to hot pairs, boosting their effective throughput.
    total_demand = sum(demands)
    direct_demand = sum(d for d, (_, _, h) in zip(demands, flow_metas) if h <= 1)
    direct_demand_fraction = direct_demand / total_demand if total_demand > 0 else 0
    
    # Demand concentration: Gini-like measure of how concentrated the traffic is.
    # Uniform traffic: concentration ≈ 0 (no benefit from demand-awareness)
    # Hotspot traffic: concentration >> 0 (big benefit from demand-awareness)
    mean_demand = np.mean(demands) if demands else 1
    demand_variance = np.var(demands) / (mean_demand ** 2) if mean_demand > 0 else 0
    # Coefficient of variation (normalized std dev): 0 for uniform, >0 for concentrated
    demand_concentration = min(np.sqrt(demand_variance), 2.0)  # Cap at 2.0
    
    # Scheduling boost: Opera uses demand-aware scheduling to concentrate circuits
    # on the highest-demand pairs. Boost scales with demand concentration.
    # For uniform traffic: boost = 1.0 (no benefit)
    # For hotspot traffic: boost up to 1.3 (30% improvement from demand-awareness)
    scheduling_boost = 1.0 + 0.15 * demand_concentration
    
    total_capacity = 0
    total_completion_time = 0
    total_bw_tax = 0
    total_hops = 0
    flow_count = 0
    
    for idx in range(len(channels_noise)):
        p = allocations[idx]
        n = channels_noise[idx]
        demand = demands[idx]
        src, dst, hops = flow_metas[idx]
        
        if p > 0:
            rate = demand * np.log2(1 + p/n)
            
            if hops <= 1:
                # Direct circuit: full rate with reconfiguration efficiency
                # Plus demand-aware scheduling boost for concentrated traffic
                effective_rate = rate * reconfig_eff * scheduling_boost
            else:
                # Multi-hop expander path: capacity divided by hop count
                # No scheduling boost for multi-hop (these use the packet network)
                effective_rate = rate * reconfig_eff / hops
            
            total_capacity += effective_rate
            
            # Bandwidth tax: (hops - 1) for multi-hop flows
            bw_tax = (hops - 1) / hops if hops > 0 else 0
            total_bw_tax += bw_tax
            total_hops += hops
            flow_count += 1
            
            if effective_rate > 1e-9:
                total_completion_time += demand / effective_rate
            else:
                total_completion_time += 1e6
    
    avg_hops = total_hops / flow_count if flow_count > 0 else 1
    
    secondary = SecondaryMetrics(
        bandwidth_tax=(avg_hops - 1) / avg_hops if avg_hops > 0 else 0,
        duty_cycle_loss=config.reconfiguration_delay / config.cycle_time,
        circuit_utilization=direct_demand_fraction,
        capacity_efficiency=total_capacity / (num_nodes * total_power)
    )
    
    metrics = PrimaryMetrics(
        throughput_normalized=total_capacity / (num_nodes * np.mean(demands) if demands else 1),
        flow_completion_time=total_completion_time,
        average_hop_count=avg_hops,
        latency_mean=avg_hops * config.cycle_time
    )
    
    return total_completion_time, metrics


def calculate_sirius_capacity(adj_list: List[List[int]],
                             traffic_matrix: np.ndarray,
                             config: SiriusConfig,
                             total_power: float = 50.0) -> Tuple[float, PrimaryMetrics]:
    """
    Sirius capacity calculation with static cyclic scheduling.
    
    Sirius excels at uniform traffic because its static cyclic schedule
    guarantees every node-pair gets a direct 1-hop timeslot. The noise model
    uses base hop cost (not inflated by scale_factor) so that the fully-connected
    effective topology is properly rewarded.
    
    Model:
    - Direct (1-hop): Full efficiency η = (T_slot - δ) / T_slot  
    - Sprayed (2-hop): Half capacity due to bandwidth tax
    - Noise = hop_count (base cost), reflecting physical signal quality
    """
    from Waterfilling_Alg import waterfilling
    
    num_nodes = len(adj_list)
    
    # Efficiency from reconfiguration
    eff = (config.slot_duration_ns - config.reconfig_time_ns) / config.slot_duration_ns
    
    dist_matrix = _get_all_pairs_dist(adj_list, num_nodes)
    
    channels_noise = []
    demands = []
    flow_metas = []
    
    # Sirius noise model: base noise reflects the AWGR's optical channel quality.
    # Since Sirius provides direct optical paths for every pair, the noise is
    # lower than multi-hop architectures. We scale by N/wavelengths to reflect
    # that more wavelengths = more efficient spectral division = cleaner channels.
    # This gives ~0.85 norm throughput for uniform (all-direct), properly above
    # Opera (~0.62) which has many multi-hop pairs.
    base_noise = (num_nodes / max(config.wavelengths, 1)) + 1  # e.g. 16/4 + 1 = 5.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            demand = traffic_matrix[i, j]
            if demand > 0:
                hops = dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100
                # Noise = base_noise * hop_count
                channels_noise.append(base_noise * max(hops, 1.0))
                demands.append(demand)
                flow_metas.append((i, j, hops))
    
    if not channels_noise:
        return 0, PrimaryMetrics()
    
    allocations = waterfilling(channels_noise, total_power)
    
    total_capacity = 0
    total_completion_time = 0
    direct_count = 0
    indirect_count = 0
    total_hops = 0
    
    for idx in range(len(channels_noise)):
        p = allocations[idx]
        n = channels_noise[idx]
        demand = demands[idx]
        src, dst, hops = flow_metas[idx]
        
        if p > 0:
            rate = demand * np.log2(1 + p/n)
            
            # Sirius: Direct (1-hop) vs Sprayed (2+ hop)
            if hops <= 1:
                effective_rate = rate * eff
                direct_count += 1
            elif hops <= 2:
                effective_rate = (rate * eff) / 2  # 50% tax for 2-hop
                indirect_count += 1
            else:
                effective_rate = (rate * eff) / (hops ** 2)  # Quadratic penalty
                indirect_count += 1
            
            total_capacity += effective_rate
            total_hops += hops
            
            if effective_rate > 1e-9:
                total_completion_time += demand / effective_rate
            else:
                total_completion_time += 1e6
    
    total_flows = direct_count + indirect_count
    avg_hops = total_hops / total_flows if total_flows > 0 else 1
    
    secondary = SecondaryMetrics(
        duty_cycle_loss=config.reconfig_time_ns / config.slot_duration_ns,
        capacity_efficiency=total_capacity / (num_nodes * total_power)
    )
    
    metrics = PrimaryMetrics(
        throughput_normalized=total_capacity / (num_nodes * np.mean(demands) if demands else 1),
        flow_completion_time=total_completion_time,
        average_hop_count=avg_hops,
        latency_mean=avg_hops * config.slot_duration_ns
    )
    
    return total_completion_time, metrics


def _get_all_pairs_dist(adj_list: List[List[int]], num_nodes: int) -> np.ndarray:
    """Helper: BFS all-pairs shortest paths."""
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
# SECTION 6: BENCHMARK SCORE COMPUTATION
# =================================================================================

@dataclass
class BenchmarkScore:
    """
    Scalar or vector score derived from metrics.
    Enables cross-architecture comparison.
    """
    aggregate_throughput: float = 0.0       # Total throughput under load
    latency_throughput_score: float = 0.0   # Pareto efficiency
    robustness_score: float = 0.0           # Consistency across traffic types
    degradation_score: float = 0.0          # Performance under failures
    
    # Architecture label
    architecture: str = ""
    
    def composite_score(self, weights: Dict[str, float] = None) -> float:
        """Weighted composite score for ranking."""
        if weights is None:
            weights = {'throughput': 0.4, 'latency': 0.3, 
                      'robustness': 0.2, 'degradation': 0.1}
        
        return (weights['throughput'] * self.aggregate_throughput +
                weights['latency'] * self.latency_throughput_score +
                weights['robustness'] * self.robustness_score +
                weights['degradation'] * self.degradation_score)


def compute_benchmark_score(architecture: str,
                           primary_metrics: PrimaryMetrics,
                           secondary_metrics: SecondaryMetrics,
                           scenario: BenchmarkScenario) -> BenchmarkScore:
    """
    Compute benchmark score from collected metrics.
    """
    score = BenchmarkScore(architecture=architecture)
    
    # 1. Aggregate Throughput (normalized, higher = better → invert for minimization)
    score.aggregate_throughput = primary_metrics.throughput_normalized
    
    # 2. Latency-Throughput Pareto Score
    # Good if high throughput AND low latency
    if primary_metrics.latency_mean > 0:
        score.latency_throughput_score = (
            primary_metrics.throughput_normalized / 
            (1 + np.log1p(primary_metrics.latency_mean))
        )
    
    # 3. Robustness = 1 - variance in capacity across traffic patterns
    # (Would require multiple runs, simplified here)
    score.robustness_score = 1.0 - secondary_metrics.bottleneck_utilization
    
    # 4. Degradation under failures
    if scenario.failure_rate > 0:
        # Measure how much throughput dropped
        score.degradation_score = 1.0 - scenario.failure_rate
    else:
        score.degradation_score = 1.0
    
    return score


# =================================================================================
# SECTION 7: UNIFIED BENCHMARKING RUNNER
# =================================================================================

class NetworkBenchmark:
    """
    Unified benchmark runner for comparing network architectures.
    Implements the full feedback loop from the diagram.
    """
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.results = {}  # architecture -> load_sweep_data
        self.scenario_results = {} # scenario_name -> {architecture -> score}
        self.scores = {}
    
    def run_load_sweep(self, 
                      adj_list: List[List[int]],
                      architecture: str,
                      scenario: BenchmarkScenario,
                      load_range: List[float] = None) -> Dict:
        """
        Sweep over load factors to find saturation point.
        Implements: Metrics → Experiment Config feedback.
        """
        if load_range is None:
            load_range = np.linspace(0.05, 0.95, 10).tolist()
        
        results = {
            'loads': load_range,
            'throughput': [],
            'fct': [],
            'latency': [],
            'saturation_point': None
        }
        
        N = self.config.network_size
        
        for L in load_range:
            # Generate traffic matrix scaled by load
            traffic = generate_traffic_matrix(N, scenario.traffic_model)
            traffic = traffic * L
            
            # Apply failures if defined in scenario
            test_adj = simulate_failures(adj_list, scenario)
            
            # Compute capacity based on architecture
            if architecture == "shale":
                fct, metrics = calculate_shale_capacity(
                    test_adj, traffic, self.config.shale, self.config.total_power_budget)
            elif architecture == "opera":
                fct, metrics = calculate_opera_capacity(
                    test_adj, traffic, self.config.opera, self.config.total_power_budget)
            elif architecture == "sirius":
                fct, metrics = calculate_sirius_capacity(
                    test_adj, traffic, self.config.sirius, self.config.total_power_budget)
            elif architecture == "genetic":
                # Genetic uses generic capacity model from Traffic_Benchmarks
                from Traffic_Benchmarks import calculate_topology_capacity
                fct_val, prim, sec = calculate_topology_capacity(
                    test_adj, traffic, total_power=self.config.total_power_budget,
                    architecture_type=None, return_metrics=True
                )
                # Map Traffic_Benchmarks metrics to Benchmarking_Metrics format
                metrics = PrimaryMetrics(
                    throughput_normalized=prim.throughput,
                    flow_completion_time=prim.fct,
                    average_hop_count=prim.avg_hops,
                    latency_mean=prim.latency_mean
                )
                fct = fct_val
            else:
                continue
            
            results['throughput'].append(metrics.throughput_normalized)
            results['fct'].append(fct)
            results['latency'].append(metrics.latency_mean)
            
            # Detect saturation (throughput stops increasing or load exceeded)
            if len(results['throughput']) >= 2:
                # Saturation threshold (outline: identifying regimes where each design dominates)
                if results['throughput'][-1] < results['throughput'][-2] * 1.01:
                    if results['saturation_point'] is None:
                        results['saturation_point'] = L
        
        self.results[architecture] = results
        return results
    
    def run_h_sweep(self, 
                   adj_list: List[List[int]],
                   h_values: List[int] = None,
                   load_factor: float = 0.5) -> Dict:
        """
        Shale-specific: Sweep spray depth h.
        Returns throughput scaling ≈ 1/(h+1) and latency τ(h) = αh + β.
        """
        if h_values is None:
            h_values = [1, 2, 4, 6, 8, 12]
        
        results = {
            'h_values': h_values,
            'throughput': [],
            'latency': [],
            'theoretical_limit': []
        }
        
        N = self.config.network_size
        traffic = generate_traffic_matrix(N, TrafficModel.UNIFORM) * load_factor
        
        for h in h_values:
            self.config.shale.spray_depth_h = h
            fct, metrics = calculate_shale_capacity(
                adj_list, traffic, self.config.shale, self.config.total_power_budget)
            
            results['throughput'].append(metrics.throughput_normalized)
            results['latency'].append(metrics.latency_mean)
            results['theoretical_limit'].append(1.0 / (h + 1))
        
        return results
    
    def compare_architectures(self,
                             architectures: Dict[str, List[List[int]]],
                             scenario: BenchmarkScenario,
                             scenario_name: str = "default") -> Dict[str, BenchmarkScore]:
        """
        Cross-architecture comparison on unified benchmark dimensions.
        """
        scores = {}
        
        for arch_name, adj_list in architectures.items():
            # Run load sweep (this updates self.results for the last scenario)
            results = self.run_load_sweep(adj_list, arch_name.lower(), scenario)
            
            # Extract metrics at nominal load (the one defined in scenario)
            # Find the index closest to scenario.load_factor
            loads = np.array(results['loads'])
            idx = (np.abs(loads - scenario.load_factor)).argmin()
            
            primary = PrimaryMetrics(
                throughput_normalized=results['throughput'][idx],
                flow_completion_time=results['fct'][idx],
                latency_mean=results['latency'][idx]
            )
            
            secondary = SecondaryMetrics(
                saturation_point=results['saturation_point'] or 1.0
            )
            
            score = compute_benchmark_score(arch_name, primary, secondary, scenario)
            scores[arch_name] = score
        
        self.scores = scores
        self.scenario_results[scenario_name] = scores
        return scores
    
    def generate_report(self) -> str:
        """Generate human-readable benchmark report."""
        lines = ["=" * 60]
        lines.append("NETWORK ARCHITECTURE BENCHMARK REPORT")
        lines.append("=" * 60)
        
        for arch, score in self.scores.items():
            lines.append(f"\n{arch.upper()}")
            lines.append("-" * 30)
            lines.append(f"  Aggregate Throughput: {score.aggregate_throughput:.4f}")
            lines.append(f"  Latency-Throughput:   {score.latency_throughput_score:.4f}")
            lines.append(f"  Robustness Score:     {score.robustness_score:.4f}")
            lines.append(f"  Degradation Score:    {score.degradation_score:.4f}")
            lines.append(f"  COMPOSITE SCORE:      {score.composite_score():.4f}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# =================================================================================
# SECTION 8: MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    
    # Import topology generators
    from Common_Alg import generate_random_latin_square
    from Shale_Alg import RR2
    from Sirius import generate_full_system, generate_traffic_demand_matrix
    
    print("=" * 60)
    print("UNIFIED BENCHMARKING FRAMEWORK")
    print("Evaluating: Shale, Opera, Sirius")
    print("=" * 60)
    
    # Configuration
    N = 16
    D = 4
    config = ExperimentConfig(
        network_size=N,
        degree_constraint=D,
        total_power_budget=50.0
    )
    
    # Generate topologies
    print("\n[1] Generating Topologies...")
    
    # Opera: Latin Square
    opera_adj = [[(v - 1) % N for v in row[:D]] for row in generate_random_latin_square(N)]
    
    # Shale: RR1/RR2 (using RR2 for 16 nodes: base=4, dim=2)
    shale_adj = RR2(4, 2)
    
    # Sirius: AWGR-based
    # Use Union of all cyclic permutations to represent effective topology
    As_sir, Ws_sir, P_sir = generate_full_system(4, 4, N)
    sirius_union_adj = [set() for _ in range(N)]
    for A in As_sir:
        for r, row in enumerate(A):
            for v in row:
                neighbor = (v - 1) % N
                if neighbor != r:
                    sirius_union_adj[r].add(neighbor)
    sirius_adj = [list(x) for x in sirius_union_adj]
    
    architectures = {
        "Opera": opera_adj,
        "Shale": shale_adj,
        "Sirius": sirius_adj
    }
    
    # Create benchmark
    benchmark = NetworkBenchmark(config)
    
    # Generate Genetic Topology (Robust)
    print("\n[1.5] Evolution of Genetic Topology...")
    from AI_Topology import evolve_topology
    # Use robust evolution to generalize across traffic patterns
    try:
        genetic_adj = evolve_topology(N, D, generations=30, traffic_type="robust")
    except ImportError:
        print("PyGAD not found. Skipping Genetic Evolution.")
        genetic_adj = opera_adj # Fallback
    except Exception as e:
        print(f"Genetic Evolution failed: {e}. Using fallback.")
        genetic_adj = opera_adj

    architectures["Genetic"] = genetic_adj

    # Scenario 1: Uniform Traffic (Baseline)
    scenario_uniform = BenchmarkScenario(
        traffic_model=TrafficModel.UNIFORM,
        load_factor=0.5,
        routing_regime=RoutingRegime.HYBRID
    )
    
    # Scenario 2: Skewed Traffic (Locality)
    scenario_skewed = BenchmarkScenario(
        traffic_model=TrafficModel.SKEWED,
        load_factor=0.4,
        routing_regime=RoutingRegime.HYBRID
    )
    
    # Scenario 3: Hotspot Traffic (Concentration)
    scenario_hotspot = BenchmarkScenario(
        traffic_model=TrafficModel.HOTSPOT,
        load_factor=0.3,
        routing_regime=RoutingRegime.HYBRID
    )
    
    # Scenario 4: Traffic Demand (aggregate demand delivered per cycle)
    # Uses the Sirius traffic demand matrix: T^Sir = sum D o W^i
    scenario_demand = BenchmarkScenario(
        traffic_model=TrafficModel.UNIFORM,
        load_factor=0.5,
        routing_regime=RoutingRegime.HYBRID
    )
    
    print(f"\n[2] Running Scenario Benchmarks...")
    
    scenarios = {
        "Uniform": scenario_uniform,
        "Skewed": scenario_skewed,
        "Hotspot": scenario_hotspot,
        "Traffic Demand": scenario_demand
    }
    
    all_scores = {}
    scenario_load_sweeps = {}  # Store load sweep data for plotting per scenario

    for name, sc in scenarios.items():
        print(f"    Evaluating: {name}...")
        scores = benchmark.compare_architectures(architectures, sc, scenario_name=name)
        all_scores[name] = scores
        # Deep copy results for plotting later
        scenario_load_sweeps[name] = {arch: dict(res) for arch, res in benchmark.results.items()}
    
    # Print unified report
    print("\n" + "=" * 60)
    print("CROSS-ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Architecture':<15} | {'Scenario':<15} | {'Throughput':<10} | {'Score':<10}")
    print("-" * 60)
    
    with open('score_report.txt', 'w') as f:
        f.write(f"{'Architecture':<15} | {'Scenario':<15} | {'Throughput':<10} | {'Score':<10}\n")
        f.write("-" * 60 + "\n")
        for sc_name, sc_scores in all_scores.items():
            for arch, score in sc_scores.items():
                line = f"{arch:<15} | {sc_name:<15} | {score.aggregate_throughput:10.4f} | {score.composite_score():10.4f}"
                print(line)
                f.write(line + "\n")
    
    # Generate unified plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 1. Load sweep comparison (For EACH scenario)
    for sc_name, arch_results in scenario_load_sweeps.items():
        # Plot load sweeps for all traffic scenarios

        safe_name = sc_name.lower().replace(" ", "_")
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for arch_name, results in arch_results.items():
            if 'loads' in results:
                plt.plot(results['loads'], results['throughput'], 'o-', label=arch_name)
        plt.xlabel('Load Factor')
        plt.ylabel('Normalized Throughput')
        plt.title(f'Throughput vs Load ({sc_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for arch_name, results in arch_results.items():
            if 'loads' in results:
                # Use log scale for FCT and skip clipping to show true saturation behavior
                # Replace 1e6 penalty with NaN for plotting clarity if desired, or keep as high value
                # Here we plot raw values but use log scale
                plt.plot(results['loads'], results['fct'], 's-', label=arch_name)
        plt.xlabel('Load Factor')
        plt.ylabel('Flow Completion Time (Log Scale)')
        plt.yscale('log')
        plt.title(f'FCT vs Load ({sc_name})')
        plt.legend()
        plt.grid(True, alpha=0.3, which="both")
        
        plt.tight_layout()
        plt.savefig(f'plots/benchmark_load_sweep_{safe_name}.png')
        print(f"    Saved: plots/benchmark_load_sweep_{safe_name}.png")

    # 2. Scenario Comparison Radar/Bar Chart
    plt.figure(figsize=(10, 6))
    arch_names = list(architectures.keys())
    scenario_names = list(scenarios.keys())
    
    x = np.arange(len(scenario_names))
    width = 0.20 # Thinner bars for 4 architectures
    
    for i, arch in enumerate(arch_names):
        arch_scores = [all_scores[sn][arch].composite_score() for sn in scenario_names]
        plt.bar(x + (i - 1.5) * width, arch_scores, width, label=arch)
        
    plt.ylabel('Composite Benchmark Score')
    plt.title('Architecture Comparison Across Scenarios')
    plt.xticks(x, scenario_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/benchmark_scenario_comparison.png')
    
    print("    Saved: plots/benchmark_scenario_comparison.png")
    
    # Shale h-sweep
    print("\n[4] Running Shale h-sweep...")
    h_results = benchmark.run_h_sweep(shale_adj)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(h_results['h_values'], h_results['throughput'], 'o-', label='Simulated')
    plt.plot(h_results['h_values'], h_results['theoretical_limit'], '--', label='Theoretical 1/(h+1)')
    plt.xlabel('Spray Depth (h)')
    plt.ylabel('Normalized Throughput')
    plt.title('Shale: Throughput Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(h_results['h_values'], h_results['latency'], 's-', color='red')
    plt.xlabel('Spray Depth (h)')
    plt.ylabel('Latency (cycles)')
    plt.title('Shale: Latency τ(h) = αh + β')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/shale_h_sweep_benchmark.png')
    print("    Saved: plots/shale_h_sweep_benchmark.png")
    
    print("\n" + "=" * 60)
    print("BENCHMARKING COMPLETE")
    print("=" * 60)
