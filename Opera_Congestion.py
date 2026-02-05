"""
===================================================================================
OPERA CONGESTION SIMULATION
===================================================================================
Implements Opera architecture simulation with hybrid circuit-packet model.

Benchmarking Framework Section 2.2: Opera Benchmark Outline
- Hybrid circuit-packet workload (92% bulk + 8% latency-sensitive)
- Dynamic rack failures
- Bandwidth tax: (L - 1) for multi-hop flows
- Reconfiguration overhead: δ / T_cycle capacity loss
===================================================================================
"""

import heapq
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# =================================================================================
# SECTION 1: OPERA METRICS (Framework Section 2.2)
# =================================================================================

@dataclass
class OperaMetrics:
    """
    Opera-specific metrics from Benchmarking Framework Section 2.2.
    
    Primary:
    - throughput: Effective throughput under hybrid waterfilling
    - avg_latency: Average latency in cycles
    - fct: Flow Completion Time
    
    Secondary:
    - bandwidth_tax: (L - 1) extra bandwidth cost
    - circuit_utilization: Fraction of circuit capacity used
    - duty_cycle_loss: δ / T_slot capacity loss from reconfiguration
    """
    throughput: float = 0.0
    avg_latency: float = 0.0
    fct: float = 0.0
    bandwidth_tax: float = 0.0
    circuit_utilization: float = 0.0
    duty_cycle_loss: float = 0.0


def calculate_opera_bandwidth_tax(hops: int) -> float:
    """
    Opera bandwidth tax calculation.
    Tax = (hops - 1) / hops for multi-hop flows.
    Bulk (1-hop) flows have 0 tax.
    """
    if hops <= 1:
        return 0.0
    return (hops - 1) / hops


def calculate_opera_duty_cycle_loss(delta: float, t_slot: float) -> float:
    """
    Duty cycle loss from reconfiguration.
    Loss = δ / T_slot (fraction of slot lost to switching).
    """
    if t_slot <= 0:
        return 0.0
    return delta / t_slot


def calculate_opera_theoretical_throughput(alpha: float, delta: float, t_cycle: float, avg_hops: float = 2.0) -> float:
    """
    Theoretical throughput for Opera hybrid model.
    
    Throughput = (1 - δ/T_cycle) * (α + (1-α)/avg_hops)
    where α = bulk fraction, δ = reconfig delay
    """
    reconfig_efficiency = 1 - (delta / t_cycle) if t_cycle > 0 else 0
    return reconfig_efficiency * (alpha + (1 - alpha) / avg_hops)




class OperaPacket:
    def __init__(self, src, dst, size, creation_time, is_bulk=False):
        self.src = src
        self.dst = dst
        self.size = size # in cells/packets
        self.creation_time = creation_time
        self.is_bulk = is_bulk
        self.hops_taken = 0
        self.path = [src]
        self.delivery_time = None

class OperaSimulation:
    def __init__(self, num_nodes, num_switches, reconfiguration_delay=2, slot_duration=10):
        self.N = num_nodes
        self.K = num_switches # Number of circuit switches
        self.reconfig_delay = reconfiguration_delay
        self.slot_duration = slot_duration
        self.cycle_time = (num_nodes - 1) * slot_duration
        
        # disjoint matching, shift the initial Latin Square for each switch
        base_square = self._generate_latin_square(num_nodes)
        self.schedules = []
        for k in range(num_switches):
            sched = np.roll(base_square, k, axis=0).tolist()
            self.schedules.append(sched)
            
        self.time = 0
        self.delivered_packets = []
        self.queues = [collections.deque() for _ in range(num_nodes)] # Multi-hop queues
        self.bulk_buffers = collections.defaultdict(list) # (src, dst) -> packets waiting for direct
        
        self.total_bytes_transmitted = 0
        self.bandwidth_tax_sum = 0
        
    def _generate_latin_square(self, n):
        row = np.arange(n)
        return np.array([np.roll(row, -i) for i in range(n)])

    def get_current_neighbor(self, switch_idx, node_idx):
        t_offset = (switch_idx * self.cycle_time) // self.K
        current_t = (self.time + t_offset) % self.cycle_time
        
        slot_idx = current_t // self.slot_duration
        intra_slot_t = current_t % self.slot_duration
        
        # Reconfiguration Delay: Switch is down at the beginning of each slot
        if intra_slot_t < self.reconfig_delay:
            return None
            
        return self.schedules[switch_idx][node_idx][slot_idx]

    def inject_traffic(self, src, dst, size, is_bulk=False):
        packet = OperaPacket(src, dst, size, self.time, is_bulk)
        if is_bulk:
            self.bulk_buffers[(src, dst)].append(packet)
        else:
            self.queues[src].append(packet)

    def get_shortest_path(self, src, dst):
        # BFS over currently active edges across all K switches
        # This represents the Expander property
        queue = collections.deque([(src, [])])
        visited = {src}
        
        # Build current graph
        adj = collections.defaultdict(list)
        for u in range(self.N):
            for k in range(self.K):
                v = self.get_current_neighbor(k, u)
                if v is not None and v != u:
                    adj[u].append(v)
        
        while queue:
            u, path = queue.popleft()
            if u == dst:
                return path
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    new_path = path + [v]
                    queue.append((v, new_path))
        return None # No path currently (shouldn't happen in expander)

    def run_step(self):
        # 1. Direct Path Processing (Bulk)
        for (src, dst), packets in list(self.bulk_buffers.items()):
            if not packets: continue
            
            direct_active = False
            for k in range(self.K):
                if self.get_current_neighbor(k, src) == dst:
                    direct_active = True
                    break
            
            if direct_active:
                pkt = packets.pop(0)
                pkt.hops_taken = 1
                pkt.delivery_time = self.time + 1
                self.delivered_packets.append(pkt)
                self.total_bytes_transmitted += pkt.size
                self.bandwidth_tax_sum += pkt.size # Tax = (1-1)*size = 0, so sum += hops*size
                
        # 2. Indirect Path Processing (Latency-Sensitive)
        # In Opera, indirect packets are sent immediately via the expander
        for src in range(self.N):
            if self.queues[src]:
                # Peek at the packet to find its destination
                pkt = self.queues[src][0]
                # Find shortest path in current expander
                path = self.get_shortest_path(src, pkt.dst)
                
                if path:
                    # One hop and one expander packet
                    next_hop = path[0]
                    pkt = self.queues[src].popleft()
                    pkt.hops_taken += 1
                    if next_hop == pkt.dst:
                        pkt.delivery_time = self.time + 1
                        self.delivered_packets.append(pkt)
                        self.total_bytes_transmitted += pkt.size
                        self.bandwidth_tax_sum += pkt.hops_taken * pkt.size
                    else:
                        pkt.path.append(next_hop)
                        self.queues[next_hop].append(pkt)
        
        self.time += 1

def run_opera_efficiency_test():
    N = 16
    K = 4
    # Reconfig delay 2, Slot 20 -> 10% capacity loss just from switching
    # Paper uses very small reconfig delay (micros), here we use 2/20 = 10%.
    
    loads = np.linspace(0.1, 0.9, 5)
    tax_results = []
    avg_latencies = []
    
    print("--- Running Opera Efficiency Simulation (BFS Routing) ---")
    for L in loads:
        sim = OperaSimulation(N, K, reconfiguration_delay=2, slot_duration=20) # 2/20 = 10% overhead
        duration = 500
        
        for t in range(duration):
            # 92% Bulk, 8% Latency-Sensitive
            if random.random() < L:
                src, dst = random.sample(range(N), 2)
                is_bulk = random.random() < 0.92
                sim.inject_traffic(src, dst, size=1, is_bulk=is_bulk)
            sim.run_step()
            
        if sim.total_bytes_transmitted > 0:
            avg_hops = sim.bandwidth_tax_sum / sim.total_bytes_transmitted
            eff_tax = (avg_hops - 1) * 100
            tax_results.append(eff_tax)
            
            lats = [p.delivery_time - p.creation_time for p in sim.delivered_packets]
            avg_latencies.append(np.mean(lats))
        else:
            tax_results.append(0)
            avg_latencies.append(0)
            
        print(f"Load {L:.1f}: Tax = {tax_results[-1]:.2f}%, Avg Latency = {avg_latencies[-1]:.2f}")

    # Plotting
    if not os.path.exists('plots'): os.makedirs('plots')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loads, tax_results, 'o-', color='blue')
    plt.axhline(y=8.4, color='red', linestyle='--', label='Paper Result (8.4%)')
    plt.xlabel('Network Load')
    plt.ylabel('Bandwidth Tax (%)')
    plt.title('Opera: Bandwidth Tax vs Load')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(loads, avg_latencies, 's-', color='green')
    plt.xlabel('Network Load')
    plt.ylabel('Avg Latency (cycles)')
    plt.title('Opera: Latency vs Load')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/opera_efficiency_report.png')
    print("Saved plots/opera_efficiency_report.png")

if __name__ == "__main__":
    run_opera_efficiency_test()
