import collections
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from Sirius import generate_full_system

class SiriusPacket:
    def __init__(self, src, dst, size, creation_time, is_indirect=False):
        self.src = src
        self.dst = dst
        self.size = size
        self.creation_time = creation_time
        self.is_indirect = is_indirect
        self.intermediate = None
        self.delivery_time = None
        self.hops = 1 if not is_indirect else 2

class SiriusNode:
    def __init__(self, node_id, num_nodes, buffer_limit=1000):
        self.node_id = node_id
        self.queues = {dst: collections.deque() for dst in range(num_nodes) if dst != node_id}
        self.indirect_buffer = [] # Changed to list for indexed removal
        self.buffer_limit = buffer_limit
        self.credits = collections.defaultdict(int) # destination -> available credits at this node
        
    def inject(self, dst, size, time):
        if sum(len(q) for q in self.queues.values()) < self.buffer_limit:
            self.queues[dst].append(SiriusPacket(self.node_id, dst, size, time))
            return True
        return False

class SiriusSimulation:
    def __init__(self, num_nodes, wavelengths, ports, slot_duration=100, reconfig_time=4):
        self.N = num_nodes
        self.W = wavelengths
        self.P = ports
        self.slot_duration = slot_duration
        self.reconfig_time = reconfig_time
        self.efficiency = (slot_duration - reconfig_time) / slot_duration
        
        # Generate Sirius connectivity cycle
        # A_list[t][u][p] = neighbor of node u at port p at timeslot t
        # W_list[t][u][v] = 1 if u connected to v at timeslot t
        self.A_list, self.W_list, _ = generate_full_system(wavelengths, ports, num_nodes)
        self.cycle_len = len(self.W_list)
        
        self.nodes = [SiriusNode(i, num_nodes) for i in range(num_nodes)]
        self.time = 0
        self.delivered_packets = []
        self.total_transmitted_cells = 0
        
    def get_direct_neighbor(self, src_id, t):
        # In this simplified model, we'll use the A_list to find the target
        # For each port p, A_list[t % cycle_len][src_id][p] is the neighbor
        neighbors = []
        for p in range(self.P):
            # A_list values are 1-indexed in Sirius.py implementation
            v = self.A_list[t % self.cycle_len][src_id][p] - 1
            if v >= 0 and v < self.N:
                neighbors.append(v)
        return neighbors

    def run_step(self):
        t = self.time
        
        # 1. Direct Transmission
        for u in range(self.N):
            neighbors = self.get_direct_neighbor(u, t)
            
            # For each active port, prioritize direct traffic
            for v in neighbors:
                # 1a. Check if any indirect traffic at 'u' needs to go to 'v' (2nd hop)
                found_indirect = False
                for i in range(len(self.nodes[u].indirect_buffer)):
                    pkt = self.nodes[u].indirect_buffer[i]
                    if pkt.dst == v:
                        # Deliver 2nd hop
                        self.nodes[u].indirect_buffer.pop(i) # Use list pop for specific index
                        pkt.delivery_time = t + 1
                        self.delivered_packets.append(pkt)
                        self.total_transmitted_cells += 1
                        found_indirect = True
                        break
                
                if found_indirect: continue
                
                # 1b. Direct traffic from u to v (1st hop)
                if v in self.nodes[u].queues and self.nodes[u].queues[v]:
                    pkt = self.nodes[u].queues[v].popleft()
                    pkt.delivery_time = t + 1
                    self.delivered_packets.append(pkt)
                    self.total_transmitted_cells += 1
                
                # 1c. If still idle, try Spraying (1st hop of indirect)
                # This implements the "Inverse Waterfilling" - filling valleys
                else:
                    # Find a destination 'd' that has demand but no direct path now
                    # and use 'v' as intermediate node
                    for d, q in self.nodes[u].queues.items():
                        if q and d != v:
                            if len(self.nodes[v].indirect_buffer) < 50: # Credit Limit C
                                pkt = q.popleft()
                                pkt.is_indirect = True
                                pkt.intermediate = v
                                self.nodes[v].indirect_buffer.append(pkt)
                                self.total_transmitted_cells += 1
                                break
                                
        self.time += 1

def run_sirius_analysis():
    N = 16
    W = 4
    P = 4
    
    loads = np.linspace(0.1, 0.9, 9)
    throughput_results = []
    avg_fct_results = []
    
    print("--- Running Sirius Efficiency Simulation ---")
    for L in loads:
        sim = SiriusSimulation(N, W, P)
        duration = 1000
        
        for t in range(duration):
            # Inject traffic based on load L
            for src in range(N):
                # Poisson arrival for each src node
                num_to_inject = np.random.poisson(L * P)
                for _ in range(num_to_inject):
                    dst = random.randint(0, N-1)
                    if dst != src:
                        sim.nodes[src].inject(dst, 1, t)
            sim.run_step()
            
        tput = len(sim.delivered_packets) / (duration * N * P)
        throughput_results.append(tput)
        
        fcts = [p.delivery_time - p.creation_time for p in sim.delivered_packets]
        avg_fct = np.mean(fcts) if fcts else 0
        avg_fct_results.append(avg_fct)
        
        print(f"Load {L:.1f}: Throughput = {tput:.4f}, Avg FCT = {avg_fct:.2f}")

    # Plotting
    if not os.path.exists('plots'): os.makedirs('plots')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loads, throughput_results, 'o-', label='Simulation')
    plt.plot(loads, loads, '--', color='gray', label='Ideal')
    plt.xlabel('Offered Load')
    plt.ylabel('Throughput')
    plt.title('Sirius: Throughput vs Load')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(loads, avg_fct_results, 's-', color='red')
    plt.xlabel('Offered Load')
    plt.ylabel('Avg FCT (slots)')
    plt.title('Sirius: Latency vs Load')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/sirius_efficiency_report.png')
    print("Saved plots/sirius_efficiency_report.png")

if __name__ == "__main__":
    run_sirius_analysis()
