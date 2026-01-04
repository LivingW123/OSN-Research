import heapq
import random
import collections

class Cell:
    def __init__(self, src, dst, h, rank, creation_time=0):
        self.src = src
        self.dst = dst
        self.h = h  # Number of spraying hops
        self.hops_taken = 0
        self.rank = rank # Priority for PIEO
        self.creation_time = creation_time
        self.sender_id = None
        self.path = [src]
        self.is_invalid = False
        
    def __lt__(self, other):
        # Used by heapq for PIEO priority. Lower rank = higher priority.
        return self.rank < other.rank

    def __repr__(self):
        return f"Cell({self.src}->{self.dst}, h={self.h}, hops={self.hops_taken}, rank={self.rank:.2f})"

class PIEOQueue:
    """
    Push-In Extract-Out Queue.
    In Shale, we extract the highest-priority (lowest rank) cell that is ELIGIBLE.
    """
    def __init__(self):
        self.heap = []
    
    def push(self, cell):
        heapq.heappush(self.heap, cell)
    
    def pop_best_eligible(self, eligibility_func):
        temp_stack = []
        best_cell = None
        
        while self.heap:
            cell = heapq.heappop(self.heap)
            if eligibility_func(cell):
                best_cell = cell
                break
            else:
                temp_stack.append(cell)
        
        # Put back ineligible cells
        for c in temp_stack:
            heapq.heappush(self.heap, c)
            
        return best_cell

    def remove_cell(self, cell):
        # Inefficient but necessary if we remove by reference
        if cell in self.heap:
            self.heap.remove(cell)
            heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)

class Switch:
    def __init__(self, node_id, bucket_capacity):
        self.node_id = node_id
        self.bucket_capacity = bucket_capacity
        
        # Credits: credits[neighbor_node][bucket_id]
        # bucket_id = (dst, hop_type) where hop_type is 'spray' or 'direct'
        self.credits = collections.defaultdict(lambda: collections.defaultdict(lambda: bucket_capacity))
        
        # Queues
        self.pieo_queue = PIEOQueue()
        
        # Buckets for occupancy tracking
        self.buckets = collections.defaultdict(list)
        
        # Token queues
        self.pending_credit_tokens = [] # (prev_node, bucket_id)
        self.pending_invalidation_tokens = [] # (prev_node, cell_to_invalidate)
        
        # Current neighbor in this timeslot
        self.current_neighbor = None

    def set_neighbor(self, neighbor):
        self.current_neighbor = neighbor

    def get_cell_next_step(self, cell):
        """
        Returns (next_hop, next_bucket_id) for a cell.
        VLB logic:
        - If hops_taken < h: We are spraying. Next hop is current neighbor.
        - If hops_taken >= h: We are in Direct phase. Next hop MUST be dst.
        """
        if cell.hops_taken < cell.h:
            # Spraying Phase
            next_hop = self.current_neighbor
            next_bucket_id = (cell.dst, 'spray')
        else:
            # Direct Phase
            next_hop = cell.dst
            next_bucket_id = (cell.dst, 'direct')
            
        return next_hop, next_bucket_id

    def is_cell_eligible(self, cell, failures):
        """
        Cell is eligible if:
        1. Next hop is the current physical neighbor (if direct phase, must match).
        2. Next hop has credits for the required bucket.
        3. Link is not failed.
        """
        next_hop, next_bucket = self.get_cell_next_step(cell)
        
        if next_hop is None:
            return False
            
        # If in direct phase, we can ONLY send if current_neighbor IS next_hop (the dst)
        if cell.hops_taken >= cell.h:
            if self.current_neighbor != cell.dst:
                return False
        
        # Check credits at the current physical neighbor
        # Note: In Shale, we only care about the neighbor we are actually connected to.
        if self.current_neighbor != next_hop:
            # This happens if we want to spray but next_hop is not current_neighbor?
            # No, in spraying phase we ALWAYS use current_neighbor.
            # In direct phase, we wait until current_neighbor == dst.
            return False

        # Check for link failure
        if (self.node_id, next_hop) in failures:
            return False

        return self.credits[next_hop][next_bucket] > 0

    def receive_cell(self, cell, from_node):
        cell.sender_id = from_node
        bucket_id = (cell.dst, 'spray' if cell.hops_taken < cell.h else 'direct')
        self.buckets[bucket_id].append(cell)
        self.pieo_queue.push(cell)

    def receive_credit_token(self, from_node, bucket_id):
        self.credits[from_node][bucket_id] += 1

    def receive_invalidation_token(self, cell_info):
        # Signal that the path for a certain destination is broken
        # In this simple model, we just mark matching cells as invalid so they are dropped
        for cell in self.pieo_queue.heap:
            if cell.dst == cell_info.dst:
                cell.is_invalid = True

    def step(self, failures):
        """
        1. Clean up invalid cells.
        2. Pick best eligible cell.
        3. Transmit if possible.
        4. Detect failures and generate invalidation tokens.
        """
        # 1. Clean up
        invalids = [c for c in self.pieo_queue.heap if c.is_invalid]
        for c in invalids:
            self.pieo_queue.remove_cell(c)
            # Free credits if it was from a previous hop
            if c.sender_id is not None:
                bucket_id = (c.dst, 'spray' if c.hops_taken < c.h else 'direct')
                self.pending_credit_tokens.append((c.sender_id, bucket_id))
        
        # 2. Select cell
        cell = self.pieo_queue.pop_best_eligible(lambda c: self.is_cell_eligible(c, failures))
        
        # 3. Check for blocked cells due to failures (Invalidation Token logic)
        invalidation_token = None
        for c in self.pieo_queue.heap:
            if c.is_invalid: continue
            
            next_hop, next_bucket = self.get_cell_next_step(c)
            # If the required link for the direct phase is failed
            if c.hops_taken >= c.h and self.current_neighbor == c.dst:
                if (self.node_id, next_hop) in failures:
                    # Path is broken! Generate invalidation token.
                    # In a real system, this would be sent back to the source or previous hop.
                    if c.sender_id is not None:
                        invalidation_token = (c.sender_id, c)
                        c.is_invalid = True # Drop it here too
                        break

        if not cell:
            return None, None, None, invalidation_token

        # 4. Process transmission
        next_hop, next_bucket = self.get_cell_next_step(cell)
        
        # Consume credit
        self.credits[next_hop][next_bucket] -= 1
        
        # Remove from local bucket tracking
        curr_bucket_id = (cell.dst, 'spray' if cell.hops_taken < cell.h else 'direct')
        if cell in self.buckets[curr_bucket_id]:
            self.buckets[curr_bucket_id].remove(cell)
            
        # Queue credit token for previous hop
        credit_token = None
        if cell.sender_id is not None:
            credit_token = (cell.sender_id, curr_bucket_id)
            
        return cell, next_hop, credit_token, invalidation_token

class Flow:
    def __init__(self, flow_id, src, dst, size, creation_time):
        self.flow_id = flow_id
        self.src = src
        self.dst = dst
        self.size = size
        self.creation_time = creation_time
        self.cells_received = 0
        self.completion_time = None

class ShaleSimulation:
    def __init__(self, num_nodes, schedules, bucket_capacity=10, token_budget_f=5, token_budget=1):
        self.num_nodes = num_nodes
        self.schedules = schedules
        self.bucket_capacity = bucket_capacity
        self.token_budget_f = token_budget_f # T_F
        self.token_budget = token_budget # T
        
        self.switches = [Switch(i, bucket_capacity) for i in range(num_nodes)]
        self.time = 0
        self.delivered_cells = []
        self.completed_flows = []
        self.active_flows = {} # flow_id -> Flow
        self.failures = set()

    def inject_flow(self, src, dst, h, size, flow_id):
        flow = Flow(flow_id, src, dst, size, self.time)
        self.active_flows[flow_id] = flow
        # Inject cells for this flow
        for i in range(size):
            rank = self.time + (i * 0.01) # Simple fifo-like rank within flow
            cell = Cell(src, dst, h, rank, creation_time=self.time)
            cell.flow_id = flow_id
            self.switches[src].receive_cell(cell, None)

    def run_step(self):
        # 1. Update current topology
        current_schedule = self.schedules[self.time % len(self.schedules)]
        for i in range(self.num_nodes):
            self.switches[i].set_neighbor(current_schedule[i])
            
        transmissions = []
        credit_tokens = []
        invalidation_tokens = []
        
        for i in range(self.num_nodes):
            cell, next_hop, c_token, i_token = self.switches[i].step(self.failures)
            
            if cell:
                if next_hop == cell.dst and cell.hops_taken >= cell.h:
                    # Delivered!
                    cell.hops_taken += 1
                    cell.path.append(next_hop)
                    self.delivered_cells.append((self.time + 1, cell))
                    
                    # Update Flow
                    if hasattr(cell, 'flow_id'):
                        flow = self.active_flows[cell.flow_id]
                        flow.cells_received += 1
                        if flow.cells_received == flow.size:
                            flow.completion_time = self.time + 1
                            self.completed_flows.append(flow)
                            # del self.active_flows[cell.flow_id] # Keep for stats or del? del is safer for mem
                else:
                    transmissions.append((cell, next_hop, i))
            
            if c_token:
                credit_tokens.append((c_token[0], i, c_token[1]))
            if i_token:
                invalidation_tokens.append((i_token[0], i_token[1]))

        # Execute movements
        for cell, to_node, from_node in transmissions:
            cell.hops_taken += 1
            cell.path.append(to_node)
            self.switches[to_node].receive_cell(cell, from_node)
            
        for to_node, from_node, bucket_id in credit_tokens:
            self.switches[to_node].receive_credit_token(from_node, bucket_id)

        for to_node, cell_info in invalidation_tokens:
            self.switches[to_node].receive_invalidation_token(cell_info)
            
        self.time += 1

def generate_rr_schedule(n):
    schedules = []
    for t in range(n - 1):
        matching = []
        for i in range(n):
            matching.append((i + t + 1) % n)
        schedules.append(matching)
    return schedules

# --- Analysis Functions ---

def check_bottlenecks(N, h, P, T_F, T, E):
    """
    Returns True if conditions are met.
    P: Propagation Delay
    E: Epoch Length
    """
    # 1. First-Hop Bottleneck Condition: P <= h * T_F * E
    cond1 = P <= h * T_F * E
    
    # 2. Penultimate Link Bottleneck: P <= h * T * (h * N^(1/h) - 1) * E
    # Note: N^(1/h) might calculate root.
    try:
        root_n = N ** (1.0/h)
    except:
        root_n = 0
        
    limit2 = h * T * (h * root_n - 1) * E
    cond2 = P <= limit2
    
    return cond1, cond2, h * T_F * E, limit2

def run_load_sweep():
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = 16
    E = N - 1 # Simple epoch
    P = 10 # Assumed propagation delay (tokens take time to return)
    T_F = 5
    T = 1
    
    flow_size = 10
    
    loads = np.linspace(0.05, 0.6, 8) # Load Factors to test
    h_values = [1, 2, 4, 6, 8, 12]
    
    results_throughput = {h: [] for h in h_values}
    results_fct = {h: [] for h in h_values}
    
    print(f"--- Starting Load Sweep (N={N}, FlowSize={flow_size}) ---")
    
    for h in h_values:
        # Check limits
        c1, c2, l1, l2 = check_bottlenecks(N, h, P, T_F, T, E)
        print(f"\nAnalyzing h={h}:")
        print(f"  Bottleneck Check (P={P}):")
        print(f"  First-Hop Limit: {l1:.1f} -> {'OK' if c1 else 'VIOLATION'}")
        print(f"  Penultimate Limit: {l2:.1f} -> {'OK' if c2 else 'VIOLATION'}")
        
        theoretical_limit = 1.0 / (2**(int(np.log2(h))+1) if h > 1 else 2) 
        # Approx mapping: h=1->0.5, h=2->0.25, h=4->0.125
        # Actually paper says: h=1 -> 1/2, h=2 -> 1/4, h=4 -> 1/8
        limit_val = 1.0 / (2 * h) if h > 1 else 0.5
        if h==4: limit_val = 0.125
        if h==2: limit_val = 0.25
        
        print(f"  Theoretical Throughput Limit: {limit_val}")

        for L in loads:
            # Create sim
            rr_sched = generate_rr_schedule(N)
            sim = ShaleSimulation(N, rr_sched, bucket_capacity=20, token_budget_f=T_F, token_budget=T)
            
            # Injection Logic
            # Total Capacity ~ N * 1 cell/cycle (line rate)
            # Desired Load L means total injection rate = L * N
            # Per node injection prob = L
            
            duration = 400
            total_sent_cells = 0
            
            flow_id_counter = 0
            for t in range(duration):
                # Inject?
                for n_idx in range(N):
                    if random.random() < (L / flow_size): # Inject FLOWS, so prob is L / size
                        dst = random.randint(0, N-1)
                        if dst != n_idx:
                            sim.inject_flow(n_idx, dst, h, flow_size, flow_id_counter)
                            flow_id_counter += 1
                            total_sent_cells += flow_size
                
                sim.run_step()
                
            # Metrics
            # Throughput = Total Delivered Cells / Duration / N (normalized to line rate)
            delivered = len(sim.delivered_cells)
            throughput = delivered / duration / N
            
            # Normalized FCT
            # FCT = t / (F + P)
            # t = completion_time - creation_time
            fcts = []
            for f in sim.completed_flows:
                t_actual = f.completion_time - f.creation_time
                norm_fct = t_actual / (f.size + P) # As per user formula (P is delay parameter)
                fcts.append(norm_fct)
            
            avg_fct = np.mean(fcts) if fcts else 0
            
            results_throughput[h].append(throughput)
            results_fct[h].append(avg_fct)
            
            print(f"  L={L:.2f} -> Tput={throughput:.3f}, NormFCT={avg_fct:.2f}")

    # --- Plotting ---
    if not os.path.exists('plots'): os.makedirs('plots')
    
    # 1. Throughput vs Load
    plt.figure(figsize=(10, 6))
    for h in h_values:
        plt.plot(loads, results_throughput[h], 'o-', label=f'h={h}')
        # Plot theoretical limits
        limit = 0.5 if h==1 else (0.25 if h==2 else 0.125)
        plt.axhline(y=limit, linestyle='--', alpha=0.3, color='gray')
        
    plt.xlabel('Load Factor (L)')
    plt.ylabel('Normalized Throughput')
    plt.title('Shale Throughput vs Load Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/shale_throughput_vs_load.png')
    
    # 2. FCT vs Load
    plt.figure(figsize=(10, 6))
    for h in h_values:
        valid_idxs = [i for i, v in enumerate(results_fct[h]) if v > 0]
        if valid_idxs:
            plt.plot([loads[i] for i in valid_idxs], [results_fct[h][i] for i in valid_idxs], 's-', label=f'h={h}')
            
    plt.xlabel('Load Factor (L)')
    plt.ylabel('Avg Normalized FCT')
    plt.title('Shale Normalized Flow Completion Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/shale_fct_vs_load.png')

if __name__ == "__main__":
    import numpy as np
    import os
    run_load_sweep()
