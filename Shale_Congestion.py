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

class ShaleSimulation:
    def __init__(self, num_nodes, schedules, bucket_capacity=10):
        """
        schedules: list of matching matrices. 
                   Each matrix is a list where mat[i] is the neighbor of node i.
        """
        self.num_nodes = num_nodes
        self.schedules = schedules # Interleaved schedules
        self.bucket_capacity = bucket_capacity
        
        self.switches = [Switch(i, bucket_capacity) for i in range(num_nodes)]
        self.time = 0
        self.delivered_cells = []
        self.failures = set() # (node1, node2) representing broken links

    def inject_traffic(self, src, dst, h, count=1):
        for _ in range(count):
            rank = random.random()
            cell = Cell(src, dst, h, rank, creation_time=self.time)
            self.switches[src].receive_cell(cell, None)

    def add_failure(self, u, v):
        self.failures.add((u, v))
        self.failures.add((v, u))

    def run_step(self):
        # 1. Update current topology
        # We interleave schedules by cycling through them
        current_schedule = self.schedules[self.time % len(self.schedules)]
        for i in range(self.num_nodes):
            self.switches[i].set_neighbor(current_schedule[i])
            
        # 2. Each switch performs its logic
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
                else:
                    transmissions.append((cell, next_hop, i))
            
            if c_token:
                prev_hop, bucket_id = c_token
                credit_tokens.append((prev_hop, i, bucket_id))
                
            if i_token:
                prev_hop, cell_info = i_token
                invalidation_tokens.append((prev_hop, cell_info))

        # 3. Execute movements
        for cell, to_node, from_node in transmissions:
            cell.hops_taken += 1
            cell.path.append(to_node)
            self.switches[to_node].receive_cell(cell, from_node)
            
        for to_node, from_node, bucket_id in credit_tokens:
            self.switches[to_node].receive_credit_token(from_node, bucket_id)

        for to_node, cell_info in invalidation_tokens:
            self.switches[to_node].receive_invalidation_token(cell_info)
            
        self.time += 1

    def report(self):
        print(f"\n--- Shale Simulation Report T={self.time} ---")
        print(f"Nodes: {self.num_nodes}, Failures: {len(self.failures)//2}")
        print(f"Total Delivered: {len(self.delivered_cells)}")
        if self.delivered_cells:
            latencies = [t - c.creation_time for t, c in self.delivered_cells]
            avg_lat = sum(latencies) / len(latencies)
            max_lat = max(latencies)
            print(f"Avg Latency: {avg_lat:.2f} cycles")
            print(f"Max Latency: {max_lat} cycles")
            
            hops = [c.hops_taken for _, c in self.delivered_cells]
            avg_hops = sum(hops) / len(hops)
            print(f"Avg Hops: {avg_hops:.2f}")

        # Check occupancy
        total_in_flight = sum(len(sw.pieo_queue) for sw in self.switches)
        print(f"Total Cells in flight: {total_in_flight}")

def generate_rr_schedule(n):
    """
    Generates a simple RR schedule where each timeslot t, 
    node i is connected to (i + t + 1) % n.
    Returns a list of n-1 matchings.
    """
    schedules = []
    for t in range(n - 1):
        matching = []
        for i in range(n):
            matching.append((i + t + 1) % n)
        schedules.append(matching)
    return schedules

if __name__ == "__main__":
    import numpy as np
    from scipy import stats

    N = 20
    rr_sched = generate_rr_schedule(N)
    sim = ShaleSimulation(N, rr_sched, bucket_capacity=20) # Increased capacity further for larger N
    
    h_range = list(range(1, 13)) # h = 1 to 12
    print(f"Injecting traffic for h values: {h_range}")
    
    # Inject traffic for each h
    for h_val in h_range:
        for _ in range(30): # 30 packets per h
            src, dst = random.sample(range(N), 2)
            sim.inject_traffic(src, dst, h=h_val, count=1)
            
    # Run simulation
    # Run enough steps to clear most traffic
    for s in range(200): # Increased steps for longer paths
        sim.run_step()
            
    sim.report()
    
    # --- Plotting Results ---
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists('plots'):
        os.makedirs('plots')

    if sim.delivered_cells:
        delivery_times = [t for t, c in sim.delivered_cells]
        latencies = [t - c.creation_time for t, c in sim.delivered_cells]
        h_values = [c.h for t, c in sim.delivered_cells]
        hops = [c.hops_taken for t, c in sim.delivered_cells]

        # 1. Latency Analysis (Avg Latency vs h)
        plt.figure(figsize=(10, 6))
        
        avg_lats = []
        std_lats = []
        unique_h = sorted(set(h_values))
        
        for h in unique_h:
            lats = [t - c.creation_time for t, c in sim.delivered_cells if c.h == h]
            avg_lats.append(np.mean(lats))
            std_lats.append(np.std(lats))
            
        plt.errorbar(unique_h, avg_lats, yerr=std_lats, fmt='-o', capsize=5, ecolor='red', label='Mean Latency')
        
        plt.xlabel('VLB Parameter (h)')
        plt.ylabel('Latency (cycles)')
        plt.title('Shale Simulation: Latency Scaling with h')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(unique_h)
        plt.savefig('plots/shale_congestion_report.png')
        print("Saved plots/shale_congestion_report.png")

        # 2. Hop Count Analysis (Avg Hops vs h)
        plt.figure(figsize=(10, 6))
        
        avg_hops = []
        std_hops = []
        
        for h in unique_h:
            hp = [c.hops_taken for t, c in sim.delivered_cells if c.h == h]
            avg_hops.append(np.mean(hp))
            std_hops.append(np.std(hp))
            
        plt.errorbar(unique_h, avg_hops, yerr=std_hops, fmt='-s', color='green', capsize=5, label='Mean Hops')
        
        plt.xlabel('VLB Parameter (h)')
        plt.ylabel('Hops Taken')
        plt.title('Shale Simulation: Hop Scaling with h')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(unique_h)
        plt.savefig('plots/shale_latency_dist.png')
        print("Saved plots/shale_latency_dist.png")

        # 3. Best Fit Analysis
        # Equation 1: Avg Latency vs h
        avg_latencies = []
        avg_hops_list = []
        valid_h = []
        
        print("\n--- Statistical Analysis ---")
        for h in h_range:
            lats = [t - c.creation_time for t, c in sim.delivered_cells if c.h == h]
            hp = [c.hops_taken for t, c in sim.delivered_cells if c.h == h]
            if lats:
                avg_l = np.mean(lats)
                avg_h = np.mean(hp)
                avg_latencies.append(avg_l)
                avg_hops_list.append(avg_h)
                valid_h.append(h)
                print(f"h={h}: Avg Latency={avg_l:.2f}, Avg Hops={avg_h:.2f}")

        # Linear Regression for Latency
        slope_l, intercept_l, r_value_l, p_value_l, std_err_l = stats.linregress(valid_h, avg_latencies)
        print(f"\nLatency vs h Best Fit: Latency = {slope_l:.2f} * h + {intercept_l:.2f}")
        print(f"R-squared: {r_value_l**2:.4f}")

        # Linear Regression for Hops
        slope_h, intercept_h, r_value_h, p_value_h, std_err_h = stats.linregress(valid_h, avg_hops_list)
        print(f"Hops vs h Best Fit: Hops = {slope_h:.2f} * h + {intercept_h:.2f}")
        print(f"R-squared: {r_value_h**2:.4f}")
        
        # Save fit plot for reference (optional, not requested to put in tex but good for user)
        plt.figure(figsize=(8,5))
        plt.plot(valid_h, avg_latencies, 'o', label='Simulated Data')
        plt.plot(valid_h, [slope_l*x + intercept_l for x in valid_h], 'r--', label=f'Fit: {slope_l:.2f}h + {intercept_l:.2f}')
        plt.xlabel('h parameter')
        plt.ylabel('Average Latency')
        plt.title('Avg Latency vs h parameter')
        plt.legend()
        plt.savefig('plots/shale_h_fit.png')
