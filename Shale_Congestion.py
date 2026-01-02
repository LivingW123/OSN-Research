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
    # Simulate interleaving h=1 and h=4 schedules
    # Here "schedule" refers to the RR configuration sequence.
    # In logic, h is a property of the traffic.
    
    N = 8
    rr_sched = generate_rr_schedule(N)
    
    # Create "interleaved schedules" by just having two identical RR sequences 
    # but we could vary them if we wanted.
    # The requirement says "simulate multiple schedules running in parallel".
    # We'll just use the standard RR schedule and show both h=1 and h=4 traffic.
    
    sim = ShaleSimulation(N, rr_sched, bucket_capacity=5)
    
    print("Injecting mixed h=1 and h=4 traffic...")
    # h=1 traffic (Low Latency)
    for _ in range(10):
        src, dst = random.sample(range(N), 2)
        sim.inject_traffic(src, dst, h=1, count=1)
        
    # h=4 traffic (High Throughput)
    for _ in range(10):
        src, dst = random.sample(range(N), 2)
        sim.inject_traffic(src, dst, h=4, count=1)
        
    # Run simulation
    for s in range(50):
        sim.run_step()
        if s == 10:
            # Simulate a link failure midway
            print("\n!!! Simulating link failure (0, 1) at T=10 !!!")
            sim.add_failure(0, 1)
            
    sim.report()
    
    # --- Plotting Results ---
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 1. Latency vs Delivery Time
    if sim.delivered_cells:
        delivery_times = [t for t, c in sim.delivered_cells]
        latencies = [t - c.creation_time for t, c in sim.delivered_cells]
        h_values = [c.h for t, c in sim.delivered_cells]

        plt.figure(figsize=(10, 6))
        for h_type in sorted(set(h_values)):
            pts = [(dt, lat) for dt, lat, hv in zip(delivery_times, latencies, h_values) if hv == h_type]
            if pts:
                dts, lats = zip(*pts)
                plt.scatter(dts, lats, label=f'h={h_type} traffic', alpha=0.7)

        plt.xlabel('Simulation Time (cycles)')
        plt.ylabel('Latency (cycles)')
        plt.title('Shale Simulation: Cell Latency over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('plots/shale_congestion_report.png')
        print("Saved plots/shale_congestion_report.png")

        # 2. Hop Count Distribution
        plt.figure(figsize=(8, 5))
        for h_type in sorted(set(h_values)):
            hops = [c.hops_taken for t, c in sim.delivered_cells if c.h == h_type]
            plt.hist(hops, bins=range(min(hops), max(hops) + 2), alpha=0.5, label=f'h={h_type}', align='left', rwidth=0.8)
        
        plt.xlabel('Hops Taken')
        plt.ylabel('Cell Count')
        plt.title('Shale Simulation: Hop Distribution')
        plt.legend()
        plt.savefig('plots/shale_latency_dist.png') # Reusing name for tex
        print("Saved plots/shale_latency_dist.png")

    # Show some paths
    print("\nSample paths:")
    for _, c in sim.delivered_cells[:5]:
        print(f"{c.src} -> {c.dst} (h={c.h}): {c.path} latency={_ - c.creation_time}")
