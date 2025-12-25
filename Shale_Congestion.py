import heapq
import random
import collections

class Cell:
    def __init__(self, src, dst, rem_hops, rank, sender_id=None, creation_time=0):
        self.src = src
        self.dst = dst
        self.rem_hops = rem_hops
        self.rank = rank
        self.sender_id = sender_id  # The node that just sent this cell
        self.creation_time = creation_time
        
    def __lt__(self, other):
        # Used by heapq for PIEO priority. Lower rank = higher priority.
        return self.rank < other.rank

class PIEOQueue:
    """
    Push-In First-Out Queue implementation using a heap.
    In the context of Shale, PIEO allows prioritizing cells based on their rank.
    """
    def __init__(self):
        self.heap = []
    
    def push(self, cell):
        heapq.heappush(self.heap, cell)
    
    def pop_best_eligible(self, eligibility_func):
        """
        Extracts the highest priority (lowest rank) cell that is eligible.
        This simulates a filtering PIEO.
        """
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

    def __len__(self):
        return len(self.heap)

class Switch:
    def __init__(self, node_id, neighbors, bucket_capacity, routing_table):
        self.node_id = node_id
        self.neighbors = neighbors
        self.bucket_capacity = bucket_capacity
        self.routing_table = routing_table
        
        self.credits = collections.defaultdict(lambda: collections.defaultdict(lambda: bucket_capacity))
        self.buckets = collections.defaultdict(list)
        self.pieo_queue = PIEOQueue()
        
        # Feedback queue (tokens to be sent back this cycle)
        self.pending_tokens = []

    def get_next_hop_and_bucket(self, cell):
        """
        Determines where the cell should go next and what bucket it will occupy there.
        In Shale:
        - If rem_hops > 0: Spray to a random neighbor, decrement rem_hops.
        - If rem_hops == 0: Route to destination via shortest path.
        """
        if cell.rem_hops > 0:
            next_hop = random.choice(self.neighbors)
            next_rem_hops = cell.rem_hops - 1
        else:
            # Routing to destination
            if self.node_id == cell.dst:
                return None, None # Already at destination
            next_hop = self.routing_table.get(cell.dst)
            if next_hop is None:
                # Fallback if no route (should not happen in connected graph)
                next_hop = random.choice(self.neighbors)
            next_rem_hops = 0
            
        return next_hop, (cell.dst, next_rem_hops)

    def is_cell_eligible(self, cell):
        """
        Eligibility is determined based on the bucket to which it 
        will be assigned at the next hop.
        """
        next_hop, next_bucket = self.get_next_hop_and_bucket(cell)
        if next_hop is None:
            return True # Ready to be consumed at destination
        
        return self.credits[next_hop][next_bucket] > 0

    def receive_cell(self, cell, from_node):
        """
        When a cell is received, assign it to a bucket and add to PIEO.
        """
        bucket_id = (cell.dst, cell.rem_hops)
        cell.sender_id = from_node
        self.buckets[bucket_id].append(cell)
        self.pieo_queue.push(cell)

    def receive_token(self, from_node, bucket_id):
        """
        A token indicates that a bucket at from_node has become eligible (freed a slot).
        """
        self.credits[from_node][bucket_id] += 1

    def step(self):
        """
        Perform one simulation step for this switch.
        Returns: (transmitted_cell, next_hop, token_to_return)
        """
        # 1. Select the best eligible cell
        cell = self.pieo_queue.pop_best_eligible(self.is_cell_eligible)
        if not cell:
            return None, None, None
            
        # 2. Forward the cell
        next_hop, next_bucket = self.get_next_hop_and_bucket(cell)
        current_bucket_id = (cell.dst, cell.rem_hops)
        self.buckets[current_bucket_id].remove(cell)
        if next_hop is not None:
            self.credits[next_hop][next_bucket] -= 1
            
        # 3. Queue a token to be sent back to previous hop
        token_to_return = None
        if cell.sender_id is not None:
            token_to_return = (cell.sender_id, current_bucket_id)
            
        return cell, next_hop, token_to_return

class ShaleSimulation:
    def __init__(self, adj_matrix, bucket_capacity=10):
        self.num_nodes = len(adj_matrix)
        self.adj_matrix = adj_matrix
        self.bucket_capacity = bucket_capacity
        
        # Build routing tables (BFS for shortest path)
        self.routing_tables = self._build_routing_tables()
        
        # Create switches
        self.switches = []
        for i in range(self.num_nodes):
            neighbors = [nb for nb in self.adj_matrix[i] if nb is not None and nb != i]
            sw = Switch(i, neighbors, bucket_capacity, self.routing_tables[i])
            self.switches.append(sw)
            
        self.time = 0
        self.delivered_cells = []
        self.total_dropped = 0 # Not using drops in this credit-based version

    def _build_routing_tables(self):
        # Normalize adjacency matrix if it's 1-indexed
        # Check if any value is equal to num_nodes
        max_val = 0
        for row in self.adj_matrix:
            for val in row:
                if val is not None:
                    max_val = max(max_val, val)
        
        normalized_adj = []
        is_one_indexed = (max_val == self.num_nodes)
        
        for row in self.adj_matrix:
            new_row = []
            for val in row:
                if val is not None:
                    if is_one_indexed:
                        new_row.append(val - 1)
                    else:
                        new_row.append(val)
            normalized_adj.append(new_row)
            
        self.adj_matrix = normalized_adj
            
        tables = []
        for src in range(self.num_nodes):
            table = {}
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                # Simple BFS for shortest path
                queue = collections.deque([(src, [])])
                visited = {src}
                while queue:
                    u, path = queue.popleft()
                    if u == dst:
                        if path:
                            table[dst] = path[0]
                        break
                    for nb in self.adj_matrix[u]:
                        if nb is not None and nb not in visited:
                            visited.add(nb)
                            queue.append((nb, path + [nb]))
            tables.append(table)
        return tables

    def inject_traffic(self, src, dst, rem_hops, count=1):
        for _ in range(count):
            # Assign a random rank for PIEO
            rank = random.random()
            cell = Cell(src, dst, rem_hops, rank, creation_time=self.time)
            # Source doesn't have a sender_id
            self.switches[src].receive_cell(cell, from_node=None)

    def run_step(self):
        self.time += 1
        
        # Collect all actions to execute simultaneously
        transmissions = [] # List of (cell, to_node)
        tokens = []        # List of (to_node, from_node, bucket_id)
        
        for i in range(self.num_nodes):
            cell, next_hop, token_info = self.switches[i].step()
            
            if cell:
                if next_hop is None:
                    # Cell arrived at destination
                    self.delivered_cells.append((self.time, cell))
                else:
                    transmissions.append((cell, next_hop, i))
            
            if token_info:
                prev_hop, bucket_id = token_info
                tokens.append((prev_hop, i, bucket_id))
                
        # Execute transmissions
        for cell, to_node, from_node in transmissions: 
            if cell.rem_hops > 0:
                cell.rem_hops -= 1
            else:
                cell.rem_hops = 0
                
            self.switches[to_node].receive_cell(cell, from_node)
            
        # Execute tokens
        for to_node, from_node, bucket_id in tokens:
            self.switches[to_node].receive_token(from_node, bucket_id)

    def report(self):
        print(f"\n--- Simulation Report at T={self.time} ---")
        print(f"Total Cells Delivered: {len(self.delivered_cells)}")
        if self.delivered_cells:
            avg_latency = sum(t - c.creation_time for t, c in self.delivered_cells) / len(self.delivered_cells)
            print(f"Average Latency: {avg_latency:.2f} cycles")
            
        for i, sw in enumerate(self.switches):
            total_occ = sum(len(b) for b in sw.buckets.values())
            print(f"Switch {i}: {total_occ} cells in buckets, {len(sw.pieo_queue)} in PIEO")

if __name__ == "__main__":
    from Shale_Alg import RR1
    
    # Use RR1(4) as a simple topology
    print("Initializing Shale Hop-by-Hop Congestion Simulation...")
    topo = RR1(4)
    sim = ShaleSimulation(topo, bucket_capacity=5)
    
    # Inject traffic: Node 0 -> Node 3 with 1 spraying hop
    print("Injecting traffic: 5 cells from Node 0 to Node 3 (rem_hops=1)")
    sim.inject_traffic(src=0, dst=3, rem_hops=1, count=5)
    
    # Run simulation for a few steps
    for s in range(20):
        sim.run_step()
        # Every 5 steps, inject more traffic to see congestion
        if s % 5 == 0:
            sim.inject_traffic(src=1, dst=3, rem_hops=1, count=2)
            
    sim.report()
