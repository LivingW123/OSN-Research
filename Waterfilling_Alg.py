import numpy as np

def waterfilling(channels, total_power):
    """
    Implements the Waterfilling algorithm for power allocation.
    
    Args:
        channels (list[float] or np.array): Noise levels (or inverse channel gains) for each channel.
        total_power (float): Total power budget to be allocated.
        
    Returns:
        np.array: Allocated power for each channel.
    """
    channels = np.array(channels)
    
    # Handle 1D case (single timeslot)
    if channels.ndim == 1:
        return _waterfilling_1d(channels, total_power)
    
    # Handle 2D case (multiple timeslots)
    elif channels.ndim == 2:
        num_timeslots, num_channels = channels.shape
        allocations = np.zeros_like(channels, dtype=float)
        
        # Handle total_power being a scalar or a list/array
        if np.isscalar(total_power):
            powers = np.full(num_timeslots, total_power)
        else:
            powers = np.array(total_power)
            if len(powers) != num_timeslots:
                raise ValueError("Length of total_power must match number of timeslots.")
        
        for t in range(num_timeslots):
            allocations[t] = _waterfilling_1d(channels[t], powers[t])
            
        return allocations
    else:
        raise ValueError("Channels must be 1D or 2D array.")

if __name__ == "__main__":
    # Example Usage
    
    # Case 1: Simple case
    noise_levels = [10, 20, 30, 40]
    P_tot = 50
    
    print(f"--- Waterfilling Algorithm Example ---")
    print(f"Noise Levels: {noise_levels}")
    print(f"Total Power: {P_tot}")
    
    allocation = waterfilling(noise_levels, P_tot)
    
    print(f"Allocated Power: {allocation}")
    print(f"Total Allocated: {np.sum(allocation)}")
    print(f"Water Level (Noise + Power): {allocation + noise_levels}")
    print("-" * 30)
    
    # Case 2: Some channels too noisy
    noise_levels_2 = [5, 10, 100, 120]
    P_tot_2 = 20
    
    print(f"Noise Levels: {noise_levels_2}")
    print(f"Total Power: {P_tot_2}")
    
    allocation_2 = waterfilling(noise_levels_2, P_tot_2)
    
    print(f"Allocated Power: {allocation_2}")
    print(f"Total Allocated: {np.sum(allocation_2)}")
    print(f"Water Level (Noise + Power): {allocation_2 + np.array(noise_levels_2)}")
