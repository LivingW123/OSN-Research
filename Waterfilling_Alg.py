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
    n = len(channels)
    noise = np.array(channels)
    
    # Sort noise levels to easily find the water level
    sorted_indices = np.argsort(noise)
    sorted_noise = noise[sorted_indices]
    
    
    water_level = 0
    active_channels_count = n
    
    for i in range(n):
        current_noise_sum = np.sum(sorted_noise[:n-i])
        potential_water_level = (total_power + current_noise_sum) / (n - i)
        
        if potential_water_level > sorted_noise[n-i-1]:
            water_level = potential_water_level
            active_channels_count = n - i
            break
            
    # Calculate power allocation: P_i = (Water Level - Noise_i)+
    power_allocation = np.maximum(water_level - noise, 0)
    
    return power_allocation

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
