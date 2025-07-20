#!/usr/bin/env python3
"""
Refined hyperparameter search configuration based on initial search results.
Focuses on the parameter ranges that showed promise.
"""

# Grid search configuration - smaller, focused grid
REFINED_GRID_SEARCH = {
    'lr_actor': [5e-5, 8e-5, 1e-4, 2e-4],
    'lr_critic': [1e-4, 2.5e-4, 5e-4, 1e-3],
    'lambda_actor': [0.85, 0.88, 0.90],
    'lambda_critic': [0.91, 0.93, 0.95],
    'noise_std': [0.02, 0.05, 0.08],
    'gamma': [0.96, 0.97],
    'reward_scale': [8.0, 10.0, 13.0],
    'td_clip': [5.0]  # Fixed based on results
}

# Random search configuration - focused ranges
REFINED_RANDOM_SEARCH = {
    'lr_actor': (5e-5, 2e-4),          # Narrowed from (1e-5, 1e-2)
    'lr_critic': (1e-4, 1e-3),         # Narrowed from (1e-4, 5e-2)
    'lambda_actor': (0.85, 0.92),      # Narrowed from (0.8, 0.99)
    'lambda_critic': (0.91, 0.96),     # Narrowed from (0.9, 0.99)
    'noise_std': (0.02, 0.1),          # Narrowed from (0.01, 1.0)
    'gamma': (0.96, 0.98),             # Narrowed from (0.95, 0.999)
    'reward_scale': (5.0, 15.0),       # Focused on moderate values
    'td_clip': (4.0, 7.0)              # Narrowed from (1.0, 10.0)
}

# Top performer variations - small perturbations around best config
TOP_PERFORMER_VARIATIONS = {
    'lr_actor': [6e-5, 7e-5, 8e-5, 9e-5, 1e-4],
    'lr_critic': [1.5e-4, 2e-4, 2.5e-4, 3e-4],
    'lambda_actor': [0.86, 0.87, 0.88, 0.89],
    'lambda_critic': [0.91, 0.915, 0.92, 0.925],
    'noise_std': [0.02, 0.025, 0.03, 0.04, 0.05],
    'gamma': [0.96, 0.963, 0.965, 0.97],
    'reward_scale': [10.0, 11.0, 12.0, 13.0, 14.0],
    'td_clip': [5.0, 5.5, 6.0]
}

def get_search_config(search_type='refined_random'):
    """
    Get search configuration based on type.
    
    Args:
        search_type: One of 'refined_grid', 'refined_random', 'top_variations'
    
    Returns:
        Dictionary of parameter ranges/values
    """
    configs = {
        'refined_grid': REFINED_GRID_SEARCH,
        'refined_random': REFINED_RANDOM_SEARCH,
        'top_variations': TOP_PERFORMER_VARIATIONS
    }
    
    if search_type not in configs:
        raise ValueError(f"Unknown search type: {search_type}. Choose from {list(configs.keys())}")
    
    return configs[search_type]

if __name__ == "__main__":
    # Print search space sizes
    import itertools
    
    print("Refined Search Configurations:")
    print("="*50)
    
    # Grid search size
    grid_size = 1
    for param, values in REFINED_GRID_SEARCH.items():
        grid_size *= len(values)
    print(f"Refined Grid Search: {grid_size} combinations")
    
    # Random search ranges
    print(f"\nRefined Random Search ranges:")
    for param, (low, high) in REFINED_RANDOM_SEARCH.items():
        print(f"  {param:15s}: [{low:.2e}, {high:.2e}]")
    
    # Top performer variations size
    var_size = 1
    for param, values in TOP_PERFORMER_VARIATIONS.items():
        var_size *= len(values)
    print(f"\nTop Performer Variations: {var_size} combinations")