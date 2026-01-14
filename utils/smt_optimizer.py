"""
SMT Optimizer - Improves Z3 solver performance and coverage.

Key optimizations:
1. Z3 timeout configuration  
2. Constraint relaxation for hard problems
3. Incremental solving strategy
4. Search space reduction for large problems
5. Day allocation heuristics for multi-city trips

Problem Analysis (from experiments):
- "Easy" queries (no constraints) with 7 days × 3 cities timeout
- The search space is LARGER when there are NO constraints
- Need aggressive pruning for multi-day, multi-city scenarios
"""
import os
from typing import Any, Dict, List, Optional, Tuple


# Z3 solver parameters for better performance
Z3_PARAMS = {
    "timeout": 120000,  # 2 minutes per solve attempt (in milliseconds)
    "threads": 4,       # Use multiple threads  
    "sat.random_seed": 42,
}

# Search space limits to prevent combinatorial explosion
SEARCH_SPACE_LIMITS = {
    "restaurants_per_city": 15,     # Max restaurants to consider per city
    "attractions_per_city": 20,     # Max attractions to consider per city
    "accommodations_per_city": 10,  # Max accommodations per city
    "flights_per_route": 5,         # Max flights per route
}

# Scaling factors for multi-day trips
MULTI_DAY_SCALING = {
    3: 1.0,   # 3-day trip: no reduction
    5: 0.8,   # 5-day trip: 80% of options
    7: 0.6,   # 7-day trip: 60% of options (most aggressive)
}

# City allocation strategy for multi-city trips
CITY_ALLOCATION = {
    # (days, cities): allocation pattern
    (3, 1): [3],
    (5, 2): [2, 3],
    (7, 3): [2, 2, 3],  # More balanced allocation
    (7, 2): [3, 4],
}


def get_z3_timeout() -> int:
    """Get Z3 timeout in milliseconds from environment or default."""
    timeout_sec = int(os.getenv("SMT_TIMEOUT_SEC", "120"))
    return timeout_sec * 1000


def configure_z3_solver(solver) -> None:
    """Configure Z3 solver with optimized parameters."""
    try:
        from z3 import set_param
        
        # Global Z3 parameters
        set_param("parallel.enable", True)
        set_param("timeout", get_z3_timeout())
        
        # Solver-specific parameters if available
        if hasattr(solver, 'set'):
            solver.set("timeout", get_z3_timeout())
            solver.set("threads", Z3_PARAMS["threads"])
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not configure Z3: {e}")


# Constraint relaxation order - relax least important first
RELAXATION_ORDER = [
    "cuisine_preference",      # Can eat different cuisine
    "attraction_diversity",    # Can revisit attractions if needed
    "restaurant_diversity",    # Can revisit restaurants
    "room_type_preference",    # Can use different room type
    "exact_budget",            # Can slightly exceed budget
    "transportation_type",     # Can use different transport
]


def get_relaxation_level() -> int:
    """Get current relaxation level (0 = strict, higher = more relaxed)."""
    return int(os.getenv("SMT_RELAX_LEVEL", "0"))


def should_relax_constraint(constraint_name: str) -> bool:
    """Check if a constraint should be relaxed based on current level."""
    level = get_relaxation_level()
    if level == 0:
        return False
    
    try:
        idx = RELAXATION_ORDER.index(constraint_name)
        return idx < level
    except ValueError:
        return False


def build_relaxed_constraints(
    original_constraints: Dict[str, Any],
    relax_level: int = 0
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Build constraints with relaxation applied.
    
    Returns:
        Tuple of (relaxed_constraints, list of relaxed constraint names)
    """
    relaxed = dict(original_constraints)
    relaxed_names = []
    
    for i, constraint in enumerate(RELAXATION_ORDER):
        if i < relax_level:
            if constraint in relaxed:
                del relaxed[constraint]
                relaxed_names.append(constraint)
    
    return relaxed, relaxed_names


def estimate_problem_complexity(query_item: Dict[str, Any]) -> str:
    """
    Estimate the complexity of a travel planning problem.
    
    Returns: "easy", "medium", or "hard"
    """
    days = int(query_item.get("days", 3))
    
    local = query_item.get("local_constraint", {})
    if isinstance(local, str):
        try:
            import ast
            local = ast.literal_eval(local)
        except:
            local = {}
    
    # Count hard constraints
    hard_constraint_count = 0
    for key in ["cuisine", "room type", "house rule", "transportation"]:
        if local.get(key):
            hard_constraint_count += 1
    
    # Complexity scoring
    if days >= 7 or hard_constraint_count >= 3:
        return "hard"
    elif days >= 5 or hard_constraint_count >= 2:
        return "medium"
    else:
        return "easy"


def get_recommended_timeout(query_item: Dict[str, Any]) -> int:
    """Get recommended timeout in seconds based on problem complexity."""
    complexity = estimate_problem_complexity(query_item)
    
    timeouts = {
        "easy": 60,
        "medium": 120,
        "hard": 180,
    }
    
    return int(os.getenv("SMT_TIMEOUT_SEC", str(timeouts.get(complexity, 120))))


def create_incremental_strategy(days: int) -> List[Dict[str, Any]]:
    """
    Create an incremental solving strategy.
    
    Instead of solving everything at once, solve in phases:
    1. First solve transportation/routing
    2. Then add accommodation constraints
    3. Finally add meals and attractions
    """
    strategies = []
    
    # Phase 1: Core routing
    strategies.append({
        "phase": "routing",
        "constraints": ["origin", "destination", "dates", "transportation"],
        "timeout": 30,
    })
    
    # Phase 2: Accommodation
    strategies.append({
        "phase": "accommodation",
        "constraints": ["room_type", "house_rule", "budget_partial"],
        "timeout": 30,
    })
    
    # Phase 3: Complete plan
    strategies.append({
        "phase": "complete",
        "constraints": ["meals", "attractions", "budget_full"],
        "timeout": 60,
    })
    
    return strategies


def get_search_space_limit(category: str, days: int) -> int:
    """
    Get the limit for a category based on trip length.
    
    Longer trips get more aggressive limits to keep search space manageable.
    """
    base_limit = SEARCH_SPACE_LIMITS.get(category, 20)
    
    # Apply scaling based on trip length
    scale = MULTI_DAY_SCALING.get(days, 1.0)
    if days >= 7:
        scale = 0.6  # Most aggressive for 7+ days
    elif days >= 5:
        scale = 0.8
    
    return max(5, int(base_limit * scale))


def get_city_allocation(days: int, num_cities: int) -> List[int]:
    """
    Get recommended day allocation for cities.
    
    This helps the solver by providing a starting point for city visits.
    """
    key = (days, num_cities)
    if key in CITY_ALLOCATION:
        return CITY_ALLOCATION[key]
    
    # Default: distribute days evenly with remainder to last city
    base = days // num_cities
    remainder = days % num_cities
    allocation = [base] * num_cities
    allocation[-1] += remainder
    
    return allocation


def estimate_search_space_size(
    days: int,
    cities: int,
    restaurants_per_city: int = 30,
    attractions_per_city: int = 40,
    accommodations_per_city: int = 20,
) -> float:
    """
    Estimate the raw search space size for a planning problem.
    
    Returns order of magnitude estimate.
    """
    # Meals: 3 per day, choose from restaurants
    meal_choices = (restaurants_per_city ** 3) ** days
    
    # Attractions: ~2 per day
    attraction_choices = (attractions_per_city ** 2) ** days
    
    # Accommodation: 1 per night
    accommodation_choices = accommodations_per_city ** (days - 1)
    
    # City allocation: permutations
    from math import factorial
    city_perms = factorial(cities) if cities > 1 else 1
    
    total = meal_choices * attraction_choices * accommodation_choices * city_perms
    
    return total


def should_use_aggressive_pruning(query_item: Dict[str, Any]) -> bool:
    """
    Determine if aggressive pruning should be used based on problem characteristics.
    
    Returns True for problems likely to timeout without pruning.
    """
    days = int(query_item.get("days", 3))
    cities = int(query_item.get("visiting_city_number", 1))
    
    local = query_item.get("local_constraint", {})
    if isinstance(local, str):
        try:
            import ast
            local = ast.literal_eval(local)
        except:
            local = {}
    
    # Count constraints that narrow the search space
    narrowing_constraints = sum(1 for k in ["cuisine", "room type", "transportation"] 
                                 if local.get(k))
    
    # Use aggressive pruning if:
    # 1. 7+ days
    # 2. 3+ cities
    # 3. Few narrowing constraints (larger search space)
    if days >= 7 and cities >= 3:
        return True
    if days >= 7 and narrowing_constraints == 0:
        return True
    if days >= 5 and cities >= 3 and narrowing_constraints == 0:
        return True
    
    return False


def get_pruning_config(query_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get pruning configuration for a query.
    
    Returns configuration dict to apply to data filtering.
    """
    days = int(query_item.get("days", 3))
    cities = int(query_item.get("visiting_city_number", 1))
    
    use_aggressive = should_use_aggressive_pruning(query_item)
    
    config = {
        "aggressive_mode": use_aggressive,
        "restaurants_per_city": get_search_space_limit("restaurants_per_city", days),
        "attractions_per_city": get_search_space_limit("attractions_per_city", days),
        "accommodations_per_city": get_search_space_limit("accommodations_per_city", days),
        "flights_per_route": get_search_space_limit("flights_per_route", days),
        "city_allocation": get_city_allocation(days, cities),
        "sort_by_rating": True,  # Keep top-rated options
        "sort_by_price": True,   # Keep cheapest options for budget
    }
    
    if use_aggressive:
        # More aggressive limits for difficult problems
        config["restaurants_per_city"] = min(10, config["restaurants_per_city"])
        config["attractions_per_city"] = min(12, config["attractions_per_city"])
        config["accommodations_per_city"] = min(6, config["accommodations_per_city"])
    
    return config


def apply_data_pruning(
    data: Dict[str, Any],
    pruning_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply pruning to database data before passing to SMT solver.
    
    This reduces the search space by keeping only top candidates.
    """
    pruned = {}
    
    for key, items in data.items():
        if not isinstance(items, list):
            pruned[key] = items
            continue
        
        # Determine limit based on data type
        if "restaurant" in key.lower():
            limit = pruning_config.get("restaurants_per_city", 15)
        elif "attraction" in key.lower():
            limit = pruning_config.get("attractions_per_city", 20)
        elif "accommodation" in key.lower() or "hotel" in key.lower():
            limit = pruning_config.get("accommodations_per_city", 10)
        elif "flight" in key.lower():
            limit = pruning_config.get("flights_per_route", 5)
        else:
            limit = 50  # Default
        
        # Sort by rating if available, then take top N
        if pruning_config.get("sort_by_rating") and items:
            try:
                # Try to sort by rating
                items = sorted(items, key=lambda x: -(x.get("rating", 0) or 0))
            except (TypeError, AttributeError):
                pass
        
        pruned[key] = items[:limit]
    
    return pruned


# Export configuration for the paper
OPTIMIZATION_CONFIG = {
    "z3_timeout_ms": get_z3_timeout(),
    "relaxation_order": RELAXATION_ORDER,
    "search_space_limits": SEARCH_SPACE_LIMITS,
    "multi_day_scaling": MULTI_DAY_SCALING,
    "complexity_thresholds": {
        "hard": {"days": 7, "constraints": 3},
        "medium": {"days": 5, "constraints": 2},
    },
    "incremental_phases": 3,
}
