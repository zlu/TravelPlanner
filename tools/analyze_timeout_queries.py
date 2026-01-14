#!/usr/bin/env python3
"""Analyze queries that are timing out in SMT solver."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

def main():
    dataset = load_dataset('osunlp/TravelPlanner', 'validation', 
                           download_mode='reuse_cache_if_exists')['validation']
    
    # Successful queries (from our run): 7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
    # Timeout/failed: 44, 45, 46, 47, 48, 49, 50, 51, etc.
    
    print("=== SUCCESSFUL QUERIES ===")
    successful_indices = [7, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    for idx in successful_indices[:5]:  # Just first 5
        item = dataset[idx - 1]
        lc_raw = item.get('local_constraint', {})
        if isinstance(lc_raw, str):
            import json
            try:
                lc = json.loads(lc_raw.replace("'", '"').replace('None', 'null'))
            except:
                lc = eval(lc_raw.replace('null', 'None').replace('true', 'True').replace('false', 'False'))
        else:
            lc = lc_raw
        
        has_constraints = any([lc.get('cuisine'), lc.get('room type'), lc.get('transportation'), lc.get('house rule')])
        print(f"Query {idx}: Level={item.get('level')}, Days={item.get('days')}, Cities={item.get('visiting_city_number')}, Budget=${item.get('budget')}, Constraints={has_constraints}")
    
    print("\n=== TIMEOUT QUERIES ===")
    timeout_indices = [44, 51, 56]
    
    for idx in timeout_indices:
        item = dataset[idx - 1]
        lc_raw = item.get('local_constraint', {})
        if isinstance(lc_raw, str):
            import json
            try:
                lc = json.loads(lc_raw.replace("'", '"').replace('None', 'null'))
            except:
                lc = eval(lc_raw.replace('null', 'None').replace('true', 'True').replace('false', 'False'))
        else:
            lc = lc_raw
        
        print(f'=== Query {idx} ===')
        print(f"Level: {item.get('level')}")
        print(f"People: {item.get('people_number')}, Days: {item.get('days')}")
        print(f"Cities: {item.get('visiting_city_number')}, Budget: ${item.get('budget')}")
        print(f"Origin: {item.get('org')} -> Dest: {item.get('dest')}")
        print(f"Dates: {item.get('date')}")
        print(f"Constraints:")
        print(f"  - Cuisine: {lc.get('cuisine')}")
        print(f"  - Room: {lc.get('room type')}")
        print(f"  - Transport: {lc.get('transportation')}")
        print(f"  - House rule: {lc.get('house rule')}")
        print()

if __name__ == '__main__':
    main()
