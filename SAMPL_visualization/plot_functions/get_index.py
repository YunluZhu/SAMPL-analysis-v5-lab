import math

def get_index(fr):
    peak_idx = math.ceil(0.5 * fr)
    total_aligned = math.ceil(0.5 * fr) + math.ceil(0.3 * fr) +1
    
    return peak_idx, total_aligned