import pandas as pd
import numpy as np

class DepartmentMapper:
    def __init__(self):
        self.mapping = {}
        self.reverse_mapping = {}
        
    def fit(self, dept_ids):
        unique_ids = sorted(list(set(dept_ids)))
        # Map IDs 0 to N-1 to actual depts
        # Reserve index N for "Unknown"
        for i, dept_id in enumerate(unique_ids):
            self.mapping[dept_id] = i
            self.reverse_mapping[i] = dept_id
        
        self.unknown_idx = len(unique_ids)
        self.num_classes = len(unique_ids) + 1 # +1 for the unknown bucket

    def transform(self, dept_ids):
        # Use .get() to return the unknown_idx if the ID isn't found
        return [self.mapping.get(d, self.unknown_idx) for d in dept_ids]

