import re

# Function to extract indices from annoying gurobi constraint
def extract_indices(constraint_string):
    # Use regex to find all occurrences of numbers within square brackets
    indices = re.findall(r'\[(\d+)\]', constraint_string)
    # Convert the indices to integers
    indices = [int(index) for index in indices]
    return indices
