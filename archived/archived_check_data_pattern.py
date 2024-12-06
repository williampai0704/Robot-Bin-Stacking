import csv

def check_pattern_deviation(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip header row
        pattern = 1
        output = []
        
        for row_index, row in enumerate(reader, start=1):
            # Check if it's an even-indexed row (0-based)
            if row_index % 2 == pattern:  # 0-based indexing, so odd row numbers are even-indexed
                # Check columns 6-9 (0-based indices 6,7,8,9)
                expected_columns = ['-1.0', '-1.0', '0.0', '0.0']
                actual_columns = row[6:10]
                
                if actual_columns != expected_columns:
                    print(f"Pattern deviation detected at row {row_index + 1}")
                    print(f"Expected: {expected_columns}")
                    print(f"Actual:   {actual_columns}")
                    output.append(row_index + 1)
                    pattern = 0 if pattern == 1 else 1
        print(output)
        if output: return output
    
    print("No pattern deviation found")
    return None

# Usage
deviation_row = check_pattern_deviation('train_data/train_50000_p0.2.csv')