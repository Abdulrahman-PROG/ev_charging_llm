import json

input_file = "output_data/ev_training_alpaca.json"
output_file = "output_data/ev_training_alpaca_fixed.json"

# Read JSONL and convert to list
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # Skip empty lines
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line.strip()}\n{e}")
                continue

# Write as JSON array
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"Fixed JSON saved to {output_file}")
