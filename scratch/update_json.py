import json

file_path = "config/fo_universe.json"

with open(file_path, "r") as f:
    data = json.load(f)

# Update mappings
if "NSE:ATHERENERG-EQ" in data:
    data["NSE:ATHERENERG-EQ"] = "AUTO"
if "NSE:MAHABANK-EQ" in data:
    data["NSE:MAHABANK-EQ"] = "BANKING"
if "NSE:SAGILITY-EQ" in data:
    data["NSE:SAGILITY-EQ"] = "SERVICES"

# Remove NIFTYFPI-EQ
if "NSE:NIFTYFPI-EQ" in data:
    del data["NSE:NIFTYFPI-EQ"]

with open(file_path, "w") as f:
    json.dump(data, f, indent=4)
print("Updated fo_universe.json")
