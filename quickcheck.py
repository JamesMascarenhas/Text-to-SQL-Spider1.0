import json, os

DATA_DIR = "data/spider_data"
dev = json.load(open(f"{DATA_DIR}/dev.json"))
missing = {ex["db_id"] for ex in dev
           if not os.path.exists(f"{DATA_DIR}/database/{ex['db_id']}/{ex['db_id']}.sqlite")}

print(len(dev), "dev examples")                  # expect 1034
print(len({ex["db_id"] for ex in dev}), "dev databases")
print("missing:", missing or "none")