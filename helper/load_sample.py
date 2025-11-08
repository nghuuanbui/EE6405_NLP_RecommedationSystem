import json 
pprint = lambda x: print(json.dumps(x, indent=2)) if isinstance(x, dict) else display(x)

file = "./Data/meta_Electronics.jsonl"

index = 4  # specify the line you want (0-based)

with open(file, "r") as fp:
    for i, line in enumerate(fp):
        if i == index:
            sample = json.loads(line)
            break

pprint(sample)