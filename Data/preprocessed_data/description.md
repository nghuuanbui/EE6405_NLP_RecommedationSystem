### This is the description for the processed data
Output directory
data/
 └── Electronics/
      ├── items.parquet
      ├── train.jsonl
      ├── valid.jsonl
      ├── test.jsonl
      └── mappings.json

🟩 1. items.parquet

A compact table of item text inputs for the item tower.

parent_asin,item_text
B00FSTW88K,[CAT] Computers [TITLE] Centon Electronics Flash Memory Card (S1-SDHU1-32G) [STORE] Centon [AVG_RATING] 4.8 [FEAT] Memory for phone, tablets ... [PATH] Electronics > Computers & Accessories > ... [DETAIL] RAM: 32 GB | Brand: Centon | Series: Cmp-flashmemorycard-sd-u1 | ...
B01ABC1234,[CAT] Computers [TITLE] SanDisk Ultra 64GB SDXC UHS-I [STORE] SanDisk ...
...


Columns

parent_asin: unique item identifier

item_text: concatenated and token-limited description (256 tokens)

*Note: items.jsonl is the json version of items.parquet


🟦 2. train.jsonl

Training triples for user-sequence modeling (e.g., Transformer user tower or retrieval model).

Each line is a JSON object:

{"user_id": "A1B2C3", "user_idx": 0, "history": [512, 18, 9021, 777], "target": 44, "ts": 1588615855070}


Fields

user_id: original string ID

user_idx: integer index

history: list of item indices the user interacted with before time t (max K = 50 by default)

target: index of the next item

ts: timestamp in milliseconds


🟨 3. valid.jsonl

Same format as train.jsonl, but built from validation interactions.

🟧 4. test.jsonl

Same schema as valid.jsonl, typically used for final retrieval or ranking evaluation.

🟪 5. mappings.json

Integer index maps linking everything together:

{
  "item2idx": {
    "B00FSTW88K": 0,
    "B01ABC1234": 1,
    "B098XYZ999": 2,
    ...
  },
  "user2idx": {
    "A1B2C3": 0,
    "Z9Y8X7": 1,
    ...
  }
}