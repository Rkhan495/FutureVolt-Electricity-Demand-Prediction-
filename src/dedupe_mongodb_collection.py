"""
dedupe_mongodb_collection.py

General-purpose duplicate cleanup for a MongoDB collection, keyed on
(Date, Time). Unlike the earlier one-off script (which targeted a single
hardcoded date), this scans the ENTIRE collection and removes duplicates
wherever they exist — useful since duplicates may keep recurring from
whatever process syncs data into MongoDB.

Keeps the FIRST document found per (Date, Time) pair (by insertion order /
_id), removes the rest. Dry-run first; only deletes if DRY_RUN is False.

Run via a manual GitHub Actions workflow_dispatch, or adapt COLLECTION_NAME
and run again anytime duplicates show up.
"""

import os
import sys
import pymongo
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — change COLLECTION_NAME to whichever collection your frontend
# actually reads for the page/chart showing duplicates.
# ---------------------------------------------------------------------------
COLLECTION_NAME = "data"  # e.g. "data" or "FutureData" — confirm which one
DRY_RUN = True

# ---------------------------------------------------------------------------
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    print("ERROR: MONGODB_URI not found in environment variables")
    sys.exit(1)

client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
client.admin.command("ping")
print("Successfully connected to MongoDB!")

db = client.FutureVolt
collection = db[COLLECTION_NAME]

total_docs = collection.count_documents({})
print(f"\n--- Scanning collection '{COLLECTION_NAME}' ({total_docs} total documents) ---")

# Group document _ids by (Date, Time)
seen = defaultdict(list)
cursor = collection.find({}, {"_id": 1, "Date": 1, "Time": 1})
for doc in cursor:
    key = (doc.get("Date"), doc.get("Time"))
    seen[key].append(doc["_id"])

duplicate_groups = {k: v for k, v in seen.items() if len(v) > 1}
total_duplicate_docs = sum(len(v) - 1 for v in duplicate_groups.values())  # extras beyond the first

print(f"Unique (Date, Time) keys: {len(seen)}")
print(f"Keys with duplicates: {len(duplicate_groups)}")
print(f"Total duplicate documents to remove (keeping one per key): {total_duplicate_docs}")

if duplicate_groups:
    print("\nSample duplicate keys (up to 10):")
    for i, (key, ids) in enumerate(duplicate_groups.items()):
        if i >= 10:
            break
        print(f"  Date={key[0]} Time={key[1]} -> {len(ids)} copies")

if total_duplicate_docs == 0:
    print("\nNo duplicates found. Nothing to do.")
    sys.exit(0)

if DRY_RUN:
    print(f"\nDRY_RUN is True — no documents were deleted.")
    print(f"Review the counts above. If correct, set DRY_RUN = False and re-run.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Delete all but the first _id for each duplicated key
# ---------------------------------------------------------------------------
ids_to_delete = []
for key, ids in duplicate_groups.items():
    ids_to_delete.extend(ids[1:])  # keep ids[0], delete the rest

result = collection.delete_many({"_id": {"$in": ids_to_delete}})
print(f"\nDeleted {result.deleted_count} duplicate documents from '{COLLECTION_NAME}'.")

remaining = collection.count_documents({})
print(f"Remaining documents in '{COLLECTION_NAME}': {remaining}")
print("\nDedupe complete.")
