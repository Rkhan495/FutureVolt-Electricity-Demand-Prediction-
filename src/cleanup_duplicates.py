"""
One-time cleanup script — checks for and removes duplicate backfilled rows
(Source: "backfill_openmeteo") for dates that were also covered by the live
daily scraper. Does a dry-run count first (always printed), and only deletes
if DRY_RUN is set to False below.

Run this ONCE via a manual GitHub Actions workflow_dispatch, not on a schedule.
"""

import os
import sys
import pymongo
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TARGET_DATE = "11-08-2026"   # format matches your DB: DD-MM-YYYY
DRY_RUN = True                # set to False only after reviewing the counts below

# ---------------------------------------------------------------------------
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    print("ERROR: MONGODB_URI not found in environment variables")
    sys.exit(1)

client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
client.admin.command("ping")
print("Successfully connected to MongoDB!")

db = client.FutureVolt
collection = db["data"]

# ---------------------------------------------------------------------------
# Step 1: Count what's there, broken down by source
# ---------------------------------------------------------------------------
backfilled_count = collection.count_documents({
    "Date": TARGET_DATE,
    "Source": "backfill_openmeteo"
})

live_count = collection.count_documents({
    "Date": TARGET_DATE,
    "Source": {"$exists": False}
})

total_count = collection.count_documents({"Date": TARGET_DATE})

print(f"\n--- Duplicate check for Date: {TARGET_DATE} ---")
print(f"Total documents for this date:        {total_count}")
print(f"Live-scraped documents (no Source):    {live_count}")
print(f"Backfilled documents (Source=openmeteo): {backfilled_count}")

if backfilled_count == 0:
    print("\nNo backfilled duplicates found for this date. Nothing to delete.")
    sys.exit(0)

# Show a sample of what would be deleted, for a sanity check
print("\nSample of documents that would be deleted (up to 3):")
for doc in collection.find({"Date": TARGET_DATE, "Source": "backfill_openmeteo"}).limit(3):
    print(f"  Date={doc.get('Date')} Time={doc.get('Time')} Load={doc.get('Load')} Source={doc.get('Source')}")

# ---------------------------------------------------------------------------
# Step 2: Delete only if DRY_RUN is explicitly False
# ---------------------------------------------------------------------------
if DRY_RUN:
    print(f"\nDRY_RUN is True — no documents were deleted.")
    print(f"Review the counts above. If they look correct, set DRY_RUN = False and re-run.")
else:
    result = collection.delete_many({
        "Date": TARGET_DATE,
        "Source": "backfill_openmeteo"
    })
    print(f"\nDeleted {result.deleted_count} backfilled duplicate documents for {TARGET_DATE}.")

    remaining = collection.count_documents({"Date": TARGET_DATE})
    print(f"Remaining documents for {TARGET_DATE}: {remaining} (should equal the live-scraped count above: {live_count})")

print("\nCleanup script complete.")
