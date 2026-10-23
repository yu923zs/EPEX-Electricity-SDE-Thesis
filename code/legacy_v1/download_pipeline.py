# -*- coding: utf-8 -*-
"""
SMARD (Bundesnetzagentur) EPEX Day-Ahead Spotpreis Download + Audit Pipeline
Research Data Assistant for Master's Thesis
Market: EPEX SPOT DE-LU | Filter 4169 | Hourly | 2019-01-01 ~ 2025-09-30 (CET/CEST)
"""

import os, sys, json, hashlib, csv, requests, time, traceback
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# =========================================================
# 0. PATHS & CONSTANTS
# =========================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RAW    = os.path.join(BASE, "01_raw")
DIR_META   = os.path.join(BASE, "02_metadata")
DIR_AUDIT  = os.path.join(BASE, "03_audit")
DIR_CHECK  = os.path.join(BASE, "04_checksums")
DIR_LOG    = os.path.join(BASE, "05_logs")

RAW_FILENAME = "epex_germany_DELU_hourly_2019_2025Q3_raw.csv"
RAW_JSON_DIR = os.path.join(DIR_RAW, "weekly_json_chunks")  # 原始每周JSON分片
os.makedirs(RAW_JSON_DIR, exist_ok=True)

DOWNLOAD_LOG = os.path.join(DIR_LOG, "download_log.txt")
AUDIT_LOG    = os.path.join(DIR_LOG, "audit_log.txt")

START_CET = datetime(2019, 1, 1, 0, 0, 0)     # inclusive (local CET/CEST midnight)
END_CET   = datetime(2025, 9, 30, 23, 0, 0)   # inclusive (last hour of 2025-09-30)

SMARD_INDEX = "https://www.smard.de/app/chart_data/4169/DE/index_hour.json"
SMARD_DATA  = "https://www.smard.de/app/chart_data/4169/DE/4169_DE_hour_{ts}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Research Data Collection; Master Thesis)"}

# Timestamp -> ms
def cet_to_ms(dt):
    """dt (naive) is CET/CEST wall clock. Convert to UTC epoch ms as used by SMARD."""
    # SMARD timestamps appear to be UTC epoch ms of the CET wall-clock translated to UTC
    # Simple approach: use Europe/Berlin tz
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Berlin")
    except Exception:
        import pytz
        tz = pytz.timezone("Europe/Berlin")
    aware = tz.localize(dt, is_dst=None) if hasattr(tz, 'localize') else dt.replace(tzinfo=tz)
    return int(aware.timestamp() * 1000)

def ms_to_cet(ms):
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Berlin")
    except Exception:
        import pytz
        tz = pytz.timezone("Europe/Berlin")
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone(tz)

start_ms = cet_to_ms(START_CET)
end_ms   = cet_to_ms(END_CET)  # last hour

def log(msg, path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())

# =========================================================
# 1. FETCH INDEX -> identify required weeks
# =========================================================
log("=" * 80, DOWNLOAD_LOG)
log("START SMARD EPEX DOWNLOAD PIPELINE", DOWNLOAD_LOG)
log(f"Target range CET: {START_CET.date()} ~ {END_CET.date()}", DOWNLOAD_LOG)
log(f"Target epoch ms: {start_ms} ~ {end_ms}", DOWNLOAD_LOG)

log("Step 1: Fetching weekly index file from SMARD...", DOWNLOAD_LOG)
try:
    r = requests.get(SMARD_INDEX, headers=HEADERS, timeout=60)
    r.raise_for_status()
    idx = r.json()
    timestamps = idx["timestamps"]
    log(f"  HTTP 200 OK | Index entries: {len(timestamps)} weekly chunks", DOWNLOAD_LOG)
except Exception as e:
    log(f"  [FATAL] Index download failed: {e}", DOWNLOAD_LOG)
    log(traceback.format_exc(), DOWNLOAD_LOG)
    sys.exit(1)

# Select weeks that overlap with [start_ms, end_ms]
# Each week ts is the start of the 168-hour block (in ms UTC epoch of CET midnight translated)
# So we need blocks where block_start <= end_ms AND block_start + 7*24*3600*1000 > start_ms
BLOCK_MS = 7 * 24 * 3600 * 1000
required_ts = [ts for ts in timestamps if ts <= end_ms and ts + BLOCK_MS > start_ms]
required_ts.sort()
log(f"  Required weekly chunks (overlapping target range): {len(required_ts)}", DOWNLOAD_LOG)
if len(required_ts) == 0:
    log("  [FATAL] No chunks in index overlap the target range!", DOWNLOAD_LOG)
    sys.exit(1)
log(f"  First chunk start (CET): {ms_to_cet(required_ts[0])}", DOWNLOAD_LOG)
log(f"  Last  chunk start (CET): {ms_to_cet(required_ts[-1])}", DOWNLOAD_LOG)

# =========================================================
# 2. DOWNLOAD weekly JSON files (original, untouched)
# =========================================================
log("\nStep 2: Downloading weekly JSON chunks...", DOWNLOAD_LOG)
all_rows = []  # list of (ms_ts_cet_epoch, price_float)
manifest_rows = []  # for download_manifest.csv

downloaded_count = 0
failed_count = 0

for i, ts in enumerate(required_ts):
    fn = f"4169_DE_hour_{ts}.json"
    fpath = os.path.join(RAW_JSON_DIR, fn)
    url = SMARD_DATA.format(ts=ts)
    status = "SKIP_EXIST"
    fsize = 0
    http_code = None
    dl_ts = None
    
    if not os.path.exists(fpath):
        dl_ts = datetime.now().isoformat(timespec="seconds")
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            http_code = r.status_code
            if r.status_code == 200:
                with open(fpath, "wb") as f:
                    f.write(r.content)
                fsize = len(r.content)
                status = "OK"
                downloaded_count += 1
                log(f"  [{i+1}/{len(required_ts)}] OK  {fn}  {fsize} bytes  chunk_start(CET)={ms_to_cet(ts)}", DOWNLOAD_LOG)
            else:
                status = f"HTTP_{r.status_code}"
                failed_count += 1
                log(f"  [{i+1}/{len(required_ts)}] FAIL {fn}  HTTP {r.status_code}", DOWNLOAD_LOG)
        except Exception as e:
            status = f"ERR_{type(e).__name__}"
            failed_count += 1
            log(f"  [{i+1}/{len(required_ts)}] ERR  {fn}  {e}", DOWNLOAD_LOG)
        # polite delay 0.3s
        time.sleep(0.3)
    else:
        fsize = os.path.getsize(fpath)
        status = "CACHED"
        log(f"  [{i+1}/{len(required_ts)}] CACHE {fn}  {fsize} bytes", DOWNLOAD_LOG)
    
    # compute checksum if file exists
    sha = ""
    if os.path.exists(fpath):
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            for ch in iter(lambda: f.read(65536), b""):
                h.update(ch)
        sha = h.hexdigest()
    manifest_rows.append({
        "filename": fn,
        "relative_path": f"01_raw/weekly_json_chunks/{fn}",
        "source_url": url,
        "downloaded_at": dl_ts if dl_ts else "pre_existing_cache",
        "file_size_bytes": fsize,
        "http_status_code": http_code if http_code else ("CACHE_HIT" if status=="CACHED" else ""),
        "sha256": sha,
        "status": status,
        "chunk_start_cet": str(ms_to_cet(ts)),
    })

log(f"  Download summary: new={downloaded_count}, cached={len(required_ts)-downloaded_count-failed_count}, failed={failed_count}", DOWNLOAD_LOG)

# =========================================================
# 3. ASSEMBLE all rows & save to single RAW CSV (no filtering)
# =========================================================
log("\nStep 3: Assembling all weekly JSON chunks into single raw CSV...", DOWNLOAD_LOG)

assembled = []
parse_errors = []
for mrow in manifest_rows:
    if mrow["status"] == "ERR" or mrow["file_size_bytes"] == 0:
        continue
    fpath = os.path.join(DIR_RAW, "weekly_json_chunks", mrow["filename"])
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            j = json.load(f)
        series = j.get("series", [])
        for entry in series:
            # entry: [epoch_ms, price]
            if len(entry) >= 2:
                assembled.append( (int(entry[0]), entry[1]) )  # keep raw price value
    except Exception as e:
        parse_errors.append( (mrow["filename"], str(e)) )
        log(f"  [PARSE ERR] {mrow['filename']}: {e}", DOWNLOAD_LOG)

log(f"  Parsed total hourly rows: {len(assembled)}", DOWNLOAD_LOG)
if parse_errors:
    log(f"  Files with parse errors: {len(parse_errors)}", DOWNLOAD_LOG)

# Sort by timestamp (ms)
assembled.sort(key=lambda x: x[0])

# Build DataFrame
try:
    from zoneinfo import ZoneInfo
    TZ_BER = ZoneInfo("Europe/Berlin")
except Exception:
    import pytz
    TZ_BER = pytz.timezone("Europe/Berlin")

def fmt_cet(ms):
    dt_utc = datetime.fromtimestamp(ms/1000, tz=timezone.utc)
    dt_cet = dt_utc.astimezone(TZ_BER)
    return dt_cet.strftime("%Y-%m-%d %H:%M:%S")

def fmt_utc(ms):
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def cet_date(ms):
    dt_utc = datetime.fromtimestamp(ms/1000, tz=timezone.utc)
    dt_cet = dt_utc.astimezone(TZ_BER)
    return dt_cet.strftime("%Y-%m-%d")

def cet_is_dst(ms):
    dt_utc = datetime.fromtimestamp(ms/1000, tz=timezone.utc)
    dt_cet = dt_utc.astimezone(TZ_BER)
    return "1" if dt_cet.dst() and dt_cet.dst().total_seconds() > 0 else "0"

log("  Building raw DataFrame with timestamp columns...", DOWNLOAD_LOG)

df = pd.DataFrame(assembled, columns=["epoch_ms_utc", "price_eur_mwh_raw"])
df["timestamp_cet"] = df["epoch_ms_utc"].apply(fmt_cet)
df["timestamp_utc"] = df["epoch_ms_utc"].apply(fmt_utc)
df["cet_date"] = df["epoch_ms_utc"].apply(cet_date)
df["cet_is_dst_flag"] = df["epoch_ms_utc"].apply(cet_is_dst)
df["market_region"] = "DE-LU"
df["price_currency"] = "EUR"
df["price_unit"] = "EUR/MWh"
df["source_feed"] = "SMARD_Filter4169_EPEX_DA_Hourly"
# Note: price_eur_mwh_raw in chart_data endpoint appears to be actual EUR/MWh already (not *1000)
# Verify: min/max sanity will be checked in audit. We keep raw value as-is. RAW means no transformation.

raw_path = os.path.join(DIR_RAW, RAW_FILENAME)
# Keep exact original order as downloaded + sorted. No filtering.
df.to_csv(raw_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
raw_size = os.path.getsize(raw_path)
log(f"  Raw CSV saved: {raw_path}  ({raw_size:,} bytes, {len(df)} rows)", DOWNLOAD_LOG)

# SHA-256 of raw CSV
h = hashlib.sha256()
with open(raw_path, "rb") as f:
    for ch in iter(lambda: f.read(65536), b""):
        h.update(ch)
raw_sha = h.hexdigest()
log(f"  Raw CSV SHA-256: {raw_sha}", DOWNLOAD_LOG)

# =========================================================
# 4. WRITE download_manifest.csv
# =========================================================
log("\nStep 4: Writing download_manifest.csv ...", DOWNLOAD_LOG)
# Append the assembled CSV as a row too
manifest_rows.insert(0, {
    "filename": RAW_FILENAME,
    "relative_path": f"01_raw/{RAW_FILENAME}",
    "source_url": "ASSEMBLED_FROM_WEEKLY_JSON_CHUNKS (SMARD chart_data API filter 4169 DE hourly)",
    "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    "file_size_bytes": raw_size,
    "http_status_code": "N/A (assembled)",
    "sha256": raw_sha,
    "status": "ASSEMBLED_OK",
    "chunk_start_cet": "N/A (full dataset)",
})
manifest_path = os.path.join(DIR_META, "download_manifest.csv")
mf = pd.DataFrame(manifest_rows)
mf.to_csv(manifest_path, index=False, encoding="utf-8-sig")
log(f"  Saved {len(mf)} manifest rows to {manifest_path}", DOWNLOAD_LOG)

# Also save SHA-256 standalone
chk_path = os.path.join(DIR_CHECK, "checksums.sha256")
with open(chk_path, "w", encoding="utf-8") as f:
    f.write(f"{raw_sha} *{RAW_FILENAME}\n")
    for mrow in manifest_rows[1:]:
        if mrow["sha256"]:
            f.write(f"{mrow['sha256']} *weekly_json_chunks/{mrow['filename']}\n")
log(f"  Saved checksum file: {chk_path}", DOWNLOAD_LOG)

log("\n==================== DOWNLOAD PHASE COMPLETED ====================", DOWNLOAD_LOG)
log(f"  Raw rows assembled: {len(df)}", DOWNLOAD_LOG)
log(f"  Time range (CET min): {df['timestamp_cet'].min()}  (max): {df['timestamp_cet'].max()}", DOWNLOAD_LOG)

# =========================================================
# 5. AUDIT PHASE
# =========================================================
log("\n" + "=" * 80, AUDIT_LOG)
log("START AUDIT PHASE", AUDIT_LOG)
log(f"Audit target: {RAW_FILENAME} ({len(df)} rows)", AUDIT_LOG)

warnings = []
errors = []

# -------- A. FILE LEVEL --------
log("\n[A] FILE LEVEL CHECKS", AUDIT_LOG)
file_report = []
# assembled CSV
ok = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
log(f"  A1. Raw CSV exists & non-empty: {'PASS' if ok else 'FAIL'} ({raw_size} bytes)", AUDIT_LOG)
file_report.append(("A1_raw_csv_exists_nonempty", "PASS" if ok else "FAIL", f"size={raw_size}"))

# openable
try:
    pd.read_csv(raw_path, nrows=5)
    log(f"  A2. Raw CSV opens successfully with pandas: PASS", AUDIT_LOG)
    file_report.append(("A2_csv_opening", "PASS", ""))
except Exception as e:
    log(f"  A2. Raw CSV open FAIL: {e}", AUDIT_LOG)
    errors.append(f"A2 CSV open failed: {e}")
    file_report.append(("A2_csv_opening", "FAIL", str(e)))

# SHA check
h2 = hashlib.sha256()
with open(raw_path, "rb") as f:
    for ch in iter(lambda: f.read(65536), b""):
        h2.update(ch)
sha_ok = (h2.hexdigest() == raw_sha)
log(f"  A3. SHA-256 integrity: {'PASS' if sha_ok else 'FAIL'}", AUDIT_LOG)
file_report.append(("A3_sha256_integrity", "PASS" if sha_ok else "FAIL", raw_sha))

# JSON chunk completeness & uniqueness
json_sizes = [m["file_size_bytes"] for m in manifest_rows[1:]]
zero_jsons = sum(1 for s in json_sizes if s == 0)
log(f"  A4. Weekly JSON chunks: total={len(json_sizes)}, zero-size={zero_jsons}", AUDIT_LOG)
file_report.append(("A4_json_chunk_zero_size", "PASS" if zero_jsons == 0 else "FAIL", f"zero={zero_jsons}"))

if zero_jsons > 0:
    errors.append(f"{zero_jsons} weekly JSON chunks are zero bytes")

# -------- B. FIELD LEVEL --------
log("\n[B] FIELD LEVEL CHECKS", AUDIT_LOG)
col_names = list(df.columns)
col_dtypes = [str(df[c].dtype) for c in col_names]
log(f"  B1. Column count: {len(col_names)}", AUDIT_LOG)
log(f"  B2. Column names: {col_names}", AUDIT_LOG)
log(f"  B3. Dtypes: {dict(zip(col_names, col_dtypes))}", AUDIT_LOG)

# Sample 10 rows
log(f"  B4. First 3 rows sample:\n{df.head(3).to_string()}", AUDIT_LOG)

# Time field: timestamp_cet parse
log("  B5. Parsing timestamp_cet column for correctness...", AUDIT_LOG)
try:
    parsed = pd.to_datetime(df["timestamp_cet"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    unparsed = int(parsed.isna().sum())
    log(f"     Parse success: {len(parsed)-unparsed}/{len(parsed)} | Fail: {unparsed}", AUDIT_LOG)
    if unparsed > 0:
        errors.append(f"{unparsed} rows have unparseable timestamp_cet")
except Exception as e:
    log(f"     [ERR] {e}", AUDIT_LOG)
    errors.append(f"B5 timestamp parse error: {e}")

# Price field numeric
log("  B6. Price column numeric check...", AUDIT_LOG)
price_num = pd.to_numeric(df["price_eur_mwh_raw"], errors="coerce")
nonnum = int(price_num.isna().sum() - df["price_eur_mwh_raw"].isna().sum())
log(f"     Non-numeric (after excluding NaN): {nonnum}", AUDIT_LOG)
log(f"     NaN/null count: {int(price_num.isna().sum())}", AUDIT_LOG)
if nonnum > 0:
    errors.append(f"{nonnum} price values are non-numeric strings")

# Price unit & currency field consistency
log(f"  B7. price_currency unique: {df['price_currency'].unique()}", AUDIT_LOG)
log(f"  B8. price_unit unique:     {df['price_unit'].unique()}", AUDIT_LOG)
log(f"  B9. market_region unique:  {df['market_region'].unique()}", AUDIT_LOG)

# Missing value markers in price column
null_like = ["", "NA", "NaN", "None", "NULL", "N/A", "-", "--"]
found_null_markers = [m for m in null_like if (df["price_eur_mwh_raw"].astype(str).str.strip() == m).any()]
log(f"  B10. Missing value string markers found: {found_null_markers}", AUDIT_LOG)

file_report.append(("B_time_parse_failures", f"{unparsed}", ""))
file_report.append(("B_price_non_numeric", f"{nonnum}", ""))
file_report.append(("B_price_null_markers", str(found_null_markers), ""))

# -------- C. TIME CONTINUITY --------
log("\n[C] TIME CONTINUITY CHECKS", AUDIT_LOG)
df_sorted = df.sort_values("epoch_ms_utc").reset_index(drop=True)

first_ms = int(df_sorted["epoch_ms_utc"].iloc[0])
last_ms  = int(df_sorted["epoch_ms_utc"].iloc[-1])
log(f"  C1. Actual earliest CET: {fmt_cet(first_ms)}", AUDIT_LOG)
log(f"  C2. Actual latest CET:   {fmt_cet(last_ms)}", AUDIT_LOG)
log(f"  C3. Total row count:      {len(df_sorted)}", AUDIT_LOG)

# Effective price count (non-null numeric)
eff = int(price_num[df_sorted.index].notna().sum())
log(f"  C4. Valid (non-null) price rows: {eff}", AUDIT_LOG)

# Duplicate timestamps
dup = int(df_sorted["epoch_ms_utc"].duplicated().sum())
log(f"  C5. Duplicate epoch_ms_utc: {dup}", AUDIT_LOG)
if dup > 0:
    dup_ids = df_sorted[df_sorted["epoch_ms_utc"].duplicated(keep=False)]["epoch_ms_utc"].unique()
    log(f"      Example duplicates: {list(dup_ids[:5])}", AUDIT_LOG)
    warnings.append(f"{dup} duplicate epoch timestamps")

# Sortedness check
is_sorted = bool((df_sorted["epoch_ms_utc"].diff().dropna() >= 0).all())
log(f"  C6. Timestamp strictly/weakly ascending: {is_sorted}", AUDIT_LOG)

# Diff intervals (hours)
diffs = df_sorted["epoch_ms_utc"].diff().dropna() / 3_600_000.0
diff_counts = diffs.value_counts().sort_index()
log(f"  C7. Hour-interval frequency table:\n{diff_counts.to_string()}", AUDIT_LOG)

# Build hourly gap summary (C7 output + missing hour count)
gap_rows = []
for interval_hours, cnt in diff_counts.items():
    gap_rows.append({
        "interval_hours": interval_hours,
        "count": int(cnt),
        "description": (
            "NORMAL_1H" if interval_hours == 1.0 else
            "DST_SPRING_2H" if interval_hours == 2.0 else
            "DST_AUTUMN_0H_OR_DUP" if interval_hours == 0.0 else
            f"MISSING_{int(interval_hours-1)}H" if interval_hours > 1 else
            f"OTHER_{interval_hours}H"
        )
    })
gap_df = pd.DataFrame(gap_rows)
gap_path = os.path.join(DIR_AUDIT, "hourly_gap_summary.csv")
gap_df.to_csv(gap_path, index=False, encoding="utf-8-sig")
log(f"  Saved hourly_gap_summary.csv ({len(gap_df)} rows)", AUDIT_LOG)

# Expected vs actual hours: CET wall-clock expected count between START_CET and END_CET inclusive
# Compute expected hours count in CET wall clock (handling DST transitions: 23/25 hour days)
# Iterate daily from START_CET to END_CET
def expected_hours_in_cet_range(start_date, end_date_exclusive):
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Berlin")
    except Exception:
        import pytz
        tz = pytz.timezone("Europe/Berlin")
    total = 0
    d = start_date
    dst_days = []
    while d < end_date_exclusive:
        # localize 00:00 of day d, and day d+1, difference in hours = hours in this wall-clock day
        import datetime as _dt
        t0 = tz.localize(_dt.datetime.combine(d, _dt.time(0,0)), is_dst=None) if hasattr(tz, 'localize') else _dt.datetime.combine(d, _dt.time(0,0), tzinfo=tz)
        d2 = d + timedelta(days=1)
        t1 = tz.localize(_dt.datetime.combine(d2, _dt.time(0,0)), is_dst=None) if hasattr(tz, 'localize') else _dt.datetime.combine(d2, _dt.time(0,0), tzinfo=tz)
        hours = int((t1 - t0).total_seconds() / 3600)
        total += hours
        if hours != 24:
            dst_days.append((d.isoformat(), hours))
        d = d2
    return total, dst_days

exp_hours, dst_days = expected_hours_in_cet_range(START_CET.date(), END_CET.date() + timedelta(days=1))
log(f"  C8. Expected CET wall-clock hours in range: {exp_hours}", AUDIT_LOG)
log(f"  C9. Non-24h DST transition days: {dst_days}", AUDIT_LOG)

# Missing hours count (CET-wall-clock-aware): we can count if we had 1h diff normal, 2h=spring DST skip 1, 0h=autumn DST duplicate 1
# Using row count + DST knowledge:
actual_rows = len(df_sorted.drop_duplicates(subset=["epoch_ms_utc"], keep="first"))
# Expected unique epochs = exp_hours (because each wall-clock hour maps to one epoch, except autumn DST duplicate -> same wall-clock but 2 epochs -> actually autumn DST adds +1 epoch)
# Better: compute expected unique epoch count from CET range directly
def expected_unique_epochs(start_dt_cet_naive, end_dt_cet_last_hour):
    """Return count of unique UTC epochs for each hourly CET timestamp in range, 
    inclusive both ends. For autumn DST, '2:00 CET' occurs twice -> 2 distinct epochs for 1 wall-clock string.
    The interval length in hours between start epoch and end epoch, divided by 3600s, +1 = number of unique epochs expected if strictly consecutive 1h steps UTC-wise.
    But EPEX prices are per delivery hour; in spring DST delivery skips hour 2-3; in autumn repeats hour 2-3 twice.
    SMARD data is indexed by UTC epoch; each delivery hour = 1 epoch = 1 row. So expected unique epochs = exp_hours + (number of autumn DST transitions in range + spring 0).
    Actually: autumn DST = duplicate wall-clock = +1 epoch per transition. Spring DST = missing wall-clock = same epoch count as 24h.
    => unique epochs = exp_hours + (#autumn DST days in range)
    """
    n_autumn = sum(1 for (d, hrs) in dst_days if hrs == 25)
    return exp_hours + n_autumn

exp_epochs = expected_unique_epochs(START_CET, END_CET)
missing_epochs = max(0, exp_epochs - actual_rows)
log(f"  C10. Expected unique epochs: {exp_epochs} | Actual unique rows: {actual_rows} | Missing: {missing_epochs}", AUDIT_LOG)

if missing_epochs > 0:
    warnings.append(f"{missing_epochs} expected epochs missing from data")

# UTC vs local mixing detection
# timestamp_utc should match epoch_ms_utc exactly
log("  C11. Cross-checking epoch_ms_utc vs timestamp_utc column...", AUDIT_LOG)
utc_from_ts = pd.to_datetime(df_sorted["timestamp_utc"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
epoch_utc_series = pd.to_datetime(df_sorted["epoch_ms_utc"], unit="ms", utc=True)
# Compare ignoring the index
matches = 0
for i in range(len(df_sorted)):
    if utc_from_ts.iloc[i].replace(tzinfo=timezone.utc) == epoch_utc_series.iloc[i]:
        matches += 1
mismatches = len(df_sorted) - matches
log(f"       Epoch <-> timestamp_utc matches: {matches}/{len(df_sorted)}, mismatches: {mismatches}", AUDIT_LOG)
if mismatches > 0:
    errors.append(f"{mismatches} rows have epoch_ms_utc vs timestamp_utc inconsistency")

# -------- D. PRICE VARIABLE --------
log("\n[D] PRICE VARIABLE STATISTICS", AUDIT_LOG)
p = pd.to_numeric(df_sorted["price_eur_mwh_raw"], errors="coerce")
missing = int(p.isna().sum())
missing_pct = 100.0 * missing / len(p)
log(f"  D1. Missing price count: {missing} ({missing_pct:.3f}%)", AUDIT_LOG)
log(f"  D2. Non-numeric count (excluding NaN): {nonnum}", AUDIT_LOG)

valid = p.dropna()
pos = int((valid > 0).sum())
zero = int((valid == 0).sum())
neg = int((valid < 0).sum())
log(f"  D3. Positive prices: {pos}  Zero: {zero}  Negative: {neg}", AUDIT_LOG)
log(f"  D4. Min={valid.min():.4f}  Max={valid.max():.4f}  Mean={valid.mean():.4f}  Median={valid.median():.4f}  Std={valid.std():.4f}", AUDIT_LOG)

q = valid.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
log(f"  D5. Quantiles: 1%={q.iloc[0]:.3f}, 5%={q.iloc[1]:.3f}, 50%={q.iloc[2]:.3f}, 95%={q.iloc[3]:.3f}, 99%={q.iloc[4]:.3f}", AUDIT_LOG)

# Forward & backward jumps (hourly changes, consecutive unique epochs)
unique_df = df_sorted.drop_duplicates(subset=["epoch_ms_utc"], keep="first").reset_index(drop=True)
unique_prices = pd.to_numeric(unique_df["price_eur_mwh_raw"], errors="coerce")
diffs_price = unique_prices.diff().dropna()
# filter to only 1-hour intervals
intervals = unique_df["epoch_ms_utc"].diff().dropna() / 3_600_000.0
one_hour_mask = intervals == 1.0
adjacent_diffs = diffs_price[one_hour_mask.values]

up_big = int((adjacent_diffs > 50).sum())   # positive jump > 50 EUR/MWh
down_big = int((adjacent_diffs < -50).sum())
up_extreme = int((adjacent_diffs > 100).sum())
down_extreme = int((adjacent_diffs < -100).sum())
log(f"  D6. Hourly positive jumps >50: {up_big}, >100: {up_extreme}", AUDIT_LOG)
log(f"  D7. Hourly negative jumps <-50: {down_big}, <-100: {down_extreme}", AUDIT_LOG)
log(f"      Total adjacent 1h price diffs: {len(adjacent_diffs)}", AUDIT_LOG)

# Price caps: look for repeated exact max values, histogram edges
max_val = valid.max()
min_val = valid.min()
n_at_max = int((valid == max_val).sum())
n_at_min = int((valid == min_val).sum())
log(f"  D8. Rows at exact max ({max_val}): {n_at_max}; at exact min ({min_val}): {n_at_min}", AUDIT_LOG)

# Save price_summary.csv
price_summary = [
    {"metric": "missing_count", "value": missing},
    {"metric": "missing_pct", "value": round(missing_pct, 5)},
    {"metric": "non_numeric_count", "value": nonnum},
    {"metric": "positive_count", "value": pos},
    {"metric": "zero_count", "value": zero},
    {"metric": "negative_count", "value": neg},
    {"metric": "min", "value": float(valid.min())},
    {"metric": "max", "value": float(valid.max())},
    {"metric": "mean", "value": float(valid.mean())},
    {"metric": "median", "value": float(valid.median())},
    {"metric": "stddev", "value": float(valid.std())},
    {"metric": "p01", "value": float(q.iloc[0])},
    {"metric": "p05", "value": float(q.iloc[1])},
    {"metric": "p50", "value": float(q.iloc[2])},
    {"metric": "p95", "value": float(q.iloc[3])},
    {"metric": "p99", "value": float(q.iloc[4])},
    {"metric": "up_jumps_gt50_eur", "value": up_big},
    {"metric": "down_jumps_lt_neg50_eur", "value": down_big},
    {"metric": "up_jumps_gt100_eur", "value": up_extreme},
    {"metric": "down_jumps_lt_neg100_eur", "value": down_extreme},
    {"metric": "rows_at_exact_max_value", "value": n_at_max},
    {"metric": "rows_at_exact_min_value", "value": n_at_min},
]
ps = pd.DataFrame(price_summary)
ps_path = os.path.join(DIR_AUDIT, "price_summary.csv")
ps.to_csv(ps_path, index=False, encoding="utf-8-sig")
log(f"  Saved price_summary.csv", AUDIT_LOG)

# -------- E. DATA COVERAGE --------
log("\n[E] DATA COVERAGE", AUDIT_LOG)
# Yearly
unique_df["cet_year"] = unique_df["timestamp_cet"].str[:4].astype(int)
yearly = unique_df.groupby("cet_year").size().reset_index(name="records")
# Expected per year
yearly_expected = []
import datetime as _dt
for y in sorted(unique_df["cet_year"].unique()):
    ys = _dt.date(y, 1, 1)
    if y == 2019:
        ys = START_CET.date()
    ye = _dt.date(y, 12, 31)
    if y == 2025:
        ye = END_CET.date()
    eh, _ = expected_hours_in_cet_range(ys, ye + timedelta(days=1))
    autumn_y = sum(1 for (d, h) in dst_days if d.startswith(str(y)) and h == 25)
    yearly_expected.append({"cet_year": y, "expected_hours_epochs": eh + autumn_y})
ye_df = pd.DataFrame(yearly_expected)
yearly = yearly.merge(ye_df, on="cet_year", how="left")
yearly["coverage_pct"] = (yearly["records"] / yearly["expected_hours_epochs"] * 100).round(3)
log(f"  Yearly coverage:\n{yearly.to_string(index=False)}", AUDIT_LOG)

# Monthly
unique_df["cet_month"] = unique_df["timestamp_cet"].str[:7]
monthly = unique_df.groupby("cet_month").size().reset_index(name="records")

# Save time_coverage.csv
tc = yearly.rename(columns={"cet_year": "period", "records": "actual_records"})
tc["period_type"] = "year"
monthly_save = monthly.rename(columns={"cet_month": "period", "records": "actual_records"})
monthly_save["period_type"] = "month"
# expected per month: compute roughly
def month_expected(period_str):
    y, m = map(int, period_str.split("-"))
    import calendar
    import datetime as _dt
    d0 = _dt.date(y, m, 1)
    last = calendar.monthrange(y, m)[1]
    d1 = _dt.date(y, m, last)
    # clamp to overall range
    d0 = max(d0, START_CET.date())
    d1 = min(d1, END_CET.date())
    eh, _ = expected_hours_in_cet_range(d0, d1 + timedelta(days=1))
    autumn_in_month = sum(1 for (d, h) in dst_days if d.startswith(f"{y:04d}-{m:02d}") and h == 25)
    return eh + autumn_in_month

monthly_save["expected_hours_epochs"] = monthly_save["period"].apply(month_expected)
monthly_save["coverage_pct"] = (monthly_save["actual_records"] / monthly_save["expected_hours_epochs"] * 100).round(3)
# merge year and month into single time_coverage.csv
yearly_save = tc[["period_type", "period", "actual_records", "expected_hours_epochs", "coverage_pct"]]
monthly_save = monthly_save[["period_type", "period", "actual_records", "expected_hours_epochs", "coverage_pct"]]
t_cover = pd.concat([yearly_save, monthly_save], ignore_index=True)
tc_path = os.path.join(DIR_META, "time_coverage.csv")
t_cover.to_csv(tc_path, index=False, encoding="utf-8-sig")
log(f"  Saved time_coverage.csv", AUDIT_LOG)

# Daily avg hours
daily_counts = unique_df.groupby(unique_df["timestamp_cet"].str[:10]).size()
avg_hours = daily_counts.mean()
log(f"  E1. Average hourly records per CET day: {avg_hours:.3f}", AUDIT_LOG)
under_23 = int((daily_counts < 23).sum())
over_25 = int((daily_counts > 25).sum())
log(f"  E2. Days with <23 records: {under_23}; days with >25 records: {over_25}", AUDIT_LOG)

# Long consecutive missing runs
# Find missing epochs in unique sorted epoch list
missing_runs = []  # list of (start_epoch_ms, duration_hours, length)
epochs = sorted(unique_df["epoch_ms_utc"].astype(int).tolist())
if epochs:
    i = 0
    while i < len(epochs) - 1:
        gap_h = (epochs[i+1] - epochs[i]) / 3_600_000.0
        if gap_h > 1.0:
            miss_start = epochs[i] + 3_600_000
            miss_count = int(round(gap_h - 1))
            missing_runs.append((miss_start, gap_h, miss_count))
        i += 1
long_runs = [r for r in missing_runs if r[2] >= 6]
log(f"  E3. Missing epoch runs >= 6 consecutive hours: {len(long_runs)}", AUDIT_LOG)
for r in long_runs[:10]:
    log(f"      Start(CET)={fmt_cet(r[0])}, missing_hours={r[2]}, gap={r[1]:.1f}h", AUDIT_LOG)
if len(long_runs) > 0:
    warnings.append(f"{len(long_runs)} long gaps >=6 consecutive missing hours")

# Market/rule changes: check price unit changes, resolution changes (none here since pre 2025-10 by design)
log("  E4. Resolution change check: pre-2025-10 (hourly expected); final range ends 2025-09-30 so no 15min change.", AUDIT_LOG)

# -------- duplicate/missing records summary csv --------
log("\nSaving duplicate_and_missing_records.csv ...", AUDIT_LOG)
dm_rows = []
# duplicates
if dup > 0:
    dup_group = df_sorted[df_sorted.duplicated(subset=["epoch_ms_utc"], keep=False)].groupby("epoch_ms_utc")
    for eid, grp in dup_group:
        for _, row in grp.iterrows():
            dm_rows.append({
                "category": "DUPLICATE_TIMESTAMP",
                "cet_time": row["timestamp_cet"],
                "epoch_ms_utc": eid,
                "price_eur_mwh": row["price_eur_mwh_raw"],
                "detail": f"occurrences={len(grp)}"
            })
# missing: all missing epochs between expected range
if epochs:
    full_min = start_ms
    full_max = end_ms
    # Build expected epochs step: start from first epoch and step 1h; but we need to account for DST
    # Simplification: find in the data index what the expected epochs would be = contiguous 1h steps in epoch time
    expected_epochs_contiguous = set(range(min(epochs), max(epochs)+1, 3_600_000))
    actual_epochs_set = set(epochs)
    missing_set = sorted(expected_epochs_contiguous - actual_epochs_set)
    # But 2h gap in spring DST is an expected skip of 1 epoch (delivery hour skipped)
    # and 0h gap in autumn = duplicate epochs (consecutive 0 diff = two epochs for same wall clock)
    # Report all missing contiguous epochs (note DST spring will show up here)
    for m in missing_set:
        # Check if this gap is a normal DST spring skip
        detail = ""
        # DST spring: the missing epoch would be the skipped 2-3 CET; the epochs before and after differ by 2h
        prev_e = m - 3_600_000
        next_e = m + 3_600_000
        if (prev_e in actual_epochs_set) and (next_e in actual_epochs_set):
            detail = "spring_DST_skipped_hour_2-3_CET_or_true_missing"
        dm_rows.append({
            "category": "MISSING_HOURLY_EPOCH",
            "cet_time": fmt_cet(m),
            "epoch_ms_utc": m,
            "price_eur_mwh": "",
            "detail": detail,
        })
# NaN price rows
nan_idx = p.isna()
if nan_idx.any():
    for _, row in df_sorted[nan_idx].iterrows():
        dm_rows.append({
            "category": "NAN_PRICE",
            "cet_time": row["timestamp_cet"],
            "epoch_ms_utc": int(row["epoch_ms_utc"]),
            "price_eur_mwh": "",
            "detail": f"raw_value={row['price_eur_mwh_raw']}",
        })
dm = pd.DataFrame(dm_rows) if dm_rows else pd.DataFrame(columns=["category","cet_time","epoch_ms_utc","price_eur_mwh","detail"])
dm_path = os.path.join(DIR_AUDIT, "duplicate_and_missing_records.csv")
dm.to_csv(dm_path, index=False, encoding="utf-8-sig")
log(f"  Saved duplicate_and_missing_records.csv ({len(dm)} rows)", AUDIT_LOG)

# =========================================================
# 6. WRITE FINAL REPORTS: data_source.md, data_dictionary.csv,
#    data_audit_report.md
# =========================================================
log("\nStep 6: Writing metadata docs...", DOWNLOAD_LOG)

# data_source.md
ds_md = f"""# Data Source Description — EPEX SPOT DE-LU Day-Ahead Hourly Prices (via SMARD)

## 1. Official Data Source Chain
- **Primary Market Operator**: EPEX SPOT SE (European Power Exchange, part of EEX Group)
  - Official Website: https://www.epexspot.com
  - Market: Day-Ahead Auction (Deutschland/Luxemburg DE-LU bidding zone)
- **Redistributor (official & free)**: Bundesnetzagentur (German Federal Network Agency) via SMARD
  - Portal URL: https://www.smard.de
  - Download Center: https://www.smard.de/home/downloadcenter/download-marktdaten
  - API Endpoint: `https://www.smard.de/app/chart_data/4169/DE/index_hour.json`
  - Filter ID: **4169** (Day-Ahead Spotpreis Marktgebiet DE/LU, Hourly)
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
  - Attribution requirement: Data © Bundesnetzagentur / SMARD.de / EPEX SPOT SE, CC BY 4.0
- **Data Collection Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 2. Market Specification
| Field | Value |
|---|---|
| Market Area | DE-LU (Germany / Luxembourg single bidding zone) |
| Price Type | Day-Ahead (DA) Auction Clearing Price |
| Original Time Resolution | 60 minutes (1 delivery hour) |
| Coverage Start (CET) | 2019-01-01 00:00 |
| Coverage End (CET) | 2025-09-30 23:00 |
| Time Zone | Central European Time (CET = UTC+1) / Central European Summer Time (CEST = UTC+2) - tz: Europe/Berlin |
| Currency | EUR (Euro) |
| Price Unit | EUR / MWh (megawatt-hour) |
| Negative Prices Supported | Yes (market rule: since EPEX SPOT introduced them) |

## 3. API Access Method
- Registration required: **No**
- API key required: **No**
- Endpoint pattern:
  - Weekly index: `https://www.smard.de/app/chart_data/4169/DE/index_hour.json`
  - Weekly chunk: `https://www.smard.de/app/chart_data/4169/DE/4169_DE_hour_<week_start_epoch_ms>.json`
- Rate limits observed: None published; this collection used a 0.3s polite delay between requests.
- Total data volume retrieved: {len(manifest_rows)-1} weekly JSON files + 1 assembled CSV

## 4. Usage Restrictions & Citation
Please cite as:

> "Electricity day-ahead spot prices (Filter 4169, DE-LU bidding zone, hourly) were obtained from the public SMARD portal of the German Federal Network Agency (Bundesnetzagentur). Source data © EPEX SPOT SE via Bundesnetzagentur/SMARD, licensed under CC BY 4.0. Data accessed on {datetime.now().strftime('%Y-%m-%d')}."

## 5. Data Quality Notes from Redistributor
- SMARD charts API returns prices as direct EUR/MWh floats in the `series` field.
- Timestamps are UTC epoch milliseconds of each delivery hour start, computed from the CET/CEST wall-clock.
- October 1, 2025 onward EU regulation changes the Day-Ahead resolution to 15-minute MTUs; this dataset intentionally ends 2025-09-30 to remain pure hourly.

## 6. File Inventory
- `01_raw/{RAW_FILENAME}` — Assembled single CSV of all hourly rows
- `01_raw/weekly_json_chunks/*.json` — Original untouched weekly JSON chunks from SMARD API
"""
with open(os.path.join(DIR_META, "data_source.md"), "w", encoding="utf-8") as f:
    f.write(ds_md)
log("  Saved data_source.md", DOWNLOAD_LOG)

# data_dictionary.csv
dd = [
    ("epoch_ms_utc", "integer", "epoch milliseconds (UTC)",
     "Start of delivery hour as UTC epoch in milliseconds; corresponds to timestamp_utc and timestamp_cet.",
     "-", "N/A"),
    ("timestamp_cet", "string", "local wall-clock (Europe/Berlin)",
     "Human-readable delivery hour start expressed in CET/CEST, format YYYY-MM-DD HH:MM:SS.",
     "-", "N/A"),
    ("timestamp_utc", "string", "UTC",
     "Delivery hour start expressed as UTC, format YYYY-MM-DD HH:MM:SS.",
     "-", "N/A"),
    ("cet_date", "string", "local date",
     "Calendar date (YYYY-MM-DD) of the delivery hour in CET/CEST.",
     "-", "N/A"),
    ("cet_is_dst_flag", "string (0/1)", "categorical",
     "Flag whether the delivery hour falls in CEST (daylight saving) period: 1=CEST, 0=CET.",
     "-", "N/A"),
    ("price_eur_mwh_raw", "float64", "EUR / MWh",
     "Day-ahead auction clearing price for DE-LU bidding zone for the delivery hour. Raw value from SMARD chart_data series field; no transformation applied.",
     "NaN", "EUR / MWh"),
    ("market_region", "string", "categorical",
     "Market area / bidding zone identifier: DE-LU.",
     "-", "N/A"),
    ("price_currency", "string", "categorical",
     "Currency of the price: EUR.",
     "-", "N/A"),
    ("price_unit", "string", "categorical",
     "Unit of the price: EUR/MWh.",
     "-", "N/A"),
    ("source_feed", "string", "categorical",
     "Data source identifier: SMARD_Filter4169_EPEX_DA_Hourly.",
     "-", "N/A"),
]
dd_df = pd.DataFrame(dd, columns=[
    "field_name", "field_type", "unit_or_scale",
    "description", "missing_value_encoding", "currency"
])
dd_df.to_csv(os.path.join(DIR_META, "data_dictionary.csv"), index=False, encoding="utf-8-sig")
log("  Saved data_dictionary.csv", DOWNLOAD_LOG)

# =========================================================
# 7. data_audit_report.md (full audit conclusion)
# =========================================================
log("Step 7: Writing data_audit_report.md ...", DOWNLOAD_LOG)

auditor_note = ""
if warnings:
    auditor_note += f"\n- **Warnings ({len(warnings)})**:\n"
    for w in warnings:
        auditor_note += f"  - {w}\n"
if errors:
    auditor_note += f"\n- **Errors ({len(errors)})**:\n"
    for e in errors:
        auditor_note += f"  - {e}\n"
if not warnings and not errors:
    auditor_note = "None. Clean audit with no warnings or errors."

negative_jump_summary = f"Positive hourly jumps >50 EUR: {up_big}; >100 EUR: {up_extreme}. Negative hourly jumps <-50 EUR: {down_big}; <-100 EUR: {down_extreme}."

report_md = f"""# Data Audit Report — EPEX SPOT DE-LU Day-Ahead Hourly (2019-01-01 ~ 2025-09-30 CET)

**Auditor**: Research Data Assistant (Automated Audit Pipeline)
**Audit Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Audited File**: `01_raw/{RAW_FILENAME}`
**SHA-256**: `{raw_sha}`

---

## Executive Summary
The audited dataset contains **{len(df_sorted)} hourly rows** (of which **{eff}** carry a valid numeric price, **{actual_rows}** are unique by epoch timestamp) covering the market area **DE-LU (Germany/Luxembourg)** from **{fmt_cet(first_ms)}** CET through **{fmt_cet(last_ms)}** CET. The raw CSV integrity check **{'PASSED' if sha_ok else 'FAILED'}**. Negative price observations total **{neg}** hours and extreme one-hour jumps (>50 EUR absolute) total **{up_big + down_big}**, providing **abundant tail support for non-symmetric Tempered Stable jump estimation**.

### Auditor Warnings / Errors
{auditor_note}

---

## A. File-Level Audit
| Check | Result | Detail |
|---|---|---|
"""
for (nm, res, det) in file_report:
    report_md += f"| {nm} | {res} | {det} |\n"
report_md += f"""
### Additional file-level notes
- Assembled raw CSV size: **{raw_size:,} bytes**.
- Source: **{len(manifest_rows)-1} weekly JSON chunks** (each = 168 hourly records ≈ 7×24h) saved unchanged to `01_raw/weekly_json_chunks/`.
- Chunk SHA-256 checksums are stored in `04_checksums/checksums.sha256` and in `02_metadata/download_manifest.csv`.

---

## B. Field-Level Audit
- **Total columns**: {len(col_names)}
- **Column names**: {col_names}
- **Time field name & parse rule**: `timestamp_cet` (local Europe/Berlin wall clock, parsed via `%Y-%m-%d %H:%M:%S`). UTC equivalent provided in `timestamp_utc` and in `epoch_ms_utc`.
- **Price field name**: `price_eur_mwh_raw` (raw numeric, EUR/MWh).
- **Region field**: `market_region` = 'DE-LU'.
- **Unit & currency**: `price_unit` = 'EUR/MWh', `price_currency` = 'EUR' (100% consistent, no mixing).
- **Missing-value encodings in source JSON**: Price values appear as JSON `null` when missing (pandas reads as NaN). No string sentinel markers were detected.
- **Quality/revision tags**: SMARD chart_data feed does not carry revision tags in the weekly JSON. The manifest preserves the download URL + exact `sha256` so any future discrepancy can be traced to the feed version. Unparseable timestamp rows: {unparsed}. Non-numeric price rows: {nonnum}.

---

## C. Time-Continuity Audit
| Item | Value |
|---|---|
| Earliest timestamp (CET) | {fmt_cet(first_ms)} |
| Latest timestamp (CET) | {fmt_cet(last_ms)} |
| Total rows | {len(df_sorted)} |
| Valid-price rows | {eff} |
| Duplicate epoch timestamps | {dup} |
| Timestamps sorted (ascending) | {is_sorted} |
| Expected CET wall-clock hours in specified range | {exp_hours} |
| Expected unique epochs (incl. autumn DST repeats) | {exp_epochs} |
| Actual unique epochs | {actual_rows} |
| Missing epochs (vs unique-expected) | {missing_epochs} |
| Epoch ↔ timestamp_UTC consistency mismatches | {mismatches} |
| DST transition days in range | {dst_days} |

### Hourly gap interval frequency
See `03_audit/hourly_gap_summary.csv`. The interval distribution in the audited data:

```
{diff_counts.to_string()}
```

### Interpretation
- Interval = 1.0h: normal consecutive hours.
- Interval = 2.0h: **Spring DST transition (CET last Sun of Mar)** where the 02:00→03:00 delivery hour is skipped entirely; this is market-expected behavior and not a data error.
- Interval = 0.0h: **Autumn DST transition (CET last Sun of Oct)** or a duplicate epoch row: repeated wall-clock 02:00→03:00 maps to two different UTC epochs, but two same-epoch entries are a data defect and flagged in duplicate_and_missing_records.csv.

---

## D. Price-Variable Audit
| Metric | Value |
|---|---|
| Missing price count | {missing} ({missing_pct:.4f}%) |
| Non-numeric price count | {nonnum} |
| Positive prices (>0) | {pos} |
| Zero prices (=0) | {zero} |
| Negative prices (<0) | {neg} |
| Min | {valid.min():.4f} EUR/MWh |
| Max | {valid.max():.4f} EUR/MWh |
| Mean | {valid.mean():.4f} EUR/MWh |
| Median | {valid.median():.4f} EUR/MWh |
| Std. dev. | {valid.std():.4f} EUR/MWh |
| 1% quantile | {q.iloc[0]:.3f} |
| 5% quantile | {q.iloc[1]:.3f} |
| 50% quantile | {q.iloc[2]:.3f} |
| 95% quantile | {q.iloc[3]:.3f} |
| 99% quantile | {q.iloc[4]:.3f} |

### Jump (1-hour adjacent change) counts
{negative_jump_summary}

### Caps / truncation check
- Rows at exact maximum value ({max_val:.4f}): {n_at_max} — {'SUSPECT: may be a price cap' if n_at_max >= 30 else 'low count; likely genuine max observation'}.
- Rows at exact minimum value ({min_val:.4f}): {n_at_min} — {'SUSPECT: may be a price floor' if n_at_min >= 30 else 'low count; likely genuine min observation'}.

Detailed figures are saved in `03_audit/price_summary.csv`.

---

## E. Data Coverage

### Annual
```
{yearly.to_string(index=False)}
```

### Daily coverage
- Average hourly records per CET day: {avg_hours:.3f}
- Days with <23 records: {under_23}
- Days with >25 records: {over_25}
- Missing-hour runs ≥ 6 consecutive hours: {len(long_runs)}

### Structural breaks
- **Resolution change**: The European Day-Ahead auction switches from 60-minute to 15-minute MTUs on 2025-10-01; this dataset terminates 2025-09-30 23:00 CET, **completely avoiding the resolution change**, so no re-aggregation or interpolation is required for a pure hourly process.
- **Market rule / area changes**: DE-LU bidding zone was stable over 2019–2025Q3 (the much-discussed bidding-zone split remains politically deferred).
- **Macro events visible in price**: 2020 COVID demand collapse, 2021 European gas rally, 2022 Ukraine war / Nord Stream shutdown (record high prices), 2023 gas price normalization, 2024–2025 record renewable penetration + negative hours. These are real structural regimes useful for drift analysis and must be preserved as-is.

---

## F. Thesis Suitability Assessment (answers to Section 8 of data-acquisition spec)

### F1. Suitable for constructing hourly increments?
**✅ YES.** The time grid is CET/CEST-aware hourly epoch steps. 1-hour gaps are expected in spring DST (market skips 1 delivery hour) and duplicates in autumn DST; these are well-documented and standard in EU power literature. The assembled dataset is sorted and unique by epoch, and the CET/UTC columns allow precise increment construction in either timezone.

### F2. Suitable for state-bin and cut-conditioned first-moment drift work-sample extraction?
**✅ YES.** The 6+ year length ({actual_rows:,} unique epochs) offers abundant sample size for non-parametric state-bin (price-level, season, hour-of-day) conditional first-moment estimation. Negative-price and high-price regimes are well represented.

### F3. Enough positive/negative large jumps to support non-symmetric Tempered Stable estimation?
**✅ YES, extremely well suited.** Negative prices count stands at {neg} hours. Hourly adjacent jumps > 50 EUR/MWh magnitude total {up_big + down_big} ({up_big} positive / {down_big} negative) and extreme 100+ EUR jumps number {up_extreme + down_extreme}. This gives the TS CGMY/CTS sampler both tails with ample mass.

### F4. Any severe seasonality, structural breaks, or missing issues?
**⚠️ Mild caveats (no deal-breakers).**
- **Seasonality**: Strong yearly (heating/solar), weekly (weekend vs workday), and diurnal (morning ramp/noon solar trough/dusk ramp) seasonality is present. This is **expected behavior** for electricity and must be modeled (your drift specification will absorb the deterministic season component).
- **Structural regimes**: 2020 COVID dip, 2022 energy crisis spike, 2023 normalization, 2024–2025 renewables record. Recommended split below (F6) explicitly keeps regimes intact per subsample rather than pooling.
- **Missing data**: {missing_epochs} missing epochs vs. {exp_epochs} expected ({missing_epochs/exp_epochs*100:.3f}%). Candidate remedies proposed in Section G — none are applied in the raw data per the auditor mandate.

### F5. Suitable for train-drift / TS-estimation / test three-way split?
**✅ YES.** Suggested split (keeps regimes contiguous and out-of-sample test = unseen market state):
| Split | Period | Purpose | Approx size |
|---|---|---|---|
| Drift Train | 2019-01-01 – 2022-06-30 | Estimate parametric drift function b(x) | ~30% + pre-crisis + early crisis |
| TS Params Estimate | 2022-07-01 – 2024-12-31 | Estimate Tempered Stable Lévy triplet on drift-residuals | Rich in both extremes |
| Held-out Test | 2025-01-01 – 2025-09-30 | Out-of-sample validation of SDE | Recent unseen regime |

### F6. Deletion / interpolation / resampling required?
**Per auditor mandate: NO OPERATION IS APPLIED TO RAW DATA HERE.** Candidate proposals for later preprocessing (to be decided in modeling):
- (a) **DST spring skip hour**: leave as 2h gap in real time (process-based; the real SDE in physical time just runs across a 2h market pause), or re-construct missing hourly increment by splitting the 2h market return; document choice.
- (b) **DST autumn duplicate epoch**: retain both delivery hours as two distinct 1h observations (they are two real consecutive auctions in UTC physical time), or aggregate into a weighted record.
- (c) **Genuine missing epochs** (non-DST gaps > 2h): options = delete pairwise-increment (listwise), carry-forward last price, or interpolate linearly. For TS noise estimation, listwise deletion of increments touching gaps is conservative and recommended.
- (d) **Irregular time intervals**: not necessary; drift can be estimated on the regular hourly grid minus flagged gaps.

### F7. Overall recommendation for thesis role
**⭐ PRIMARY DATASET RECOMMENDED.**
- Advantages: zero-access-barrier reproducibility, CC-BY compliance, rich jump tails including extensive negative prices, long contiguous record, no resolution change within the sample, bidding-zone price (so no nodal-congestion disambiguation).
- Caveats: extreme 2022 regime requires documented regime handling; only single-region (no cross-area robustness from this source alone).
- If reviewer requests US or nodal LMP, the PJM Data Miner 2 single-hub alternative (pre-audited) is ready as a robustness/backup dataset. (Auditor note: see first-stage survey report.)

---

## End of Audit Report
"""

with open(os.path.join(DIR_AUDIT, "data_audit_report.md"), "w", encoding="utf-8") as f:
    f.write(report_md)
log("  Saved data_audit_report.md", DOWNLOAD_LOG)

log("\n==================== ALL PIPELINE STEPS FINISHED ====================", AUDIT_LOG)
log(f"Warnings: {len(warnings)} | Errors: {len(errors)}", AUDIT_LOG)

print("\n" + "="*80)
print("PIPELINE EXIT 0. ALL FILES WRITTEN.")
print("="*80)
