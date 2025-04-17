# 🦅 Apache Arrow Columnar Store Benchmark

This project explores the feasibility of using **Apache Arrow** as a **columnar store** for fast querying and aggregation on nested JSON data serialized to **Parquet**. The goal is to evaluate performance in terms of **latency** and **memory efficiency** during various queries.

## 📂 Dataset

- **Format:** Parquet
- **Rows:** 10 million
- **File size:** 657 MB
- **Peak memory usage:** ~300 MB
- **Structure:** Nested JSON
- **File used:** `logs_optimized.parquet`
- **Data Generation:** Skipped (uses pre-generated file)

## 🚀 Queries Executed

| Query                        | Description                             | Rows Scanned | Result Size                     | Latency   | Peak Memory |
| ---------------------------- | --------------------------------------- | ------------ | ------------------------------- | --------- | ----------- |
| get_field_values_by_doc_ids  | Level values for 100 doc_ids (5 unique) | 262,144      | 262,144 rows                    | 763.368ms | 30.94 MB    |
| get_field_values             | All source_regions (5 unique)           | 10,000,000   | 10,000,000 rows                 | 1.076s    | 114.11 MB   |
| get_numeric_stats_by_doc_ids | Payload stats for 100 doc_ids           | 262,144      | min=85, max=20464, avg=10799.62 | 740.270ms | 39.62 MB    |
| get_numeric_stats            | All payload stats (count=10M)           | 10,000,000   | min=50, max=20479, avg=10264.00 | 625.174ms | 41.11 MB    |
| get_tag_buckets              | Tag buckets (49 unique tags)            | 10,000,000   | 10,000,000 rows                 | 2.985s    | 419.28 MB   |

## 📊 Notable Observations

- Apache Arrow enables **efficient in-memory processing** with **low latency**, even with nested and large-scale datasets.
- Most queries completed under **1 second**, with the exception of tag bucketing due to higher cardinality and complex structure.
- **Memory usage remained modest** despite 10M-row scans — showcasing Arrow’s memory efficiency.

## 🛠️ Technologies Used

- [Apache Arrow](https://arrow.apache.org/)
- [Parquet](https://parquet.apache.org/)
- Rust (with `arrow`, `parquet` crates)

## 🧪 Next Steps

- Benchmark against row-based formats
- Evaluate filtering + projection performance
- Test scalability across 100M+ records
- Integrate Arrow compute kernels for more complex analytics
