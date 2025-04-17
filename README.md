# 🦅 Apache Arrow Columnar Store Benchmark

This project explores the feasibility of using **Apache Arrow** as a **columnar store** for fast querying and aggregation on nested JSON data serialized to **Parquet**. The goal is to evaluate performance in terms of **latency** and **memory efficiency** during various queries.

## 📂 Dataset

- **Format:** Parquet
- **Rows:** 10 million
- **Raw data size** 8.1gb
- **File size:** 600 MB
- **Peak memory usage:** ~300 MB
- **Structure:** Nested JSON
- **File used:** `logs_optimized.parquet`
- **Data Generation:** Skipped (uses pre-generated file)

## 🚀 Queries Executed

| Query                        | Description                             | Rows Scanned | Latency   | Peak Memory |
| ---------------------------- | --------------------------------------- | ------------ | --------- | ----------- |
| get_field_values_by_doc_ids  | Level values for 100 doc_ids (5 unique) | 524,288      | 749.207ms | 29.50 MB    |
| get_field_values             | All source_regions (5 unique)           | 10,000,000   | 998.085ms | 111.28 MB   |
| get_numeric_stats_by_doc_ids | Payload stats for 100 doc_ids           | 524,288      | 738.985ms | 36.81 MB    |
| get_numeric_stats            | All payload stats (count=10M)           | 10,000,000   | 588.706ms | 38.20 MB    |
| get_tag_buckets              | Tag buckets (49 unique tags)            | 10,000,000   | 2.986s    | 423.19 MB   |

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
