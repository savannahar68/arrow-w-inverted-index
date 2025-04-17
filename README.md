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

| Query                          | Description                                    | Rows Scanned | Result Size    | Latency   | Peak Memory |
| ------------------------------ | ---------------------------------------------- | ------------ | -------------- | --------- | ----------- |
| `get_field_values_by_doc_ids`  | Fetch field `level` for 100 specific doc_ids   | 10M          | 100 values     | 882.29 ms | 30.36 MB    |
| `get_field_values`             | Extract all values for `source_region`         | 10M          | 10M values     | 1.01 s    | 109.89 MB   |
| `get_numeric_stats_by_doc_ids` | Min/Max/Sum/Avg of `payload_size` for 100 docs | 10M          | 1 stats result | 914.12 ms | 113.08 MB   |
| `get_numeric_stats`            | Global stats for `payload_size`                | 10M          | 1 stats result | 601.55 ms | 114.28 MB   |
| `get_tag_buckets`              | Bucket and count unique `tags`                 | 10M          | 49 unique tags | 2.95 s    | 407.81 MB   |

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
