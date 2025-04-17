// src/main.rs
use arrow::array::{
    Array,
    ArrayRef,
    AsArray, // Trait needed for .as_primitive(), .as_string(), .as_list(), etc.
    BooleanArray,
    DictionaryArray,
    Int64Array,
    ListArray,
    RecordBatchReader, // Keep this trait import
    StringArray,
    StructArray,
    TimestampMicrosecondArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::kernels::filter;
use arrow::datatypes::{DataType, Field, Fields, Int32Type, Schema, TimeUnit}; // Assuming Int32 keys for dictionary
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use memmap2::Mmap;
use parquet::arrow::{
    arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    arrow_writer::ArrowWriter,
    ProjectionMask, // Keep this import
};
use parquet::basic::{Compression, Encoding};
use parquet::file::{
    metadata::ParquetMetaData,
    properties::{EnabledStatistics, WriterProperties},
    reader::FileReader, // Keep this import
    serialized_reader::SerializedFileReader,
};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::thread::sleep;
// Removed Mutex and sleep as they weren't used in the provided snippet
use std::time::{Duration as StdDuration, Instant}; // Keep StdDuration if needed elsewhere
use thiserror::Error;
use uuid::Uuid;

// Import the allocator and stats module
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod query_stats; // Assuming query_stats.rs exists in src/
use query_stats::*;

// --- Error Handling (Unchanged) ---
#[derive(Error, Debug)]
pub enum ArrowError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Conversion error: {0}")]
    Conversion(String),

    #[error("Field not found: {0}")]
    FieldNotFound(String),

    #[error("Unsupported type for operation: {0}")]
    UnsupportedType(String),
}

type Result<T> = std::result::Result<T, ArrowError>;

// --- Struct Definitions (Unchanged) ---
#[derive(Serialize, Deserialize, Debug, Clone)]
struct LogSource {
    ip: String,
    host: String,
    region: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct UserMetrics {
    login_time_ms: i64,
    clicks: i64,
    active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct User {
    id: String,
    session_id: String,
    metrics: UserMetrics,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Answer {
    #[serde(rename = "nxDomain")]
    nx_domain: bool,
    response_time_ms: i64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LogRecord {
    doc_id: i64,
    timestamp: String,
    level: String,
    message: String,
    source: LogSource,
    user: User,
    payload_size: i64,
    tags: Vec<String>,
    answers: Vec<Answer>,
    processed: bool,
}

// --- Data Generation (Optimized with Rayon - Unchanged) ---
fn generate_random_log_record(i: usize, base_time: DateTime<Utc>) -> LogRecord {
    let mut rng = rand::thread_rng();
    let levels = ["info", "warn", "error", "debug", "trace"];
    let regions = [
        "us-east-1",
        "eu-west-1",
        "eu-west-2",
        "ap-south-1",
        "us-west-2",
    ];
    let hosts = (1..=20)
        .map(|n| format!("server-{}.region.local", n))
        .collect::<Vec<_>>();
    let offset_ms = rng.gen_range(-30000..30000);
    let timestamp = base_time + chrono::Duration::milliseconds(offset_ms);
    let answers_len = rng.gen_range(0..=3);
    let answers = (0..answers_len)
        .map(|_| Answer {
            nx_domain: rng.gen_bool(0.3),
            response_time_ms: rng.gen_range(5..150),
        })
        .collect::<Vec<_>>();
    LogRecord {
        doc_id: i as i64,
        timestamp: timestamp.to_rfc3339(),
        level: levels[rng.gen_range(0..levels.len())].to_string(),
        message: format!("Log message {} for record {}", Uuid::new_v4(), i),
        source: LogSource {
            ip: format!("10.0.{}.{}", rng.gen_range(1..255), rng.gen_range(1..255)),
            host: hosts[rng.gen_range(0..hosts.len())].clone(),
            region: regions[rng.gen_range(0..regions.len())].to_string(),
        },
        user: User {
            id: format!("user_{}", rng.gen_range(1000..50000)),
            session_id: Uuid::new_v4().to_string(),
            metrics: UserMetrics {
                login_time_ms: rng.gen_range(10..1500),
                clicks: rng.gen_range(0..100),
                active: rng.gen_bool(0.75),
            },
        },
        payload_size: rng.gen_range(50..20_480),
        // Generate fewer unique tags for better dictionary encoding demo
        tags: (0..rng.gen_range(1..8))
            .map(|_| format!("tag_{}", rng.gen_range(1..50))) // Keep original tag generation
            .collect::<Vec<_>>(),
        answers,
        processed: rng.gen_bool(0.9),
    }
}

// Parallel data generation using Rayon (Unchanged)
fn generate_log_records(count: usize) -> Vec<LogRecord> {
    let base_time = Utc::now();
    println!("Generating {} records in parallel...", count);
    let start = Instant::now();
    let records = (0..count)
        .into_par_iter()
        .map(|i| generate_random_log_record(i, base_time))
        .collect();
    println!("Generated {} records in {:?}", count, start.elapsed());
    records
}

// --- Arrow Schema Creation ---
// Use Dictionary encoding for 'level', 'source_region', and 'tags' values
fn create_arrow_schema() -> Schema {
    Schema::new(vec![
        Field::new("doc_id", DataType::Int64, false),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, None), // Store as native timestamp
            true, // Timestamps can be null if parsing fails
        ),
        // Use Dictionary for low-cardinality string fields
        Field::new(
            "level",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("message", DataType::Utf8, false),
        Field::new("source_ip", DataType::Utf8, false), // IPs might be high cardinality
        Field::new("source_host", DataType::Utf8, false), // Hosts might be high cardinality
        // Use Dictionary for low-cardinality string fields
        Field::new(
            "source_region",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("user_id", DataType::Utf8, false), // User IDs likely high cardinality
        Field::new("user_session_id", DataType::Utf8, false), // Session IDs high cardinality
        Field::new("user_metrics_login_time_ms", DataType::Int64, false),
        Field::new("user_metrics_clicks", DataType::Int64, false),
        Field::new("user_metrics_active", DataType::Boolean, false),
        Field::new("payload_size", DataType::Int64, false),
        // Use Dictionary for list items if tags have repeated values
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true, // Items within the list can be null? Assuming not based on generation
            ))),
            true, // The list itself can be null or empty
        ),
        Field::new(
            "answers", // List of Structs
            DataType::List(Arc::new(Field::new(
                "item", // Name of the list element field
                DataType::Struct(Fields::from(vec![
                    Field::new("nx_domain", DataType::Boolean, false),
                    Field::new("response_time_ms", DataType::Int64, false),
                ])),
                true, // Structs within the list can be null? Assuming not.
            ))),
            true, // The list itself can be null or empty
        ),
        Field::new("processed", DataType::Boolean, false),
    ])
}

// --- Create RecordBatch from LogRecords (Handles Dictionary Encoding) ---
fn create_record_batch_from_records(records: &[LogRecord]) -> Result<RecordBatch> {
    if records.is_empty() {
        // Return an empty RecordBatch with the correct schema
        return Ok(RecordBatch::new_empty(Arc::new(create_arrow_schema())));
        // Or return an error:
        // return Err(ArrowError::Conversion("Cannot create RecordBatch from empty records".to_string()));
    }

    let schema = Arc::new(create_arrow_schema());

    // --- Primitive Columns ---
    let doc_ids: Int64Array = records.iter().map(|r| r.doc_id).collect();
    let user_metrics_login_time: Int64Array = records
        .iter()
        .map(|r| r.user.metrics.login_time_ms)
        .collect();
    let user_metrics_clicks: Int64Array = records.iter().map(|r| r.user.metrics.clicks).collect();
    let user_metrics_active: BooleanArray = records
        .iter()
        .map(|r| Some(r.user.metrics.active))
        .collect();
    let payload_sizes: Int64Array = records.iter().map(|r| r.payload_size).collect();
    let processed: BooleanArray = records.iter().map(|r| Some(r.processed)).collect();

    // --- Timestamp Column ---
    let timestamps: TimestampMicrosecondArray = records
        .iter()
        .map(|r| {
            DateTime::parse_from_rfc3339(&r.timestamp)
                .ok()
                .map(|dt| dt.timestamp_micros())
        })
        .collect();

    // --- Simple String Columns (No Dictionary) ---
    let messages: StringArray = records.iter().map(|r| Some(r.message.as_str())).collect();
    let source_ips: StringArray = records.iter().map(|r| Some(r.source.ip.as_str())).collect();
    let source_hosts: StringArray = records
        .iter()
        .map(|r| Some(r.source.host.as_str()))
        .collect();
    let user_ids: StringArray = records.iter().map(|r| Some(r.user.id.as_str())).collect();
    let user_session_ids: StringArray = records
        .iter()
        .map(|r| Some(r.user.session_id.as_str()))
        .collect();

    // --- Dictionary Encoded String Columns ---
    let levels: DictionaryArray<Int32Type> = records
        .iter()
        .map(|r| Some(r.level.as_str())) // Option required by `FromIterator` for DictionaryArray
        .collect();
    let source_regions: DictionaryArray<Int32Type> = records
        .iter()
        .map(|r| Some(r.source.region.as_str()))
        .collect();

    // --- Tags Column (List<Dictionary<String>>) ---
    let mut tag_values_builder = arrow::array::StringBuilder::new();
    let mut tag_offsets = vec![0i32]; // Use i32 offsets for ListArray
    let mut current_offset = 0;

    for record in records {
        for tag in &record.tags {
            tag_values_builder.append_value(tag);
        }
        current_offset += record.tags.len() as i32;
        tag_offsets.push(current_offset);
    }

    // Build the dictionary array for tag values
    let tag_dict_values: DictionaryArray<Int32Type> = tag_values_builder
        .finish() // Finishes the StringBuilder into a StringArray
        .iter() // Iterate over Option<&str>
        .collect(); // Collect into DictionaryArray

    let tag_offsets_buffer = OffsetBuffer::new(tag_offsets.into());
    let tags_field = schema.field_with_name("tags")?.clone(); // Get field definition from schema
    let tags_list_type = match tags_field.data_type() {
        DataType::List(field) => field.as_ref().clone(),
        _ => return Err(ArrowError::FieldNotFound("tags list item".to_string())),
    };

    let tags_array = ListArray::new(
        Arc::new(tags_list_type), // Use the field obtained from schema
        tag_offsets_buffer,
        Arc::new(tag_dict_values) as ArrayRef, // Values are the dictionary array
        None, // Nullability bitmap for the list itself (can be None if no null lists)
    );

    // --- Answers Column (List<Struct>) ---
    let mut answer_offsets = vec![0i32]; // Use i32 offsets
    let mut current_answer_offset = 0;
    let mut nx_domain_values = Vec::new();
    let mut response_time_values = Vec::new();

    for record in records {
        for answer in &record.answers {
            nx_domain_values.push(answer.nx_domain);
            response_time_values.push(answer.response_time_ms);
        }
        current_answer_offset += record.answers.len() as i32;
        answer_offsets.push(current_answer_offset);
    }

    let nx_domain_array = BooleanArray::from(nx_domain_values);
    let response_time_array = Int64Array::from(response_time_values);

    // Get the struct field definition from the schema
    let answers_field = schema.field_with_name("answers")?.clone();
    let answers_list_item_field = match answers_field.data_type() {
        DataType::List(field) => field.clone(),
        _ => return Err(ArrowError::FieldNotFound("answers list item".to_string())),
    };
    let answers_struct_type = match answers_list_item_field.data_type() {
        DataType::Struct(fields) => fields.clone(),
        _ => return Err(ArrowError::FieldNotFound("answers struct type".to_string())),
    };

    let struct_array = StructArray::new(
        answers_struct_type, // Use fields from schema
        vec![
            Arc::new(nx_domain_array) as ArrayRef,
            Arc::new(response_time_array) as ArrayRef,
        ],
        None, // Nullability bitmap for the structs themselves
    );

    let answer_offsets_buffer = OffsetBuffer::new(answer_offsets.into());

    let answers_array = ListArray::new(
        answers_list_item_field, // Use the field obtained from schema
        answer_offsets_buffer,
        Arc::new(struct_array) as ArrayRef,
        None, // Nullability bitmap for the list itself
    );

    // --- Create RecordBatch ---
    RecordBatch::try_new(
        schema.clone(), // Use the Arc<Schema>
        vec![
            Arc::new(doc_ids),
            Arc::new(timestamps),
            Arc::new(levels),
            Arc::new(messages),
            Arc::new(source_ips),
            Arc::new(source_hosts),
            Arc::new(source_regions),
            Arc::new(user_ids),
            Arc::new(user_session_ids),
            Arc::new(user_metrics_login_time),
            Arc::new(user_metrics_clicks),
            Arc::new(user_metrics_active),
            Arc::new(payload_sizes),
            Arc::new(tags_array),
            Arc::new(answers_array),
            Arc::new(processed),
        ],
    )
    .map_err(ArrowError::Arrow)
}

// --- Write RecordBatch to Parquet (Optimized - Unchanged) ---
fn write_record_batch_to_parquet(
    batch: &RecordBatch,
    file_path: &str,
    compression: Compression,
    row_group_size: Option<usize>,
) -> Result<()> {
    println!(
        "Writing {} rows to Parquet file {}...",
        batch.num_rows(),
        file_path
    );

    let file = File::create(file_path)?;
    // Increased buffer size
    let buf_writer = BufWriter::with_capacity(2 * 1024 * 1024, file);

    // Configure writer properties
    let props = WriterProperties::builder()
        .set_compression(compression)
        .set_statistics_enabled(EnabledStatistics::Page) // Keep page stats enabled
        // Enable dictionary encoding globally, ArrowWriter will apply it to Utf8/Binary columns
        .set_dictionary_enabled(true)
        // Example: Set specific encoding for a column if needed
        // .set_column_encoding(batch.schema().field_with_name("message")?.into(), Encoding::PLAIN)
        // Set row group size if specified
        .set_max_row_group_size(row_group_size.unwrap_or(1024 * 1024)) // Default or specified
        .build();

    // Create Arrow writer
    let mut writer = ArrowWriter::try_new(buf_writer, batch.schema(), Some(props))?;

    // Write the batch
    writer.write(batch)?;

    // Close the writer and flush buffers
    writer.close()?;

    Ok(())
}

// --- Write LogRecords to Parquet (Optimized - Unchanged) ---
fn write_records_to_parquet(
    records: Vec<LogRecord>,
    file_path: &str,
    compression: Compression,
    row_group_size: Option<usize>,
) -> Result<()> {
    if records.is_empty() {
        println!("No records to write.");
        // Optionally create an empty parquet file with schema
        // let schema = Arc::new(create_arrow_schema());
        // let batch = RecordBatch::new_empty(schema);
        // write_record_batch_to_parquet(&batch, file_path, compression, row_group_size)?;
        return Ok(());
    }

    let start_time = Instant::now();
    println!(
        "Starting conversion and write of {} records to Parquet file {}...",
        records.len(),
        file_path
    );

    // 1. Create RecordBatch (Handles dictionary encoding now)
    let batch_creation_start = Instant::now();
    let batch = create_record_batch_from_records(&records)?;
    println!(
        "Created RecordBatch in {:?}",
        batch_creation_start.elapsed()
    );

    // 2. Write to Parquet
    let write_time = Instant::now();
    write_record_batch_to_parquet(&batch, file_path, compression, row_group_size)?;

    let duration = start_time.elapsed();
    println!(
        "Successfully wrote {} records to {} in {:?} (Write took {:?})",
        records.len(),
        file_path,
        duration,
        write_time.elapsed()
    );

    Ok(())
}

// --- Helper for Field Names (Unchanged) ---
// Note: This might need adjustment if your schema uses nested field names like 'user.id'
// The current schema flattens them (e.g., 'user_id').
fn field_name_to_column(field_name: &str) -> String {
    field_name.replace('.', "_")
}

// --- Result Structs (Unchanged) ---
#[derive(Debug)]
struct FieldValueResult {
    value_map: HashMap<String, Vec<i64>>,
}

#[derive(Debug)]
struct NumericStatsResult {
    min: Option<i64>,
    max: Option<i64>,
    sum: i64,
    count: usize,
    avg: f64,
}

// --- Parquet Reader Setup (Optimized) ---

/// Holds the Parquet reader builder and metadata.
/// Avoids repeated file opening and metadata parsing.
struct ParquetReaderContext {
    file_path: String,
    builder: ParquetRecordBatchReaderBuilder<File>,
    schema: Arc<Schema>,
    metadata: Arc<ParquetMetaData>, // Keep metadata accessible
}

impl ParquetReaderContext {
    /// Creates a new context by opening the file and reading metadata.
    fn new(file_path: &str) -> Result<Self> {
        let file = File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let metadata = builder.metadata().clone();
        let schema = builder.schema().clone();

        Ok(Self {
            file_path: file_path.to_string(),
            builder,
            schema,
            metadata,
        })
    }

    /// Gets the Arrow schema.
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Creates a record batch reader with the specified column projection.
    /// Reopens the file for each reader instance.
    fn get_reader_with_projection(
        &self,
        column_indices: Vec<usize>,
    ) -> Result<ParquetRecordBatchReader> {
        // Need to reopen the file for each independent reader instance
        let file = File::open(&self.file_path)?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

        // Apply projection if indices are provided
        if !column_indices.is_empty() {
            let schema_descr = self.metadata.file_metadata().schema_descr();
            let projection = ProjectionMask::roots(schema_descr, column_indices);
            builder = builder.with_projection(projection);
        } else {
            // If no indices, read all columns (or consider reading none/minimal if appropriate)
            // This case should ideally be avoided if only schema is needed.
            println!("Warning: Creating reader with no projection (reading all columns).");
        }

        // Set batch size (optional, tune based on memory/performance)
        builder = builder.with_batch_size(8192); // Example batch size

        builder.build().map_err(ArrowError::Parquet)
    }

    /// Gets column indices for the given field names.
    fn get_column_indices(&self, required_fields: &[&str]) -> Result<Vec<usize>> {
        required_fields
            .iter()
            .map(|&field_name| {
                self.schema
                    .fields()
                    .iter()
                    .position(|f| f.name() == field_name)
                    .ok_or_else(|| ArrowError::FieldNotFound(field_name.to_string()))
            })
            .collect()
    }
}

// --- Query by Doc IDs using compute kernels (Optimized with streaming and projection) ---
fn get_field_values_by_doc_ids(
    reader_context: &ParquetReaderContext, // Use shared context
    field_name: &str,
    doc_ids: &[i64],
) -> Result<(FieldValueResult, QueryStats)> {
    let mut stats = QueryStats::new(
        "get_field_values_by_doc_ids",
        field_name,
        Some(doc_ids.len()),
    );
    let overall_start = Instant::now();

    // Setup phase (mostly done by context creation)
    let (doc_id_col_name, field_col_name) = ("doc_id", field_name_to_column(field_name));

    // Get column indices directly from context's schema
    let required_indices = time_section!(stats, setup, {
        println!(
            "Querying get_field_values_by_doc_ids for field '{}' with {} doc_ids",
            field_col_name,
            doc_ids.len()
        );
        reader_context.get_column_indices(&[doc_id_col_name, &field_col_name])?
    });

    // Create a reader with only the columns we need
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection(required_indices)?
    });

    // Indices within the *projected* batch (will always be 0 and 1 if only two columns projected)
    let doc_id_idx_proj = 0;
    let field_idx_proj = 1;

    // Create a set of doc_ids for faster lookup
    let doc_id_set: std::collections::HashSet<i64> = doc_ids.iter().cloned().collect();

    // Processing phase - use streaming approach
    let value_map = time_section!(stats, processing, {
        let mut map: HashMap<String, Vec<i64>> = HashMap::new();
        let mut total_rows_scanned = 0;
        let mut matched_rows = 0;

        // Process each batch
        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows_scanned += batch.num_rows();

            let doc_id_array = batch
                .column(doc_id_idx_proj)
                .as_primitive::<arrow::datatypes::Int64Type>();

            // Create filter mask in parallel for potentially large batches
            let filter_mask: Vec<bool> = (0..batch.num_rows())
                .into_par_iter() // Use Rayon for parallel check
                .map(|i| doc_id_set.contains(&doc_id_array.value(i)))
                .collect();

            // Count matches (can be done during mask creation or after)
            let current_matches = filter_mask.iter().filter(|&&x| x).count();
            if current_matches == 0 {
                continue; // Skip batch if no matches
            }
            matched_rows += current_matches;

            // Apply filter using Arrow compute kernel
            let filter_array = BooleanArray::from(filter_mask);
            let filtered_doc_ids = filter::filter(&doc_id_array, &filter_array)?
                .as_primitive::<arrow::datatypes::Int64Type>()
                .clone(); // Clone needed if used multiple times below

            let field_array = batch.column(field_idx_proj);

            // --- Process based on field type ---
            // Use macros or a helper function to reduce repetition
            macro_rules! process_array {
                ($arr_type:ty, $val_to_string:expr) => {{
                    // Store the filtered result in a variable
                    let filtered_array = filter::filter(field_array, &filter_array)?;

                    // Then downcast from the stored reference
                    let typed_array = filtered_array
                        .as_any()
                        .downcast_ref::<$arr_type>()
                        .ok_or_else(|| {
                            ArrowError::Conversion(format!(
                                "Failed to downcast filtered column to {}",
                                stringify!($arr_type)
                            ))
                        })?;

                    for i in 0..typed_array.len() {
                        if !typed_array.is_null(i) {
                            // Check for nulls before accessing value
                            let value = $val_to_string(typed_array, i);
                            let doc_id = filtered_doc_ids.value(i); // Safe access due to filter
                            map.entry(value).or_default().push(doc_id);
                        }
                    }
                }};
            }

            macro_rules! process_dict_array {
                ($key_type:ty) => {{
                    // First, make sure to store the filtered result in a variable to extend its lifetime
                    let filtered_array = filter::filter(field_array, &filter_array)?;

                    // Then downcast from the stored reference
                    let dict_array = filtered_array
                                        .as_any()
                                        .downcast_ref::<DictionaryArray<$key_type>>()
                                        .ok_or_else(|| ArrowError::Conversion(format!("Failed to downcast filtered column to DictionaryArray<{}>", stringify!($key_type))))?;
                    let keys = dict_array.keys();
                    let values = dict_array.values().as_string::<i32>(); // Assuming Utf8 values

                    for i in 0..keys.len() {
                         if !keys.is_null(i) { // Check for nulls before accessing key
                            let key = keys.value(i);
                            let value_str = values.value(key as usize).to_string(); // Get string from dictionary
                            let doc_id = filtered_doc_ids.value(i); // Safe access due to filter
                            map.entry(value_str).or_default().push(doc_id);
                         }
                    }
                }};
            }

            match field_array.data_type() {
                DataType::Utf8 => process_array!(StringArray, |arr: &StringArray, idx: usize| arr
                    .value(idx)
                    .to_string()),
                DataType::Boolean => {
                    process_array!(BooleanArray, |arr: &BooleanArray, idx: usize| arr
                        .value(idx)
                        .to_string())
                }
                DataType::Int64 => process_array!(Int64Array, |arr: &Int64Array, idx: usize| arr
                    .value(idx)
                    .to_string()),
                DataType::Dictionary(_, value_type) if **value_type == DataType::Utf8 => {
                    // Assuming Int32 keys based on schema, adjust if different
                    process_dict_array!(Int32Type);
                }
                // Add other numeric types (Int32, Float64, etc.) if needed
                _ => {
                    return Err(ArrowError::UnsupportedType(format!(
                        "Unsupported data type for field value aggregation: {:?}",
                        field_array.data_type()
                    )));
                }
            }
        }

        stats.set_result_rows(total_rows_scanned);
        println!(
            "Scanned {} rows, matched {} rows for doc_ids",
            total_rows_scanned, matched_rows
        );
        map
    });

    // Finalize stats
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();
    stats.print_summary();

    Ok((FieldValueResult { value_map }, stats))
}

// --- Get All Field Values (Optimized with streaming, projection, dictionary handling) ---
fn get_field_values(
    reader_context: &ParquetReaderContext, // Use shared context
    field_name: &str,
) -> Result<(FieldValueResult, QueryStats)> {
    let mut stats = QueryStats::new("get_field_values", field_name, None);
    let overall_start = Instant::now();

    // Setup phase
    let (doc_id_col_name, field_col_name) = ("doc_id", field_name_to_column(field_name));

    // Get column indices
    let required_indices = time_section!(stats, setup, {
        println!("Querying get_field_values for field '{}'", field_col_name);
        reader_context.get_column_indices(&[doc_id_col_name, &field_col_name])?
    });

    // Create reader with projection
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection(required_indices)?
    });

    // Indices within the projected batch
    let doc_id_idx_proj = 0;
    let field_idx_proj = 1;

    // Processing phase
    let value_map = time_section!(stats, processing, {
        // Estimate capacity based on field cardinality if known
        let estimated_capacity = match field_name {
            "level" | "source_region" | "processed" => 10, // Low cardinality
            _ => 1000,                                     // Default guess
        };
        let mut map: HashMap<String, Vec<i64>> = HashMap::with_capacity(estimated_capacity);
        let mut total_rows = 0;

        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows += batch.num_rows();

            let doc_id_array = batch
                .column(doc_id_idx_proj)
                .as_primitive::<arrow::datatypes::Int64Type>();
            let field_array = batch.column(field_idx_proj);

            // --- Process based on field type ---
            macro_rules! process_batch {
                ($arr_type:ty, $val_to_string:expr) => {{
                    let typed_array = field_array
                        .as_any()
                        .downcast_ref::<$arr_type>()
                        .ok_or_else(|| {
                            ArrowError::Conversion(format!(
                                "Failed to downcast column to {}",
                                stringify!($arr_type)
                            ))
                        })?;

                    for i in 0..batch.num_rows() {
                        if !typed_array.is_null(i) {
                            let doc_id = doc_id_array.value(i); // Safe access
                            let value = $val_to_string(typed_array, i);
                            map.entry(value).or_default().push(doc_id);
                        }
                    }
                }};
            }
            macro_rules! process_dict_batch {
                ($key_type:ty) => {{
                    let dict_array = field_array
                                        .as_any()
                                        .downcast_ref::<DictionaryArray<$key_type>>()
                                        .ok_or_else(|| ArrowError::Conversion(format!("Failed to downcast column to DictionaryArray<{}>", stringify!($key_type))))?;
                    let keys = dict_array.keys();
                    let values = dict_array.values().as_string::<i32>(); // Assuming Utf8 values

                    for i in 0..batch.num_rows() {
                         if !keys.is_null(i) && !doc_id_array.is_null(i) { // Check nulls
                            let key = keys.value(i);
                            let value_str = values.value(key as usize).to_string(); // Get string from dictionary
                            let doc_id = doc_id_array.value(i); // Safe access
                            map.entry(value_str).or_default().push(doc_id);
                         }
                    }
                }};
            }

            match field_array.data_type() {
                DataType::Utf8 => process_batch!(StringArray, |arr: &StringArray, idx: usize| arr
                    .value(idx)
                    .to_string()),
                DataType::Boolean => {
                    process_batch!(BooleanArray, |arr: &BooleanArray, idx: usize| arr
                        .value(idx)
                        .to_string())
                }
                DataType::Int64 => process_batch!(Int64Array, |arr: &Int64Array, idx: usize| arr
                    .value(idx)
                    .to_string()),
                DataType::Dictionary(_, value_type) if **value_type == DataType::Utf8 => {
                    // Assuming Int32 keys based on schema
                    process_dict_batch!(Int32Type);
                }
                _ => {
                    return Err(ArrowError::UnsupportedType(format!(
                        "Unsupported data type for field value aggregation: {:?}",
                        field_array.data_type()
                    )));
                }
            }
        }
        stats.set_result_rows(total_rows);
        map
    });

    // Finalize stats
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();
    stats.print_summary();

    Ok((FieldValueResult { value_map }, stats))
}

// --- Get Numeric Stats by Doc IDs (Optimized) ---
fn get_numeric_stats_by_doc_ids(
    reader_context: &ParquetReaderContext, // Use shared context
    field_name: &str,
    doc_ids: &[i64],
) -> Result<(NumericStatsResult, QueryStats)> {
    let mut stats = QueryStats::new(
        "get_numeric_stats_by_doc_ids",
        field_name,
        Some(doc_ids.len()),
    );
    let overall_start = Instant::now();

    // Setup phase
    let (doc_id_col_name, field_col_name) = ("doc_id", field_name_to_column(field_name));

    // Get column indices
    let required_indices = time_section!(stats, setup, {
        println!(
            "Querying get_numeric_stats_by_doc_ids for field '{}' with {} doc_ids",
            field_col_name,
            doc_ids.len()
        );
        reader_context.get_column_indices(&[doc_id_col_name, &field_col_name])?
    });

    // Create reader with projection
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection(required_indices)?
    });

    // Indices within the projected batch
    let doc_id_idx_proj = 0;
    let field_idx_proj = 1;

    // Create doc_id set
    let doc_id_set: std::collections::HashSet<i64> = doc_ids.iter().cloned().collect();

    // Processing phase
    let numeric_stats = time_section!(stats, processing, {
        let mut min_value: Option<i64> = None;
        let mut max_value: Option<i64> = None;
        let mut sum_value: i64 = 0;
        let mut count: usize = 0;
        let mut total_rows_scanned = 0;
        let mut matched_rows = 0;

        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows_scanned += batch.num_rows();

            let doc_id_array = batch
                .column(doc_id_idx_proj)
                .as_primitive::<arrow::datatypes::Int64Type>();
            let field_array = batch.column(field_idx_proj);

            // Ensure the field is Int64
            let int_field_array = field_array.as_primitive::<arrow::datatypes::Int64Type>();
            // This check happens once per query now due to schema check in context

            // Create filter mask (parallelized)
            let filter_mask: Vec<bool> = (0..batch.num_rows())
                .into_par_iter() // Use Rayon for parallel check
                .map(|i| {
                    // Use value method and check for null
                    doc_id_set.contains(&doc_id_array.value(i))
                })
                .collect();

            let current_matches = filter_mask.iter().filter(|&&x| x).count();
            if current_matches == 0 {
                continue;
            }
            matched_rows += current_matches;

            // Apply filter to the numeric field
            let filter_array = BooleanArray::from(filter_mask);
            let filtered_values = filter::filter(int_field_array, &filter_array)?
                .as_primitive::<arrow::datatypes::Int64Type>()
                .clone(); // Clone needed for iteration

            // Aggregate stats from the filtered values
            for i in 0..filtered_values.len() {
                if !filtered_values.is_null(i) {
                    // Check for nulls
                    let value = filtered_values.value(i); // Safe access

                    min_value = Some(min_value.map_or(value, |min| min.min(value)));
                    max_value = Some(max_value.map_or(value, |max| max.max(value)));
                    sum_value = sum_value.saturating_add(value); // Prevent overflow
                    count += 1;
                }
            }
        }

        stats.set_result_rows(total_rows_scanned);
        println!(
            "Scanned {} rows, matched {} rows for doc_ids",
            total_rows_scanned, matched_rows
        );

        let avg = if count > 0 {
            sum_value as f64 / count as f64
        } else {
            0.0
        };

        NumericStatsResult {
            min: min_value,
            max: max_value,
            sum: sum_value,
            count,
            avg,
        }
    });

    // Finalize stats
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();
    stats.print_summary();

    Ok((numeric_stats, stats))
}

// --- Get Numeric Stats for all records (Optimized) ---
fn get_numeric_stats(
    reader_context: &ParquetReaderContext, // Use shared context
    field_name: &str,
) -> Result<(NumericStatsResult, QueryStats)> {
    let mut stats = QueryStats::new("get_numeric_stats", field_name, None);
    let overall_start = Instant::now();

    // Setup phase
    let field_col_name = field_name_to_column(field_name);

    // Get column index
    let required_indices = time_section!(stats, setup, {
        println!("Querying get_numeric_stats for field '{}'", field_col_name);
        reader_context.get_column_indices(&[&field_col_name])? // Only need the target field
    });

    // Create reader with projection
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection(required_indices)?
    });

    // Index within the projected batch (always 0 if only one column)
    let field_idx_proj = 0;

    // Processing phase
    let numeric_stats = time_section!(stats, processing, {
        let mut min_value: Option<i64> = None;
        let mut max_value: Option<i64> = None;
        let mut sum_value: i64 = 0;
        let mut count: usize = 0;
        let mut total_rows = 0;

        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows += batch.num_rows();

            let field_array = batch.column(field_idx_proj);

            // Ensure the field is Int64
            let int_field_array = field_array.as_primitive::<arrow::datatypes::Int64Type>();

            // Aggregate stats directly from the batch
            for i in 0..batch.num_rows() {
                if !int_field_array.is_null(i) {
                    // Check for nulls
                    let value = int_field_array.value(i); // Safe access

                    min_value = Some(min_value.map_or(value, |min| min.min(value)));
                    max_value = Some(max_value.map_or(value, |max| max.max(value)));
                    sum_value = sum_value.saturating_add(value); // Prevent overflow
                    count += 1;
                }
            }
        }

        stats.set_result_rows(total_rows);

        let avg = if count > 0 {
            sum_value as f64 / count as f64
        } else {
            0.0
        };

        NumericStatsResult {
            min: min_value,
            max: max_value,
            sum: sum_value,
            count,
            avg,
        }
    });

    // Finalize stats
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();
    stats.print_summary();

    Ok((numeric_stats, stats))
}

// --- Get Tag Buckets (Optimized with DictionaryArray) ---
fn get_tag_buckets(
    reader_context: &ParquetReaderContext, // Use shared context
) -> Result<(HashMap<String, Vec<i64>>, QueryStats)> {
    let mut stats = QueryStats::new("get_tag_buckets", "tags", None);
    let overall_start = Instant::now();

    // Setup phase
    let (doc_id_col_name, tags_col_name) = ("doc_id", "tags");

    // Get column indices
    let required_indices = time_section!(stats, setup, {
        println!("Querying get_tag_buckets");
        reader_context.get_column_indices(&[doc_id_col_name, tags_col_name])?
    });

    // Create reader with projection
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection(required_indices)?
    });

    // Indices within the projected batch
    let doc_id_idx_proj = 0;
    let tags_idx_proj = 1;

    // Processing phase
    let tag_buckets = time_section!(stats, processing, {
        let estimated_capacity = 50; // Keep capacity hint
        let mut map: HashMap<String, Vec<i64>> = HashMap::with_capacity(estimated_capacity);
        let mut total_rows = 0;

        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows += batch.num_rows();

            let doc_id_array = batch
                .column(doc_id_idx_proj)
                .as_primitive::<arrow::datatypes::Int64Type>();
            let tags_list_array = batch.column(tags_idx_proj).as_list::<i32>(); // Use i32 offsets

            // Process each row in the batch
            for row_idx in 0..batch.num_rows() {
                if tags_list_array.is_null(row_idx) || doc_id_array.is_null(row_idx) {
                    continue; // Skip row if tags list or doc_id is null
                }

                let doc_id = doc_id_array.value(row_idx); // Safe access
                let tags_in_row = tags_list_array.value(row_idx); // This is an ArrayRef

                // *** Optimization: Downcast to DictionaryArray ***
                // Assuming Int32 keys based on schema generation
                if let Some(tag_dict_array) = tags_in_row
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                {
                    let keys = tag_dict_array.keys();
                    // Get the dictionary values (StringArray) ONCE per batch if possible,
                    // or ensure efficient access.
                    let dict_values = tag_dict_array.values().as_string::<i32>();

                    for key_idx in 0..keys.len() {
                        if !keys.is_null(key_idx) {
                            let key = keys.value(key_idx);
                            // Retrieve the string value from the dictionary using the key
                            let tag_str = dict_values.value(key as usize).to_string();
                            // Use the string value as the map key
                            map.entry(tag_str).or_default().push(doc_id);
                        }
                    }
                } else {
                    // Fallback or error if not dictionary encoded as expected
                    eprintln!(
                        "Warning: Tags column is not dictionary encoded as expected in row {}",
                        row_idx
                    );
                }
            }
        }

        stats.set_result_rows(total_rows);
        map
    });

    // Finalize stats
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();
    stats.print_summary();

    Ok((tag_buckets, stats))
}

// --- Main function (Updated to use ParquetReaderContext) ---
fn main() -> Result<()> {
    // --- Data Generation and Writing (Optional) ---
    let record_count = 10_000_000; // 10 million records
    let file_path = "logs_optimized.parquet";
    let generate_and_write = false; // Set to false to skip generation and use existing file

    if generate_and_write {
        let records = generate_log_records(record_count);
        write_records_to_parquet(
            records,
            file_path,
            Compression::ZSTD(Default::default()), // Try ZSTD for potentially better compression
            Some(1024 * 1024),                     // 1M rows per row group (adjust based on memory)
        )?;
    } else {
        println!(
            "Skipping data generation. Using existing file: {}",
            file_path
        );
        // Ensure the file exists if skipping generation
        if !std::path::Path::new(file_path).exists() {
            return Err(ArrowError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Parquet file not found: {}", file_path),
            )));
        }
    }

    std::thread::sleep(std::time::Duration::from_secs(10));

    // --- Querying ---
    println!("\n--- Starting Queries ---");

    // Create the reader context once
    let reader_context = ParquetReaderContext::new(file_path)?;

    // Test queries using the context
    let doc_ids: Vec<i64> = (0..100).collect(); // Sample some doc_ids across the range

    // Test get_field_values_by_doc_ids (using dictionary 'level')
    let (level_result, _level_stats) =
        get_field_values_by_doc_ids(&reader_context, "level", &doc_ids)?;
    println!(
        "Level values for {} doc_ids: {} unique levels found.",
        doc_ids.len(),
        level_result.value_map.len()
    );
    // println!("Level values map: {:?}", level_result.value_map); // Optional: print map

    // Test get_field_values (using dictionary 'source_region')
    let (all_regions, _all_regions_stats) = get_field_values(&reader_context, "source_region")?;
    println!(
        "All source_regions: {} unique regions found.",
        all_regions.value_map.len()
    );

    std::thread::sleep(std::time::Duration::from_secs(10));

    // Test get_numeric_stats_by_doc_ids
    let (payload_stats, _payload_stats_query) =
        get_numeric_stats_by_doc_ids(&reader_context, "payload_size", &doc_ids)?;
    println!(
        "Payload stats for {} doc_ids: {:?}",
        doc_ids.len(),
        payload_stats
    );

    std::thread::sleep(std::time::Duration::from_secs(10));

    // Test get_numeric_stats
    let (all_payload_stats, _all_payload_stats_query) =
        get_numeric_stats(&reader_context, "payload_size")?;
    println!("All payload stats: {:?}", all_payload_stats);

    std::thread::sleep(std::time::Duration::from_secs(10));

    // Test get_tag_buckets (Optimized with DictionaryArray)
    let (tag_buckets, _tag_buckets_stats) = get_tag_buckets(&reader_context)?;
    println!("Tag buckets: {} unique tags found.", tag_buckets.len());

    println!("\n--- Queries Finished ---");

    Ok(())
}
