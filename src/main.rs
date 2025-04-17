// src/main.rs
use arrow::array::RecordBatchReader;
use arrow::array::{
    Array,
    ArrayRef,
    AsArray, // Trait needed for .as_primitive(), .as_string(), .as_list(), etc.
    BooleanArray,
    DictionaryArray,
    Int64Array,
    ListArray,
    // RecordBatchReader, // Removed unused import
    StringArray,
    StructArray,
    TimestampMicrosecondArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::kernels::filter;
use arrow::datatypes::{DataType, Field, Fields, Int32Type, Schema, TimeUnit}; // Assuming Int32 keys for dictionary
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
// Removed memmap2 as it's no longer used directly here
use parquet::arrow::{
    arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    arrow_writer::ArrowWriter,
    ProjectionMask, // Keep this import
};
use parquet::basic::{Compression, Encoding};
use parquet::file::statistics::{Statistics, TypedStatistics}; // Import Int64Statistics
use parquet::file::{
    metadata::ParquetMetaData,
    properties::{EnabledStatistics, WriterProperties},
    // reader::FileReader, // Removed unused import
    // serialized_reader::SerializedFileReader, // Removed unused import
};
use parquet::schema::types::SchemaDescPtr; // Use SchemaDescPtr
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
// Removed sleep as it wasn't used in the provided functions, keep if needed in main
use std::time::{Duration as StdDuration, Instant}; // Keep StdDuration if needed elsewhere
use thiserror::Error;
use uuid::Uuid;

// Import the allocator and stats module
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Ensure query_stats module exists and is correctly defined
// You need to create src/query_stats.rs with the content provided previously.
#[path = "query_stats.rs"] // Explicitly specify path if needed
mod query_stats;
use query_stats::*;

// --- Error Handling (Added StatsError) ---
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

    #[error("Statistics error: {0}")]
    StatsError(String), // Added for stats-related issues
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

// --- Arrow Schema Creation (Unchanged) ---
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

// --- Create RecordBatch from LogRecords (Handles Dictionary Encoding - Unchanged) ---
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
    // Increased buffer size for better write performance
    let buf_writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

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

// --- Parquet Reader Setup (FIXED: Removed mmap, using standard File reader) ---

/// Holds the Parquet reader context (metadata, schema, caches).
/// Avoids repeated file opening and metadata parsing for setup, but reopens file for actual reading.
struct ParquetReaderContext {
    file_path: String,
    schema: Arc<Schema>,
    metadata: Arc<ParquetMetaData>, // Keep metadata accessible
    schema_descr: SchemaDescPtr,    // Store the schema descriptor pointer
    // Cache for document ID sets to avoid repeated allocations
    doc_id_set_cache: RefCell<Option<(Vec<i64>, HashSet<i64>)>>, // Consider a more robust cache if needed
    // Cache for column indices to avoid repeated lookups
    column_indices_cache: RefCell<HashMap<String, usize>>,
    // Cache for field name to column name mapping
    field_column_cache: RefCell<HashMap<String, String>>,
}

impl ParquetReaderContext {
    /// Creates a new context by opening the file once to read metadata and schema.
    fn new(file_path: &str) -> Result<Self> {
        println!("Creating ParquetReaderContext for: {}", file_path);
        let file = File::open(file_path)?;

        // Create a builder just to get metadata and schema, then discard it
        // The actual reading will happen later by reopening the file
        let temp_builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let metadata = temp_builder.metadata().clone();
        let schema = temp_builder.schema().clone();
        let schema_descr = metadata.file_metadata().schema_descr_ptr(); // Get schema descriptor ptr
        println!("Successfully read metadata and schema.");

        Ok(Self {
            file_path: file_path.to_string(),
            schema,
            metadata,
            schema_descr,                         // Store it
            doc_id_set_cache: RefCell::new(None), // Initialize caches
            column_indices_cache: RefCell::new(HashMap::new()),
            field_column_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Gets the Arrow schema.
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Get or create a HashSet for doc_ids. (Simplified cache logic)
    fn get_doc_id_set<'a>(&'a self, doc_ids: &[i64]) -> HashSet<i64> {
        // Basic caching: if the input slice matches the cached vec, return the cached set.
        // Note: This is a simple cache. For more complex scenarios, consider LRU or other strategies.
        let mut cache = self.doc_id_set_cache.borrow_mut();
        if let Some((cached_vec, cached_set)) = cache.as_ref() {
            if cached_vec == doc_ids {
                // println!("Reusing cached doc_id set."); // Reduce verbosity
                return cached_set.clone(); // Clone the set for use
            }
        }

        // println!("Creating new doc_id set."); // Reduce verbosity
        let set: HashSet<i64> = doc_ids.iter().cloned().collect();
        *cache = Some((doc_ids.to_vec(), set.clone())); // Store a clone of the vec and set
        set
    }

    /// Find row groups that may contain the given document IDs based on min/max statistics (FIXED: Statistics access)
    fn find_row_groups_for_doc_ids(&self, doc_ids: &[i64]) -> Result<Vec<usize>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Find the column index for doc_id
        let doc_id_col_name = "doc_id";
        let doc_id_col_idx = self.get_column_index(doc_id_col_name)?;

        // Find min and max of requested doc_ids
        let min_doc_id = *doc_ids.iter().min().ok_or_else(|| {
            ArrowError::StatsError("Cannot find min in empty doc_ids".to_string())
        })?;
        let max_doc_id = *doc_ids.iter().max().ok_or_else(|| {
            ArrowError::StatsError("Cannot find max in empty doc_ids".to_string())
        })?;

        let mut relevant_row_groups = Vec::new();
        let row_groups = self.metadata.row_groups();

        // println!("Checking {} row groups for doc_id range [{}, {}]", row_groups.len(), min_doc_id, max_doc_id); // Reduce verbosity

        // For each row group, check if its doc_id range overlaps with our requested range
        for (rg_idx, row_group) in row_groups.iter().enumerate() {
            let column_chunks = row_group.columns();

            // Find the doc_id column chunk
            if let Some(column_chunk) = column_chunks.get(doc_id_col_idx) {
                // Check if statistics are available and are of the expected type (Int64)
                if let Some(stats) = column_chunk.statistics() {
                    // *** FIXED: Match on the Statistics enum variant ***
                    match stats {
                        Statistics::Int64(int_stats) => {
                            // Safely extract min and max values from the specific statistics struct
                            if let (Some(rg_min), Some(rg_max)) =
                                (int_stats.min_opt(), int_stats.max_opt())
                            {
                                // If there's any overlap between the ranges, include this row group
                                // Overlap condition: !(request_max < group_min || request_min > group_max)
                                if !(max_doc_id < *rg_min || min_doc_id > *rg_max) {
                                    // println!("Row group {} [{}, {}] overlaps.", rg_idx, rg_min, rg_max);
                                    relevant_row_groups.push(rg_idx);
                                } else {
                                    // println!("Row group {} [{}, {}] does NOT overlap.", rg_idx, rg_min, rg_max);
                                }
                            } else {
                                // Statistics exist but min/max are None, must include
                                // println!("Row group {} has Int64 stats but no min/max, including.", rg_idx);
                                relevant_row_groups.push(rg_idx);
                            }
                        }
                        _ => {
                            // Statistics exist but are not Int64 (unexpected for doc_id), must include
                            // println!("Row group {} has non-Int64 stats for doc_id column, including.", rg_idx);
                            relevant_row_groups.push(rg_idx);
                        }
                    }
                } else {
                    // If statistics are not available, we have to include this row group
                    // println!("Row group {} has no stats, including.", rg_idx);
                    relevant_row_groups.push(rg_idx);
                }
            } else {
                // If column chunk is not found (shouldn't happen if index is correct), include to be safe
                println!(
                    "Warning: Column chunk for doc_id not found in row group {}, including.",
                    rg_idx
                );
                relevant_row_groups.push(rg_idx);
            }
        }
        // println!("Found {} relevant row groups.", relevant_row_groups.len()); // Reduce verbosity
        Ok(relevant_row_groups)
    }

    /// Get column index with caching for better performance (Unchanged)
    fn get_column_index(&self, column_name: &str) -> Result<usize> {
        let mut cache = self.column_indices_cache.borrow_mut();

        if let Some(&idx) = cache.get(column_name) {
            return Ok(idx);
        }

        let idx = self
            .schema
            .fields()
            .iter()
            .position(|f| f.name() == column_name)
            .ok_or_else(|| ArrowError::FieldNotFound(column_name.to_string()))?;

        cache.insert(column_name.to_string(), idx);
        Ok(idx)
    }

    /// Get field name to column mapping with caching (Unchanged)
    fn get_column_name(&self, field_name: &str) -> String {
        let mut cache = self.field_column_cache.borrow_mut();

        if let Some(col_name) = cache.get(field_name) {
            return col_name.clone();
        }

        let col_name = field_name_to_column(field_name);
        cache.insert(field_name.to_string(), col_name.clone());
        col_name
    }

    /// Creates a record batch reader with the specified column projection and row group selection.
    /// (FIXED: Removed mmap, uses standard File reader by reopening)
    fn get_reader_with_projection_and_row_groups(
        &self,
        column_indices: Vec<usize>,
        row_group_indices: Vec<usize>,
    ) -> Result<ParquetRecordBatchReader> {
        // Need to reopen the file for each independent reader instance
        // println!("Opening file '{}' to create reader...", self.file_path); // Reduce verbosity
        let file = File::open(&self.file_path)?;

        // Use the standard builder with the opened file
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

        // Apply projection if indices are provided
        if !column_indices.is_empty() {
            // Use the stored schema descriptor pointer
            let projection = ProjectionMask::roots(&self.schema_descr, column_indices.clone()); // Clone indices if needed later
            builder = builder.with_projection(projection);
            // println!("Applying projection to columns: {:?}", column_indices); // Reduce verbosity
        } else {
            // If no indices, read all columns (or consider reading none/minimal if appropriate)
            println!("Warning: Creating reader with no projection (reading all columns).");
        }

        // Apply row group selection if indices are provided
        if !row_group_indices.is_empty() {
            builder = builder.with_row_groups(row_group_indices.clone()); // Clone indices if needed later
                                                                          // println!("Applying row group filter: {:?}", row_group_indices); // Reduce verbosity
        } else {
            // println!("Reading all row groups."); // Reduce verbosity
        }

        // Set a smaller batch size to reduce peak memory usage
        // Smaller batches help with memory usage but too small can hurt performance
        // Tune this value based on your specific workload
        builder = builder.with_batch_size(8192); // Increased batch size slightly
                                                 // println!("Setting batch size to 8192."); // Reduce verbosity

        builder.build().map_err(ArrowError::Parquet)
    }

    /// Creates a record batch reader with the specified column projection.
    /// Reopens the file for each reader instance. (Unchanged logic, calls fixed function)
    fn get_reader_with_projection(
        &self,
        column_indices: Vec<usize>,
    ) -> Result<ParquetRecordBatchReader> {
        // Get all row groups (no filtering)
        let all_row_groups: Vec<usize> = (0..self.metadata.num_row_groups()).collect(); // Use num_row_groups()
        self.get_reader_with_projection_and_row_groups(column_indices, all_row_groups)
    }

    /// Gets column indices for the given field names. (Unchanged logic, calls fixed function)
    fn get_column_indices(&self, required_fields: &[&str]) -> Result<Vec<usize>> {
        required_fields
            .iter()
            .map(|&field_name| {
                let col_name = self.get_column_name(field_name); // Ensure we use the mapped column name
                self.get_column_index(&col_name)
            })
            .collect()
    }
}

// --- Query by Doc IDs using compute kernels (Optimized with RowGroup pruning and memory optimizations - Unchanged logic, uses fixed context) ---
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
    let (doc_id_col_name, field_col_name) = ("doc_id", reader_context.get_column_name(field_name));

    // Get column indices directly from context's schema
    let required_indices = time_section!(stats, setup, {
        // println!( // Reduce verbosity
        //     "Querying get_field_values_by_doc_ids for field '{}' ({}) with {} doc_ids",
        //     field_name, // Original field name for logging
        //     field_col_name, // Mapped column name for lookup
        //     doc_ids.len()
        // );
        reader_context.get_column_indices(&[doc_id_col_name, &field_col_name])?
    });

    // Find row groups that may contain our doc_ids
    let row_group_indices = time_section!(stats, filter_creation, {
        reader_context.find_row_groups_for_doc_ids(doc_ids)?
    });
    let num_total_row_groups = reader_context.metadata.num_row_groups();
    println!(
        "Selected {}/{} row groups for field '{}' and {} doc_ids.",
        row_group_indices.len(),
        num_total_row_groups,
        field_name,
        doc_ids.len()
    );

    // Create a reader with only the columns we need and only the relevant row groups
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection_and_row_groups(
            required_indices.clone(),
            row_group_indices,
        )? // Clone indices if needed
    });

    // Indices within the *projected* batch
    // Find the indices based on the *actual* projected schema, not assuming 0 and 1
    let projected_schema = batch_reader.schema();
    let doc_id_idx_proj = projected_schema.index_of(doc_id_col_name)?;
    let field_idx_proj = projected_schema.index_of(&field_col_name)?;

    // Create a set of doc_ids for faster lookup - use the cached version
    let doc_id_set = reader_context.get_doc_id_set(doc_ids);

    // Processing phase - use streaming approach
    let value_map = time_section!(stats, processing, {
        // Use capacity hints for collections based on field type
        let estimated_capacity = match field_name {
            "level" => 5,                // We know there are only 5 log levels
            "source_region" => 5,        // We know there are only 5 regions
            "processed" => 2,            // Boolean field has only 2 possible values
            _ => doc_ids.len().min(100), // Default reasonable capacity
        };

        let mut map: HashMap<String, Vec<i64>> = HashMap::with_capacity(estimated_capacity);
        let mut total_rows_scanned = 0;
        let mut matched_rows = 0;

        // Process each batch
        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows_scanned += batch.num_rows();

            if batch.num_rows() == 0 {
                continue;
            } // Skip empty batches

            let doc_id_array = batch
                .column(doc_id_idx_proj)
                .as_primitive::<arrow::datatypes::Int64Type>();

            // Create filter mask in parallel for potentially large batches
            let filter_mask: Vec<bool> = (0..batch.num_rows())
                .into_par_iter() // Use Rayon for parallel check
                .map(|i| !doc_id_array.is_null(i) && doc_id_set.contains(&doc_id_array.value(i))) // Check for nulls
                .collect();

            // Count matches (can be done during mask creation or after)
            let current_matches = filter_mask.iter().filter(|&&x| x).count();
            if current_matches == 0 {
                continue; // Skip batch if no matches
            }
            matched_rows += current_matches;

            // Apply filter using Arrow compute kernel
            let filter_array = BooleanArray::from(filter_mask); // No need to clone filter_mask
            let filtered_doc_id_array = filter::filter(doc_id_array, &filter_array)?; // Use original doc_id_array
            let filtered_doc_ids =
                filtered_doc_id_array.as_primitive::<arrow::datatypes::Int64Type>();

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
                                "Failed to downcast filtered column '{}' to {}",
                                field_col_name,
                                stringify!($arr_type)
                            ))
                        })?;

                    for i in 0..typed_array.len() {
                        // Iterate up to length of filtered array
                        if !typed_array.is_null(i) && !filtered_doc_ids.is_null(i) {
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
                                        .ok_or_else(|| ArrowError::Conversion(format!("Failed to downcast filtered column '{}' to DictionaryArray<{}>", field_col_name, stringify!($key_type))))?;
                    let keys = dict_array.keys();
                    // Ensure values are StringArray before accessing
                    let values = dict_array.values()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| ArrowError::Conversion(format!("Dictionary values for '{}' are not Utf8", field_col_name)))?;


                    for i in 0..keys.len() { // Iterate up to length of filtered keys
                         if !keys.is_null(i) && !filtered_doc_ids.is_null(i) { // Check for nulls before accessing key and doc_id
                            let key = keys.value(i);
                            // Avoid unnecessary string allocation by using the string value directly
                            if !values.is_null(key as usize) { // Check null value in dictionary itself
                                let value_str = values.value(key as usize);
                                let doc_id = filtered_doc_ids.value(i); // Safe access due to filter
                                map.entry(value_str.to_string()).or_default().push(doc_id);
                            }
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
                DataType::Dictionary(key_type, value_type)
                    if **key_type == DataType::Int32 && **value_type == DataType::Utf8 =>
                {
                    // Assuming Int32 keys based on schema, adjust if different
                    process_dict_array!(Int32Type);
                }
                // Add other numeric types (Int32, Float64, etc.) if needed
                dt => {
                    return Err(ArrowError::UnsupportedType(format!(
                        "Unsupported data type for field value aggregation on '{}': {:?}",
                        field_col_name, dt
                    )));
                }
            }
        }

        stats.set_result_rows(total_rows_scanned);
        // println!( // Reduce verbosity
        //     "Scanned {} rows, matched {} rows for doc_ids",
        //     total_rows_scanned, matched_rows
        // );
        map
    });

    // Finalize stats
    stats.timing.total = overall_start.elapsed();
    stats.update_memory();
    stats.print_summary();

    Ok((FieldValueResult { value_map }, stats))
}

// --- Get All Field Values (Optimized with streaming, projection, dictionary handling, and memory optimizations - Unchanged logic, uses fixed context) ---
fn get_field_values(
    reader_context: &ParquetReaderContext, // Use shared context
    field_name: &str,
) -> Result<(FieldValueResult, QueryStats)> {
    let mut stats = QueryStats::new("get_field_values", field_name, None);
    let overall_start = Instant::now();

    // Setup phase
    let (doc_id_col_name, field_col_name) = ("doc_id", reader_context.get_column_name(field_name));

    // Get column indices
    let required_indices = time_section!(stats, setup, {
        // println!("Querying get_field_values for field '{}' ({})", field_name, field_col_name); // Reduce verbosity
        reader_context.get_column_indices(&[doc_id_col_name, &field_col_name])?
    });

    // Create reader with projection
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection(required_indices.clone())? // Clone indices if needed
    });

    // Indices within the projected batch
    let projected_schema = batch_reader.schema();
    let doc_id_idx_proj = projected_schema.index_of(doc_id_col_name)?;
    let field_idx_proj = projected_schema.index_of(&field_col_name)?;

    // Processing phase
    let value_map = time_section!(stats, processing, {
        // Estimate capacity based on field cardinality if known
        let estimated_capacity = match field_name {
            "level" => 5,         // We know there are only 5 log levels
            "source_region" => 5, // We know there are only 5 regions
            "processed" => 2,     // Boolean field has only 2 possible values
            _ => 1000,            // Default guess for high cardinality fields
        };
        let mut map: HashMap<String, Vec<i64>> = HashMap::with_capacity(estimated_capacity);
        let mut total_rows = 0;

        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows += batch.num_rows();
            if batch.num_rows() == 0 {
                continue;
            }

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
                                "Failed to downcast column '{}' to {}",
                                field_col_name,
                                stringify!($arr_type)
                            ))
                        })?;

                    for i in 0..batch.num_rows() {
                        if !typed_array.is_null(i) && !doc_id_array.is_null(i) {
                            // Check nulls
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
                        .ok_or_else(|| {
                            ArrowError::Conversion(format!(
                                "Failed to downcast column '{}' to DictionaryArray<{}>",
                                field_col_name,
                                stringify!($key_type)
                            ))
                        })?;
                    let keys = dict_array.keys();
                    let values = dict_array
                        .values()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            ArrowError::Conversion(format!(
                                "Dictionary values for '{}' are not Utf8",
                                field_col_name
                            ))
                        })?;

                    for i in 0..batch.num_rows() {
                        if !keys.is_null(i) && !doc_id_array.is_null(i) {
                            // Check nulls
                            let key = keys.value(i);
                            if !values.is_null(key as usize) {
                                // Check null value in dictionary itself
                                // Avoid unnecessary string allocation
                                let value_str = values.value(key as usize);
                                let doc_id = doc_id_array.value(i); // Safe access
                                map.entry(value_str.to_string()).or_default().push(doc_id);
                            }
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
                DataType::Dictionary(key_type, value_type)
                    if **key_type == DataType::Int32 && **value_type == DataType::Utf8 =>
                {
                    // Assuming Int32 keys based on schema
                    process_dict_batch!(Int32Type);
                }
                dt => {
                    return Err(ArrowError::UnsupportedType(format!(
                        "Unsupported data type for field value aggregation on '{}': {:?}",
                        field_col_name, dt
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

// --- Get Numeric Stats by Doc IDs (Optimized with RowGroup pruning and streaming aggregation - Unchanged logic, uses fixed context) ---
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
    let (doc_id_col_name, field_col_name) = ("doc_id", reader_context.get_column_name(field_name));

    // Get column indices
    let required_indices = time_section!(stats, setup, {
        // println!( // Reduce verbosity
        //     "Querying get_numeric_stats_by_doc_ids for field '{}' ({}) with {} doc_ids",
        //     field_name, field_col_name, doc_ids.len()
        // );
        reader_context.get_column_indices(&[doc_id_col_name, &field_col_name])?
    });

    // Find row groups that may contain our doc_ids
    let row_group_indices = time_section!(stats, filter_creation, {
        reader_context.find_row_groups_for_doc_ids(doc_ids)?
    });
    let num_total_row_groups = reader_context.metadata.num_row_groups();
    println!(
        "Selected {}/{} row groups for numeric stats on '{}' and {} doc_ids.",
        row_group_indices.len(),
        num_total_row_groups,
        field_name,
        doc_ids.len()
    );

    // Create reader with projection and row group selection
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection_and_row_groups(
            required_indices.clone(),
            row_group_indices,
        )? // Clone indices
    });

    // Indices within the projected batch
    let projected_schema = batch_reader.schema();
    let doc_id_idx_proj = projected_schema.index_of(doc_id_col_name)?;
    let field_idx_proj = projected_schema.index_of(&field_col_name)?;

    // Create doc_id set - use the cached version
    let doc_id_set = reader_context.get_doc_id_set(doc_ids);

    // Processing phase - use streaming aggregation to minimize memory usage
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
            if batch.num_rows() == 0 {
                continue;
            }

            let doc_id_array = batch
                .column(doc_id_idx_proj)
                .as_primitive::<arrow::datatypes::Int64Type>();
            let field_array = batch.column(field_idx_proj);

            // Ensure the field is Int64
            let int_field_array = field_array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    ArrowError::Conversion(format!("Field '{}' is not Int64", field_col_name))
                })?;

            // Create filter mask (parallelized)
            let filter_mask: Vec<bool> = (0..batch.num_rows())
                .into_par_iter() // Use Rayon for parallel check
                .map(|i| !doc_id_array.is_null(i) && doc_id_set.contains(&doc_id_array.value(i))) // Check nulls
                .collect();

            let current_matches = filter_mask.iter().filter(|&&x| x).count();
            if current_matches == 0 {
                continue;
            }
            matched_rows += current_matches;

            // Apply filter to the numeric field
            let filter_array = BooleanArray::from(filter_mask); // No need to clone
            let filtered_array = filter::filter(int_field_array, &filter_array)?;
            let filtered_values = filtered_array.as_primitive::<arrow::datatypes::Int64Type>();

            // Aggregate stats from the filtered values directly in streaming fashion
            for i in 0..filtered_values.len() {
                // Iterate over filtered length
                if !filtered_values.is_null(i) {
                    // Check for nulls in filtered array
                    let value = filtered_values.value(i); // Safe access

                    min_value = Some(min_value.map_or(value, |min| min.min(value)));
                    max_value = Some(max_value.map_or(value, |max| max.max(value)));
                    sum_value = sum_value.saturating_add(value); // Prevent overflow
                    count += 1;
                }
            }
        }

        stats.set_result_rows(total_rows_scanned);
        // println!( // Reduce verbosity
        //     "Scanned {} rows, matched {} rows for doc_ids",
        //     total_rows_scanned, matched_rows
        // );

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

// --- Get Numeric Stats for all records (Optimized with streaming aggregation - Unchanged logic, uses fixed context) ---
fn get_numeric_stats(
    reader_context: &ParquetReaderContext, // Use shared context
    field_name: &str,
) -> Result<(NumericStatsResult, QueryStats)> {
    let mut stats = QueryStats::new("get_numeric_stats", field_name, None);
    let overall_start = Instant::now();

    // Setup phase
    let field_col_name = reader_context.get_column_name(field_name);

    // Get column index
    let required_indices = time_section!(stats, setup, {
        // println!("Querying get_numeric_stats for field '{}' ({})", field_name, field_col_name); // Reduce verbosity
        reader_context.get_column_indices(&[&field_col_name])? // Only need the target field
    });

    // Create reader with projection
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection(required_indices.clone())? // Clone indices
    });

    // Index within the projected batch (always 0 if only one column)
    let projected_schema = batch_reader.schema();
    let field_idx_proj = projected_schema.index_of(&field_col_name)?;

    // Processing phase - use streaming aggregation to minimize memory usage
    let numeric_stats = time_section!(stats, processing, {
        let mut min_value: Option<i64> = None;
        let mut max_value: Option<i64> = None;
        let mut sum_value: i64 = 0;
        let mut count: usize = 0;
        let mut total_rows = 0;

        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows += batch.num_rows();
            if batch.num_rows() == 0 {
                continue;
            }

            let field_array = batch.column(field_idx_proj);

            // Ensure the field is Int64
            let int_field_array = field_array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    ArrowError::Conversion(format!("Field '{}' is not Int64", field_col_name))
                })?;

            // Aggregate stats directly from the batch in streaming fashion
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

// --- Get Tag Buckets (Optimized with DictionaryArray and memory optimizations - Unchanged logic, uses fixed context) ---
fn get_tag_buckets(
    reader_context: &ParquetReaderContext, // Use shared context
) -> Result<(HashMap<String, Vec<i64>>, QueryStats)> {
    let mut stats = QueryStats::new("get_tag_buckets", "tags", None);
    let overall_start = Instant::now();

    // Setup phase
    let (doc_id_col_name, tags_col_name) = ("doc_id", "tags"); // Use original field name for mapping

    // Get column indices using mapped names
    let doc_id_mapped = reader_context.get_column_name(doc_id_col_name);
    let tags_mapped = reader_context.get_column_name(tags_col_name);

    let required_indices = time_section!(stats, setup, {
        // println!("Querying get_tag_buckets for field '{}' ({})", tags_col_name, tags_mapped); // Reduce verbosity
        reader_context.get_column_indices(&[&doc_id_mapped, &tags_mapped])?
    });

    // Create reader with projection
    let mut batch_reader = time_section!(stats, filter_creation, {
        reader_context.get_reader_with_projection(required_indices.clone())? // Clone indices
    });

    // Indices within the projected batch
    let projected_schema = batch_reader.schema();
    let doc_id_idx_proj = projected_schema.index_of(&doc_id_mapped)?;
    let tags_idx_proj = projected_schema.index_of(&tags_mapped)?;

    // Processing phase
    let tag_buckets = time_section!(stats, processing, {
        // Estimate capacity for better memory usage
        let estimated_capacity = 50; // Based on tag generation logic (1-50)
        let mut map: HashMap<String, Vec<i64>> = HashMap::with_capacity(estimated_capacity);
        let mut total_rows = 0;

        // Process in chunks to limit memory usage
        // const CHUNK_SIZE: usize = 10000; // Removed chunking for memory stats, re-add if needed
        // let mut chunk_rows = 0;

        while let Some(batch_result) = batch_reader.next() {
            let batch = batch_result?;
            total_rows += batch.num_rows();
            if batch.num_rows() == 0 {
                continue;
            }

            let doc_id_array = batch
                .column(doc_id_idx_proj)
                .as_primitive::<arrow::datatypes::Int64Type>();

            // Downcast the tags column to ListArray
            let tags_list_array = batch
                .column(tags_idx_proj)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| {
                    ArrowError::Conversion(format!("Column '{}' is not a ListArray", tags_mapped))
                })?;

            // Process each row in the batch
            for row_idx in 0..batch.num_rows() {
                if tags_list_array.is_null(row_idx) || doc_id_array.is_null(row_idx) {
                    continue; // Skip row if tags list or doc_id is null
                }

                let doc_id = doc_id_array.value(row_idx); // Safe access
                let tags_in_row = tags_list_array.value(row_idx); // This is an ArrayRef (the list items for this row)

                // *** Optimization: Downcast list items to DictionaryArray ***
                // Assuming Int32 keys based on schema generation
                if let Some(tag_dict_array) = tags_in_row
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                {
                    let keys = tag_dict_array.keys();
                    // Get the dictionary values (StringArray) ONCE per dictionary array for efficiency
                    let dict_values = tag_dict_array
                        .values()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            ArrowError::Conversion("Tag dictionary values are not Utf8".to_string())
                        })?;

                    for key_idx in 0..keys.len() {
                        // Iterate over keys in the dictionary for this row's tags
                        if !keys.is_null(key_idx) {
                            let key = keys.value(key_idx);
                            if !dict_values.is_null(key as usize) {
                                // Check null value in dictionary itself
                                // Retrieve the string value from the dictionary using the key
                                // Avoid unnecessary string allocation
                                let tag_str = dict_values.value(key as usize);
                                // Use the string value as the map key
                                map.entry(tag_str.to_string()).or_default().push(doc_id);
                            }
                        }
                    }
                } else {
                    // Fallback or error if not dictionary encoded as expected
                    // Check if it's a simple StringArray (if schema allowed non-dictionary lists)
                    if let Some(string_array) = tags_in_row.as_any().downcast_ref::<StringArray>() {
                        for i in 0..string_array.len() {
                            if !string_array.is_null(i) {
                                let tag_str = string_array.value(i);
                                map.entry(tag_str.to_string()).or_default().push(doc_id);
                            }
                        }
                    } else {
                        // If it's neither Dictionary nor StringArray, it's an unexpected type
                        return Err(ArrowError::Conversion(format!(
                             "Tags list item in row {} has unexpected type: {:?}. Expected DictionaryArray<Int32Type> or StringArray.",
                             row_idx, // Provide more context
                             tags_in_row.data_type()
                         )));
                    }
                }

                // Periodically update memory stats to track usage
                // chunk_rows += 1;
                // if chunk_rows >= CHUNK_SIZE {
                //     stats.update_memory();
                //     chunk_rows = 0;
                // }
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
    let record_count = 10_000_000; // 10 million records for reasonable testing time
    let file_path = "logs_optimized.parquet";
    let generate_and_write = false; // Set to true to generate new data

    if generate_and_write {
        println!("Generating and writing {} records...", record_count);
        let records = generate_log_records(record_count);
        write_records_to_parquet(
            records,
            file_path,
            Compression::ZSTD(Default::default()), // Use ZSTD
            Some(512 * 1024),                      // 128k rows per row group
        )?;
        println!("Finished writing.");
    } else {
        println!(
            "Skipping data generation. Using existing file: {}",
            file_path
        );
        // Ensure the file exists if skipping generation
        if !std::path::Path::new(file_path).exists() {
            eprintln!("Error: Parquet file not found: {}", file_path);
            eprintln!("Set generate_and_write = true in main() to create it.");
            return Err(ArrowError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Parquet file not found: {}", file_path),
            )));
        }
    }

    // Optional short pause
    std::thread::sleep(std::time::Duration::from_secs(10));

    // --- Querying ---
    println!("\n--- Starting Queries ---");

    // Create the reader context once
    let reader_context = ParquetReaderContext::new(file_path)?;

    // Test queries using the context
    // Select a wider range of doc_ids to increase chances of hitting multiple row groups
    let num_docs_to_query = 100; // Query 5k docs
    let doc_ids: Vec<i64> = (0..num_docs_to_query).collect();
    // println!("Querying with {} doc_ids (sampled).", doc_ids.len()); // Reduce verbosity

    // Test get_field_values_by_doc_ids (using dictionary 'level')
    match get_field_values_by_doc_ids(&reader_context, "level", &doc_ids) {
        Ok((level_result, _level_stats)) => {
            println!(
                "[OK] Level values for {} doc_ids: {} unique levels found.",
                doc_ids.len(),
                level_result.value_map.len()
            );
            // println!("Level values map: {:?}", level_result.value_map); // Optional: print map
        }
        Err(e) => eprintln!("[ERR] Getting level values by doc_id: {}", e),
    }

    // Test get_field_values (using dictionary 'source_region')
    match get_field_values(&reader_context, "source_region") {
        Ok((all_regions, _all_regions_stats)) => {
            println!(
                "[OK] All source_regions: {} unique regions found.",
                all_regions.value_map.len()
            );
        }
        Err(e) => eprintln!("[ERR] Getting all source_region values: {}", e),
    }

    // Optional short pause
    // std::thread::sleep(std::time::Duration::from_secs(1));

    // Test get_numeric_stats_by_doc_ids
    match get_numeric_stats_by_doc_ids(&reader_context, "payload_size", &doc_ids) {
        Ok((payload_stats, _payload_stats_query)) => {
            println!(
                "[OK] Payload stats for {} doc_ids: min={:?}, max={:?}, avg={:.2}",
                doc_ids.len(),
                payload_stats.min,
                payload_stats.max,
                payload_stats.avg
            );
        }
        Err(e) => eprintln!("[ERR] Getting payload_size stats by doc_id: {}", e),
    }

    // Optional short pause
    // std::thread::sleep(std::time::Duration::from_secs(1));

    // Test get_numeric_stats
    match get_numeric_stats(&reader_context, "payload_size") {
        Ok((all_payload_stats, _all_payload_stats_query)) => {
            println!(
                "[OK] All payload stats: min={:?}, max={:?}, avg={:.2}, count={}",
                all_payload_stats.min,
                all_payload_stats.max,
                all_payload_stats.avg,
                all_payload_stats.count
            );
        }
        Err(e) => eprintln!("[ERR] Getting all payload_size stats: {}", e),
    }

    // Optional short pause
    // std::thread::sleep(std::time::Duration::from_secs(1));

    // Test get_tag_buckets (Optimized with DictionaryArray)
    match get_tag_buckets(&reader_context) {
        Ok((tag_buckets, _tag_buckets_stats)) => {
            println!("[OK] Tag buckets: {} unique tags found.", tag_buckets.len());
        }
        Err(e) => eprintln!("[ERR] Getting tag buckets: {}", e),
    }

    println!("\n--- Queries Finished ---");

    Ok(())
}
