use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use sysinfo::{ProcessExt, System, SystemExt};

// Track global scan count across queries
static SCAN_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct TimingBreakdown {
    pub setup: Duration,
    pub filter_creation: Duration,
    pub join_operation: Duration,
    pub collect: Duration,
    pub processing: Duration,
    pub total: Duration,
}

impl fmt::Display for TimingBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Setup: {:?}, Filter: {:?}, Join: {:?}, Collect: {:?}, Process: {:?}, Total: {:?}",
            self.setup,
            self.filter_creation,
            self.join_operation,
            self.collect,
            self.processing,
            self.total
        )
    }
}

#[derive(Debug, Clone)]
pub struct QueryStats {
    pub query_name: String,
    pub field_name: String,
    pub doc_ids_count: Option<usize>,
    pub scan_count: usize,
    pub result_rows: usize,
    pub timing: TimingBreakdown,
    pub memory_before: u64, // in KB
    pub memory_after: u64,  // in KB
    pub memory_peak: u64,   // in KB
}

impl QueryStats {
    pub fn new(query_name: &str, field_name: &str, doc_ids_count: Option<usize>) -> Self {
        let memory = get_process_memory_usage();
        SCAN_COUNT.fetch_add(1, Ordering::SeqCst);

        QueryStats {
            query_name: query_name.to_string(),
            field_name: field_name.to_string(),
            doc_ids_count,
            scan_count: SCAN_COUNT.load(Ordering::SeqCst),
            result_rows: 0,
            timing: TimingBreakdown {
                setup: Duration::default(),
                filter_creation: Duration::default(),
                join_operation: Duration::default(),
                collect: Duration::default(),
                processing: Duration::default(),
                total: Duration::default(),
            },
            memory_before: memory,
            memory_after: memory,
            memory_peak: memory,
        }
    }

    pub fn update_memory(&mut self) {
        let current = get_process_memory_usage();
        self.memory_after = current;
        if current > self.memory_peak {
            self.memory_peak = current;
        }
    }

    pub fn set_result_rows(&mut self, rows: usize) {
        self.result_rows = rows;
    }

    pub fn print_summary(&self) {
        let before_mb = self.memory_before as f64 / 1024.0;
        let after_mb = self.memory_after as f64 / 1024.0;
        let peak_mb = self.memory_peak as f64 / 1024.0;
        let memory_diff = if self.memory_after > self.memory_before {
            format!(
                "+{:.2} MB",
                (self.memory_after - self.memory_before) as f64 / 1024.0
            )
        } else {
            format!(
                "-{:.2} MB",
                (self.memory_before - self.memory_after) as f64 / 1024.0
            )
        };

        println!("┌─────────────────────────────────────────────────────────────────────┐");
        println!("│ 📊 QUERY STATS SUMMARY                                              │");
        println!("├─────────────────────────────────────────────────────────────────────┤");
        println!("│ Query:         {:<52} │", self.query_name);
        println!("│ Field:         {:<52} │", self.field_name);
        if let Some(count) = self.doc_ids_count {
            println!("│ Doc IDs:       {:<52} │", format!("{} ids", count));
        }
        println!("│ Scan #:        {:<52} │", self.scan_count);
        println!("│ Result Rows:   {:<52} │", self.result_rows);
        println!("├─────────────────────────────────────────────────────────────────────┤");
        println!("│ ⏱️  Timing                                                          │");
        println!(
            "│   Setup:       {:<52} │",
            format!("{:?}", self.timing.setup)
        );
        println!(
            "│   Filter:      {:<52} │",
            format!("{:?}", self.timing.filter_creation)
        );
        println!(
            "│   Join:        {:<52} │",
            format!("{:?}", self.timing.join_operation)
        );
        println!(
            "│   Collect:     {:<52} │",
            format!("{:?}", self.timing.collect)
        );
        println!(
            "│   Processing:  {:<52} │",
            format!("{:?}", self.timing.processing)
        );
        println!(
            "│   Total:       {:<52} │",
            format!("{:?}", self.timing.total)
        );
        println!("├─────────────────────────────────────────────────────────────────────┤");
        println!("│ 🧠 Memory                                                           │");
        println!("│   Before:      {:<52} │", format!("{:.2} MB", before_mb));
        println!("│   After:       {:<52} │", format!("{:.2} MB", after_mb));
        println!("│   Peak:        {:<52} │", format!("{:.2} MB", peak_mb));
        println!("│   Diff:        {:<52} │", memory_diff);
        println!("└─────────────────────────────────────────────────────────────────────┘");
    }

    pub fn print_compact(&self) {
        println!(
            "[{}] '{}' - {:?} - {} rows - Mem: {}KB → {}KB",
            self.query_name,
            self.field_name,
            self.timing.total,
            self.result_rows,
            self.memory_before,
            self.memory_after
        );
    }
}

fn get_process_memory_usage() -> u64 {
    let mut system = System::new_all();
    system.refresh_all();
    let pid = sysinfo::get_current_pid().expect("Failed to get PID");
    system
        .process(pid)
        .map(|process| process.memory() / 1024) // Convert to KB
        .unwrap_or(0)
}

// Helper macro for timing sections of code
#[macro_export]
macro_rules! time_section {
    ($stats:expr, $section:ident, $code:block) => {{
        let start = Instant::now();
        let result = $code;
        $stats.timing.$section = start.elapsed();
        $stats.update_memory();
        result
    }};
}
