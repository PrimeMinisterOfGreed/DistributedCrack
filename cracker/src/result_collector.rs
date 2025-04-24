use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::timers::{ClockStats, Context, GlobalClock};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ResultData {
    context_name: String,
    busy_time: u64,
    observation_time: u64,
    task_count: u64,
    throughput: f64,
    mean_service_time: f64,
}

pub struct ResultFile {
    results: Vec<ResultData>,
}

impl ResultFile {
    pub fn save_to_file(&self, filepath: &str) -> Result<(), std::io::Error> {
        let mut wtb = csv::WriterBuilder::new();
        if std::fs::exists(filepath).ok().unwrap() {
            wtb.has_headers(false);
        }
        wtb.delimiter(';' as u8)
            .quote_style(csv::QuoteStyle::Necessary)
            .flexible(true);
        let mut file = std::fs::File::options()
            .read(true)
            .create(true)
            .append(true)
            .write(true)
            .open(filepath)?;
        let mut writer = wtb.from_writer(file);
        for data in &self.results {
            writer.serialize(data)?;
        }
        Ok(())
    }

    pub fn collect() -> Option<Self> {
        let mut result_file = ResultFile {
            results: Vec::new(),
        };

        for context in GlobalClock::instance().get_contexts() {
            let stats = context.stats();
            let result_data = ResultData {
                context_name: context.name().to_string(),
                busy_time: stats.busy_time,
                observation_time: stats.observation_time,
                task_count: stats.task_count,
                throughput: stats.throughput(),
                mean_service_time: stats.mean_service_time(),
            };
            result_file.results.push(result_data);
        }
        if result_file.results.is_empty() {
            return None;
        }
        Some(result_file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let result_data = ResultData {
            context_name: "TestContext".to_string(),
            busy_time: 100,
            observation_time: 200,
            task_count: 50,
            throughput: 2.0,
            mean_service_time: 1.5,
        };
        let result_file = ResultFile {
            results: vec![
                result_data,
                ResultData {
                    context_name: "AnotherContext".to_string(),
                    busy_time: 150,
                    observation_time: 250,
                    task_count: 75,
                    throughput: 3.0,
                    mean_service_time: 2.0,
                },
            ],
        };
        let filepath = "test_results.csv";
        result_file.save_to_file(filepath).unwrap();
    }
}
