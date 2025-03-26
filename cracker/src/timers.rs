use lazy_static::lazy_static;
use std::{default, time::Instant};
#[derive(Clone, Copy)]
pub struct ClockStats {
    pub busy_time: u64,
    pub observation_time: u64,
    pub start: Instant,
    pub task_count: u64,
}

impl ClockStats {
    pub fn new() -> Self {
        ClockStats {
            busy_time: 0,
            observation_time: 0,
            start: Instant::now(),
            task_count: 0,
        }
    }

    pub fn utilization(&self) -> f64 {
        self.busy_time as f64 / self.observation_time as f64
    }

    pub fn throughput(&self) -> f64 {
        self.task_count as f64 / self.observation_time as f64
    }

    pub fn latency(&self) -> f64 {
        self.observation_time as f64 / self.task_count as f64
    }

    pub fn mean_service_time(&self) -> f64 {
        self.busy_time as f64 / self.task_count as f64
    }
}

struct Context {
    name: String,
    stats: ClockStats,
}

impl Context {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            stats: ClockStats::new(),
        }
    }
}

pub struct GlobalClock {
    pub start: Instant,
    contexts: Vec<Context>,
}

lazy_static! {
    static ref GLOBAL_CLOCK: GlobalClock = GlobalClock::new();
}

impl GlobalClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            contexts: Vec::new(),
        }
    }

    pub fn with_context(&mut self, name: &str, f: impl Fn() -> i32) {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_millis();
        let mut ctxs = &mut self.contexts;
        let mut ctx = ctxs.iter_mut().find(|ctx| ctx.name == name);
        let mut res = match ctx {
            Some(ctx) => ctx,
            None => {
                ctxs.push(Context::new(name));
                ctxs.last_mut().unwrap()
            }
        };
        res.stats.busy_time += elapsed as u64;
        res.stats.observation_time += elapsed as u64;
        res.stats.task_count += result as u64;
    }

    pub fn get_stats(&self, name: &str) -> Option<ClockStats> {
        let ctx = self.contexts.iter().find(|ctx| ctx.name == name);
        if let Some(ct) = ctx {
            Some(ct.stats)
        } else {
            None
        }
    }

    pub fn report_stats(&self) {
        for ctx in &self.contexts {
            println!("Context: {}", ctx.name);
            println!("Utilization: {}", ctx.stats.utilization());
            println!("Throughput: {}", ctx.stats.throughput());
            println!("Latency: {}", ctx.stats.latency());
            println!("Mean Service Time: {}", ctx.stats.mean_service_time());
        }
    }
}
