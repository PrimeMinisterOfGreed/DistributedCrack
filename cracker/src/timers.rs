use lazy_static::lazy_static;
use std::{
    default,
    sync::{Mutex, MutexGuard},
    time::Instant,
};
#[derive(Clone, Copy, Default, Debug)]
pub struct ClockStats {
    pub busy_time: u64,
    pub observation_time: u64,
    pub task_count: u64,
}

impl ClockStats {
    pub fn new() -> Self {
        ClockStats {
            busy_time: 0,
            observation_time: 0,
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

#[derive(Clone, Debug)]
pub struct Context {
    name: heapless::String<32>,
    stats: ClockStats,
}

impl Context {
    pub fn new(name: &str) -> Self {
        let mut this = Self {
            name: heapless::String::new(),
            stats: ClockStats::new(),
        };
        this.name.push_str(name).unwrap();
        this
    }
}

pub struct GlobalClock {
    pub start: Instant,
    contexts: Vec<Context>,
}

lazy_static! {
    static ref GLOBAL_CLOCK: Mutex<GlobalClock> = Mutex::new(GlobalClock::new());
}

impl GlobalClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            contexts: Vec::new(),
        }
    }
    pub fn instance() -> MutexGuard<'static, GlobalClock> {
        GLOBAL_CLOCK.lock().unwrap()
    }

    pub fn add_context(&mut self, ctx: Context) {
        self.contexts.push(ctx);
    }

    pub fn get_contexts(&self) -> impl Iterator<Item = &Context> {
        self.contexts.iter()
    }

    pub fn get_context(&self, name: &str) -> Option<&Context> {
        self.contexts.iter().find(|ctx| ctx.name == name)
    }

    pub fn with_context(&mut self, name: &str, mut f: impl FnMut() -> i32) {
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

#[cfg(test)]
mod tests {
    use std::mem::transmute;

    use super::*;

    #[test]
    fn test_clock() {
        let mut clock = GlobalClock::new();
        clock.with_context("test", || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            1
        });
        clock.with_context("test", || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            2
        });
        clock.report_stats();
    }

    #[test]
    fn test_context_serialization() {
        let mut clock = GlobalClock::new();
        clock.with_context("test", || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            1
        });
        clock.with_context("test", || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            2
        });
        let mut stat = clock.get_context("test").unwrap();

        let mut buffer = [0u8; size_of::<Context>()];
        unsafe {
            core::ptr::copy(
                transmute::<_, *const [u8; size_of::<Context>()]>(stat),
                buffer.as_mut_ptr() as *mut [u8; size_of::<Context>()],
                size_of::<Context>(),
            )
        };
        println!("Buffer: {:?}", buffer);
        let mut stat_copy: Context = unsafe { transmute(buffer) };
        println!("Stat : {:?}", stat);
        println!("Stat Copy: {:?}", stat_copy);
    }
}
