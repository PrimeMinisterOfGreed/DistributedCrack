pub enum ComputeContext {
    Brute(usize, usize),
    Chunked(Vec<String>),
}

pub fn compute(context: ComputeContext) {
    match context {
        ComputeContext::Brute(start, end) => {
            brute_mode(start, end);
        }
        ComputeContext::Chunked(chunks) => {
            chunked_mode(chunks);
        }
    }
}

pub fn chunked_mode(chunk: Vec<String>) -> Option<String> {
    None
}

pub fn brute_mode(start: usize, end: usize) -> Option<String> {
    None
}
