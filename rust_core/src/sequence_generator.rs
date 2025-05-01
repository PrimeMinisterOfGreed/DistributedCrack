#[derive(Debug)]
#[repr(C)]
struct SequenceGeneratorCtx {
    index: usize,
    base_len: u8,
    curren_len: u8,
    buffer: [i8; 32],
}

const span: usize = 93;

unsafe extern "C" {
    fn new_seq_generator(base_len: u8) -> SequenceGeneratorCtx;
    fn seq_gen_next_sequence(ctx: *mut SequenceGeneratorCtx);
    fn seq_gen_skip_to(ctx: *mut SequenceGeneratorCtx, address: usize);
}

pub struct SequenceGenerator {
    ctx: SequenceGeneratorCtx,
}

pub trait ChunkGenerator {
    fn generate_flatten_chunk(&mut self, chunks: usize) -> GeneratorResult;
}

pub struct GeneratorResult {
    pub strings: Vec<u8>,
    pub sizes: Vec<u8>,
}

impl SequenceGenerator {
    pub fn new(base_len: u8) -> Self {
        unsafe {
            Self {
                ctx: new_seq_generator(base_len),
            }
        }
    }

    pub fn absolute_index(&self) -> usize {
        self.ctx.index as usize + (span * (self.ctx.base_len as usize - 1))
    }

    pub fn remaining_this_size(&self) -> usize {
        93 * (self.ctx.curren_len as usize + 1) - self.absolute_index()
    }

    pub fn next_sequence(&mut self) {
        unsafe {
            seq_gen_next_sequence(&mut self.ctx);
        }
    }

    pub fn skip_to(&mut self, address: usize) {
        unsafe {
            seq_gen_skip_to(&mut self.ctx, address);
        }
    }

    pub fn get_buffer(&self) -> &[i8] {
        &self.ctx.buffer
    }

    pub fn current(&self) -> String {
        let mut s = String::new();
        for i in 0..self.ctx.curren_len as usize {
            s.push(self.ctx.buffer[i] as u8 as char);
        }
        s
    }
}

impl ChunkGenerator for SequenceGenerator {
    fn generate_flatten_chunk(&mut self, chunks: usize) -> GeneratorResult {
        let mut chunk: Vec<u8> = Vec::with_capacity((self.ctx.curren_len as usize + 1) * span);
        let mut sizes: Vec<u8> = Vec::with_capacity(chunks);
        for _ in 0..chunks {
            let buffer = self.ctx.buffer;
            sizes.push(self.ctx.curren_len);
            for c in buffer {
                if c == 0 {
                    break;
                }
                chunk.push(c as u8);
            }
            self.next_sequence();
        }
        GeneratorResult {
            strings: chunk,
            sizes: sizes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation() {
        let mut generator = SequenceGenerator::new(4);
        println!("Remaining this size: {}", generator.remaining_this_size());
        println!("Ctx {:?}", generator.ctx);
        for _ in 0..10 {
            println!("{}", generator.current());
            generator.next_sequence();
        }
    }

    #[test]
    fn test_compact_generation() {
        let mut generator = SequenceGenerator::new(4);
        let chunk = generator.generate_flatten_chunk(10);
        println!("{:?}", chunk.strings);
    }
}
