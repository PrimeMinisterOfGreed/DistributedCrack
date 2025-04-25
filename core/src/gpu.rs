use std::{
    ffi::{CStr, CString, c_char},
    ptr::copy,
};
/*
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Md5TransformResult {
    pub data: *mut ::core::ffi::c_char,
    pub size: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Md5BruterResult {
    pub data: [::core::ffi::c_char; 33usize],
    pub found: bool,
}

unsafe extern "C" {
    unsafe fn md5_gpu(
        data: *mut ::core::ffi::c_char,
        sizes: *mut u8,
        array_size: usize,
        maxthreads: ::core::ffi::c_int,
    ) -> Md5TransformResult;
}
unsafe extern "C" {
    unsafe fn md5_bruter(
        start_address: usize,
        end_address: usize,
        target_md5: *const ::core::ffi::c_char,
        maxthreads: ::core::ffi::c_int,
        base_str_len: ::core::ffi::c_int,
    ) -> Md5BruterResult;
}

unsafe extern "C" {
    fn md5String(input: *const c_char, output: *mut c_char);
    fn md5HexDigest(input: *const u8, output: *mut c_char);
}
*/

include!(concat!(env!("OUT_DIR"), "/md5_bindings.rs"));

pub fn md5_cpu(value: &CString) -> String {
    let mut buffer = [0u8; 16];
    let mut hexdigest = [0u8; 32];
    unsafe {
        md5String(value.clone().into_raw(), buffer.as_mut_ptr() as *mut u8);
        md5HexDigest(buffer.as_mut_ptr(), hexdigest.as_mut_ptr() as *mut i8);
    };
    String::from_utf8_lossy(&hexdigest).into_owned()
}

pub fn md5_transform(data: &Vec<u8>, sizes: &Vec<u8>, maxthreads: u32) -> Vec<String> {
    let dataptr = data.as_ptr();
    let sizeptr = sizes.as_ptr();
    let array_size = sizes.len();
    let mut results = Vec::<String>::new();
    unsafe {
        let mut buffer = [0u8; 32];
        let result = md5_gpu(
            dataptr as *mut ::core::ffi::c_char,
            sizeptr as *mut u8,
            array_size,
            maxthreads as i32,
        );
        for i in 0..result.size {
            copy(result.data.add(i * 32), buffer.as_mut_ptr() as *mut i8, 32);
            results.push(String::from_utf8(buffer.to_vec()).unwrap());
        }
    }
    results
}

pub fn md5_brute(
    start_address: usize,
    end_address: usize,
    target_md5: &CString,
    maxthreads: u32,
    base_str_len: u32,
) -> Option<String> {
    let mut buffer = [0u8; 32];
    unsafe {
        let result = md5_bruter(
            start_address,
            end_address,
            target_md5.as_ptr(),
            maxthreads as i32,
            base_str_len as i32,
        );
        if !result.found {
            None
        } else {
            copy(result.data.as_ptr(), buffer.as_mut_ptr() as *mut i8, 32);
            let mut size = 0;
            while buffer[size] != 0 {
                size += 1;
            }

            Some(String::from_utf8(buffer[0..size].to_vec()).unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::sequence_generator::{ChunkGenerator, SequenceGenerator};

    use super::*;

    #[test]
    fn test_md5_transform() {
        let mut generator = SequenceGenerator::new(4);
        let data = generator.generate_flatten_chunk(1000);
        let result = md5_transform(&data.strings, &data.sizes, 1000);
        assert_eq!(result[0], "98abe3a28383501f4bfd2d9077820f11")
    }

    #[test]
    fn test_md5_brute() {
        let result = md5_brute(
            0,
            1000,
            &CString::from_str("98abe3a28383501f4bfd2d9077820f11").unwrap(),
            1000,
            4,
        );
        assert_eq!(result.unwrap(), "!!!!")
    }
}
