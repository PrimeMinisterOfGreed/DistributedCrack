use std::{env, fmt::format, path::Path};

pub fn main() {
    //find path of the build folder
    let mut build = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("build");
    println!(
        "cargo:rustc-link-search=native={}/libs/md5",
        build.display()
    );
    println!("cargo:rustc-link-lib=static=md5");
    println!(
        "cargo:rustc-link-search=native={}/libs/gpu_md5",
        build.display()
    );
    println!("cargo:rustc-link-lib=dylib=md5_gpu");
    // copy libmd5_gpu.so to build dir
    let lib_path = format!("{}/libs/gpu_md5/libmd5_gpu.so", build.display());
    let target_path = format!("{}/libmd5_gpu.so", env::var("OUT_DIR").unwrap());
    std::fs::copy(lib_path, &target_path).expect("Failed to copy libmd5_gpu.so");
    println!("Copied libmd5_gpu.so to {}", &target_path.as_str());
}
