use std::{env, path::Path};

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
}
