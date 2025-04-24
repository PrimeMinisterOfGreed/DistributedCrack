use bindgen::{self, callbacks::ParseCallbacks};
use pkg_config::Config;
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
    // parse and use mpich pkg config
    let openmpi = Config::new().atleast_version("3.0").probe("mpi").unwrap();
    println!("cargo:rustc-link-lib=static={}", openmpi.libs[0]);
    println!(
        "cargo:rustc-link-search=native={}",
        openmpi.link_paths[0].display()
    );
    for lib in openmpi.libs.iter() {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }
    // link to pthread

    // copy libmd5_gpu.so to build dir
    let lib_path = format!("{}/libs/gpu_md5/libmd5_gpu.so", build.display());
    println!("lib_path: {}", lib_path);
    let tgt = env::var("OUT_DIR").unwrap();
    let target_path = Path::new(&tgt)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("deps");
    std::fs::copy(lib_path, &target_path)
        .expect(format!("Failed to copy libmd5_gpu.so to {}", target_path.display()).as_str());
    println!("Copied libmd5_gpu.so to {}", target_path.display());

    // Generate bindings for openmpi /usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h
    let bindings = bindgen::Builder::default()
        .header("/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h")
        .clang_arg("-I/usr/lib/x86_64-linux-gnu/openmpi/include")
        .fit_macro_constants(false)
        .use_core()
        .clang_macro_fallback()
        .derive_debug(true)
        .derive_default(true)
        .default_macro_constant_type(bindgen::MacroTypeVariation::Signed)
        .generate()
        .expect("Unable to generate bindings");
    let out_path = Path::new(&env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}
