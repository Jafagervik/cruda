use clap::Parser;
use rustcuda::*;

static PTX: &str = include_str!("../ptxdir/path.ptx");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    /// Name of kernel to run
    #[arg(short, long)]
    name: String,

    /// Size of square matrix
    #[arg(short, long, default_value = 3)]
    size: u8,
}

fn main() {
    let args = Args::parse();

    // Match on name and run corresponding kernel for this amount of work
    match args.name {
        String::from("gemm") => {}
        String::from("daxpy") => {}
        String::from("flipv") => {}
        String::from("fliph") => {}
        _ => {}
    };

    println!("Hello, world!");
}

unsafe fn runner() {
    launch!(
    module.kernel<<<1,1,10,stream>>>()
    )
}
