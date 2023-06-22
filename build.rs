use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../rustcuda")
        .copy_to("ptxdir/path.ptx")
        .build()
        .unwrap();
}
