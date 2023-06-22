use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../rustcuda")
        .copy_to("some/path.ptx")
        .build()
        .unwrap();
}
