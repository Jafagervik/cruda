#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![allow(improper_ctypes_definitions)]

use cuda_std::{vek, *};

extern crate alloc;

/// Gets the 1d index from a 2d index
macro_rules! at {
    ($row:ident, $col:ident, $width:ident) => {
        $row * $width + $col
    };
}

const BLOCK_SIZE: usize = 16;

// static PTX: &str = include_str!("../ptxdir/path.ptx");

#[kernel]
pub unsafe fn add(a: &[f32], b: &[f32], c: *mut f32) {
    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = &mut *c.add(idx);
        *elem = a[idx] + b[idx];
    }
}

/// Classic SAXPY
/// a: matrix
/// x: vector
/// y: value
/// n: size
#[kernel]
pub unsafe fn saxpy(a: &[f32], x: &[f32], y: f32, res: *mut f32, n: u32) {
    let idx = thread::index_1d();

    // If we're out of bounds, return!
    if idx >= n {
        return;
    }

    // start from correct index
    let out = &mut *res.add(idx);

    // Multiply one row in the matrix with the values in the vector, and store
    // to the right index in out, calculated at last step by the idx.
    let val = a
        .iter()
        .skip(n * idx)
        .take(n)
        .enumerate()
        .map(|(i, elem)| elem * x.iter().nth(i).unwrap_or_else(|| 1))
        .sum();

    *out = val;
}

/// Classic DAXPY
/// a: matrix
/// x: vector
/// y: value
/// n: size
#[kernel]
pub unsafe fn daxpy(a: &[f64], x: &[f64], y: f64, res: *mut f64, n: u32) {
    let idx = thread::index_1d();

    // If we're out of bounds, return!
    if idx >= n {
        return;
    }

    // start from correct index
    let out = &mut *res.add(idx);

    // Multiply one row in the matrix with the values in the vector, and store
    // to the right index in out, calculated at last step by the idx.
    let val = a
        .iter()
        .skip(n * idx)
        .take(n)
        .enumerate()
        .map(|(i, elem)| elem * x.iter().nth(i).unwrap_or_else(|| 1))
        .sum();

    *out = val;
}

// TODO: Add convolution

/// Classic GEMM with f64 IEEE 754
/// a: matrix
/// b: matrix b where b is transposed
/// n: size
#[kernel]
pub unsafe fn dgemm(a: &[f64], b: &[f64], c: *mut f64, n: u32) {
    let block_row = thread::block_idx_y();
    let block_col = thread::block_idx_x();

    // If we're out of bounds, return!
    if row >= n || col >= n {
        return;
    }

    // get sub matrix
    let c_sub = 0;

    let mut c_value: f64 = 0.0;

    let row = thread::thread_idx_y();
    let col = thread::thread_idx_x();

    for m in 0..(n / BLOCK_SIZE) {
        // TODO: Add get sub matrix
        let a_sub_mat = 0;
        let b_sub_mat = 0;

        // Shared memory
        let s_a = shared_array![f64; BLOCK_SIZE*BLOCK_SIZE];
        let s_b = shared_array![f64; BLOCK_SIZE*BLOCK_SIZE];

        s_a[at!(row, col, n)] = a.iter().nth(at!(row, col, stride)).unwrap();
        s_b[at!(row, col, n)] = b.iter().nth(at!(row, col, stride)).unwrap();

        thread::sync_threads();

        for e in 0..BLOCK_SIZE {
            c_value += s_a[at!(row, e, N)] * s_b[at!(col, e, n)];
        }

        thread::sync_threads();
    }

    // start from correct index
    let out = &mut *c.add(at!(row, col, (n / BLOCK_SIZE)));

    *out = c_value;
}

// Transposition of a matrix
#[kernel]
pub unsafe fn transpose<T: Copy>(in_mat: &[T], out_mat: *mut T, n: usize) {
    let x = thread::block_idx_x() * BLOCK_SIZE + thread::thread_idx_x();
    let y = thread::block_idx_y() * BLOCK_SIZE + thread::thread_idx_y();

    if x >= n || y >= n {
        return;
    }

    let tile = shared_array![T; BLOCK_SIZE * BLOCK_SIZE];

    let width = thread::grid_dim_x() * BLOCK_SIZE;

    for j in 0..BLOCK_SIZE.step_by(BLOCK_SIZE) {
        tile[(thread::thread_idx_y() + j) * BLOCK_SIZE + thread::thread_idx_x()] =
            in_mat.iter().nth(at!((y + j), x, width)).unwrap();
    }
    thread::sync_threads();

    for j in 0..BLOCK_SIZE.step_by(BLOCK_SIZE) {
        let elem = &mut *out_mat.add(at!((y + j), x, width));

        *elem = tile[at!((thread::thread_idx_y + j), thread::thread_idx_x, BLOCK_SIZE)];
    }
}

/// Flip an image vertically
#[kernel]
pub unsafe fn img_flip_v() {
    todo!()
}
