use ndarray::prelude::*;
use ndarray_linalg::{Factorize,Inverse};
use mtx_dim;
use node::StrippedNode;
use compact_predictor::median_matrix;

pub fn vec_mtx_to_array<T: Clone>(source: &Vec<Vec<T>>) -> Array<T,Ix2> {
    let dim = mtx_dim(source);
    let array_encoding = Array::from_shape_vec(dim,source.iter().flat_map(|x| x.iter().cloned()).collect()).unwrap();
    array_encoding
}

pub fn array_to_vec_mtx<T: Clone>(source: &Array<T,Ix2>) -> Vec<Vec<T>> {

    let mut output = vec![];

    for row in source.outer_iter() {
        output.push(row.to_vec());
    }

    output
}

pub fn weigh_leaves(leaves: &Vec<&StrippedNode>, leaf_encoding:&Vec<Vec<bool>>, truth_vec: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let dim = mtx_dim(leaf_encoding);
    let array_encoding = Array::from_shape_vec(dim,leaf_encoding.iter().flat_map(|x| x.iter()).map(|y| *y as i8 as f64).collect()).unwrap();
    let raw_predictions = vec_mtx_to_array(&median_matrix(leaves));
    let truth = vec_mtx_to_array(truth_vec);
    let weighted_predictions: Option<Array<f64,Ix2>> = None;
    if let Ok(factorized) = array_encoding.factorize() {
        if let Ok(encoding_inverse) = factorized.inv() {
            let weighted_predictions = Some(truth * encoding_inverse);
        };
    };
    let mut weights = weighted_predictions.unwrap() / raw_predictions;
    weights.mapv_inplace(|x| x.abs());

    array_to_vec_mtx(&weights)

}
