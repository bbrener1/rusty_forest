

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }

    // #[test]
    // fn slow_description_test() {
    //     let test_vector_1 = vec![-2.,-1.,0.,1.,1000];
    //     let test_vector_2 = vec![0.,0.,0.,0.,0.,1.,2.];
    //     let test_vector_3 = vec![2.,3.,1.,0.,4.,1.,3.,-1.]; //1.5,1.5
    //
    //     let description_1 = slow_description(test_vector_1);
    //     let description_2 = slow_description(test_vector_2);
    //     let description_3 = slow_description(test_vector_3);
    //
    //     assert_eq!(description_1,(0.,1.0));
    //     assert_eq!(description_2,(0.,0.));
    //     assert_eq!(description_3,(1.5,1.5));
    // }
    //
    // fn slow_vs_fast () {
    //
    //     /// Generate some random feature vectors
    //
    //     let mut thr_rng = rand::thread_rng();
    //     let rng = thr_rng.gen_iter::<f64>();
    //
    //     let counts = Vec::new();
    //
    //     for feature in 0..10 {
    //         counts.push(rng.take(49).collect());
    //     }
    //
    //     let matrix = matrix_flip(&counts);
    //
    //     let slow_medians = Vec::new();
    //
    //     for feature in counts {
    //         let mut feature_descriptions = Vec::new();
    //         for (i, sample) in feature.iter().enumerate() {
    //
    //         feature_descriptions.push(slow_description(feature[i..]));
    //         }
    //         slow_medians.push(feature_descriptions);
    //     }
    //
    //     // We have generated feature medians and median distances using the slow function, checked above.
    //     //
    //
    //
    //     let model = OnlineMADM::new(&matrix);
    //
    //     let fast_medians = model.variance_by_feature();
    //
    //     assert_eq!(slow_medians,fast_medians);
    //
    // }

}
