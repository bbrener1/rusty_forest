
// fn sort_median(input: &Vec<f64> ) -> (usize,f64,Vec<(usize,f64)>) {
//
//     let mut index = 0;
//     let mut value = 0.;
//
//
//     let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
//     intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
//     let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
//
//     if intermediate1.len() % 2 == 0 {
//         index = intermediate1.len()/2;
//         value = (intermediate1[index].1 + intermediate1[index-1].1) / 2.
//     }
//     else {
//         if intermediate1.len() % 2 == 1 {
//             index = (intermediate1.len()-1)/2;
//             value = *intermediate1[index].1;
//         }
//         else {
//             panic!("Median failed!");
//         }
//     }
//
//     intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
//     let out = intermediate2.iter().map(|x| (x.0,((x.1).1).clone())).collect();
//
//     (index,value,out)
//
// }


// fn recompute_distances_from_median(input: &mut Vec<(usize,f64)>, index: usize, change: f64 ) {
//     for i in input {
//         match (i.0).cmp(&index){
//             Ordering::Greater => i.1 -= change,
//             Ordering::Less => i.1 += change,
//             Ordering::Equal => i.1 = 0.
//         }
//     }
// }
//     fn find_split(&mut self, feature:usize) -> (usize,f64) {
//
//         // let tree: &Tree = self.tree.borrow();
//
//         // let counts: &Vec<Vec<f64>> = self.tree.counts.upgrade().expect("Dead tree").borrow();
//
//         println!("Splitting!");
//
//         let global_counts = self.counts.upgrade().expect("Missing counts?");
//
//         let node_counts: Vec<Vec<f64>> = self.indecies.iter().map(|x| self.output_features.iter().map(|y| global_counts[*x][*y]).collect()).collect();
//         // let node_counts: Vec<Vec<f64>> = self.indecies.iter().map(|x| global_counts[x.clone()]).collect();
//
//
//         // let mut local_variance = self.variance.clone();
//         // let local_means = self.means.clone();
//
//         println!("Node counts cloned: {}, {}", node_counts.len(),node_counts[0].len());
//
//
//         let dimensions: (usize,usize) = (node_counts.len(),node_counts[0].len());
//
//         let mut local_mean_upper_sum: Vec<f64> = vec![0f64;dimensions.1];
//         let mut local_mean_lower_sum: Vec<f64> = inner_axis_sum(&node_counts);
//
//
//         let mut local_variance_upper_sum: Vec<f64> = vec![0f64;dimensions.1];
//         let mut local_variance_lower_sum: Vec<f64> = inner_axis_variance_sum(&node_counts, None);
//
//         let mut feature_vector: Vec<(usize,f64)> = self.indecies.iter().map(|x| global_counts[*x].clone()).enumerate().map(|x| (x.0,x.1[feature])).collect();
//
//         feature_vector.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//
//         let mut agg_cv: Vec<(f64,f64)> = Vec::new();
//         let mut cv_mat: Vec<(Vec<f64>,Vec<f64>)> = Vec::new();
//         let mut svar_mat: Vec<(Vec<f64>,Vec<f64>)> = Vec::new();
//         let mut sm_mat: Vec<(Vec<f64>,Vec<f64>)> = Vec::new();
//
//         for (si,sample) in feature_vector.iter().map(|x| &node_counts[x.0]).enumerate() {
//             for (fi,feature) in sample.iter().enumerate() {
//
//                 if si % 300 == 0 {
//                     println!("{:?}", local_mean_upper_sum.iter().take(10).collect::<Vec<&f64>>());
//                     println!("{:?}", local_mean_lower_sum.iter().take(10).collect::<Vec<&f64>>());
//                     println!("{:?}", local_variance_upper_sum.iter().take(10).collect::<Vec<&f64>>());
//                     println!("{:?}", local_variance_lower_sum.iter().take(10).collect::<Vec<&f64>>());
//                 }
//
//                 local_mean_upper_sum[fi] = local_mean_upper_sum[fi] + feature;
//                 local_mean_lower_sum[fi] = local_mean_lower_sum[fi] - feature;
//
// // CHECK INDEXING HERE, BE VERY CAREFUL
//
// ////////////////////////////////// CALCULATE YOUR OWN DAMN MEANS //////////////////////////////////////
// ////////////////////////////// VARIANCE IS SQUARED DIPSHIT //////////////////////////////////////////////
//
//                 // Computing running variance has to be done with two sums, one for positively-deviating features and one for negatively-deviating features.
//
//
//
//                 local_variance_upper_sum[fi] = local_variance_upper_sum[fi] + (local_mean_upper_sum[fi]/((si+1) as f64) - feature).powi(2);
//                 local_variance_lower_sum[fi] = local_variance_lower_sum[fi] -
//                 (local_mean_lower_sum[fi]/((dimensions.0 - si) as f64) - feature).powi(2);
//
//             }
//
//             let upper_split_variance: Vec<f64> = local_variance_upper_sum.iter().map(|x| x / ((si+1) as f64)).collect();
//             let lower_split_variance: Vec<f64> = local_variance_lower_sum.iter().map(|x| x / ((dimensions.0 - si) as f64)).collect();
//
//             let upper_split_means: Vec<f64> = local_mean_upper_sum.iter().map(|x| x / ((si+1) as f64)).collect();
//             let lower_split_means: Vec<f64> = local_mean_lower_sum.iter().map(|x| x / ((dimensions.0 -si) as f64)).collect();
//
//             let upper_split_cv: Vec<f64> = (0..dimensions.1).map(|x| upper_split_variance[x].sqrt()/upper_split_means[x]).collect();
//             let lower_split_cv: Vec<f64> = (0..dimensions.1).map(|x| lower_split_variance[x].sqrt()/lower_split_means[x]).collect();
//
//             let upper_mean_cv: f64 = upper_split_cv.mean();
//             let lower_mean_cv: f64 = lower_split_cv.mean();
//
//             if si % 300 == 0 {
//                 println!("CVs:{},{}", upper_mean_cv,lower_mean_cv);
//             }
//
//             agg_cv.push((upper_mean_cv,lower_mean_cv));
//             cv_mat.push((upper_split_cv,lower_split_cv));
//             svar_mat.push((upper_split_variance,lower_split_variance));
//             sm_mat.push((upper_split_means,lower_split_means));
//
//             // for
//             // node_counts[feature.0].map(|x| )
//         }
//
//         println!("{:?}", sm_mat.iter().cloned().rev().take(20).collect::<Vec<(Vec<f64>,Vec<f64>)>>());
//         println!("{:?}", svar_mat.iter().cloned().rev().take(20).collect::<Vec<(Vec<f64>,Vec<f64>)>>());
//
//         println!("{:?}", agg_cv.iter().cloned().rev().take(20).collect::<Vec<(f64,f64)>>());
//
//         println!("{:?}", &agg_cv[1..agg_cv.len()-1].iter().map(|x| ((x.0 + x.1) as f64) / 2.).collect::<Vec<f64>>());
//
//         argmin(&agg_cv[1..agg_cv.len()-1].iter().map(|x| ((x.0 + x.1) as f64) / 2.).collect())
//
//
//
//         // let result = Vec::new();
//         //
//         //
//         // result
//         // (0,0.)
//     }


fn median_distance(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature:usize, old_median_distance:(usize,f64,usize,f64), old_median:(usize,f64), new_median:(usize,f64), removed: (usize,usize,usize,f64,usize)) -> (usize,f64,usize,f64) {

    println!("Computing new median distance!");

    let mut new_median_distance = old_median_distance.clone();

    let change =  new_median.1 - old_median.1;

    let sample_space = sorted_rank_table[feature].len()-removed.1-1;

    let mut median_distance_ordered = vec![(old_median_distance.0,old_median_distance.1),(old_median_distance.2,old_median_distance.3)];
    median_distance_ordered.sort_unstable_by_key(|x| sorted_rank_table[feature][x.0].2);
    let (mut left_boundary, mut right_boundary) = (sorted_rank_table[feature][median_distance_ordered[0].0,sorted_rank_table[feature][median_distance_ordered[1].0);

    if right_boundary.1 == removed.1 {
        right_boundary = removed;
    }

    if left_boundary.1 == removed.1 {
        left_boundary = removed;
    }


    if change > 0 {
        loop {
            match (right_boundary.3 - new_median.1).abs().partial_cmp(&(left_boundary.3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
                Ordering::Greater => {
                    right_boundary = sorted_rank_table[feature][right_boundary.4];
                    left_boundary = sorted_rank_table[feature][left_boundary.4];
                },
                Ordering::Less => break,
                Ordering::Equal =>{
                    right_boundary = sorted_rank_table[feature][right_boundary.4];
                    left_boundary = sorted_rank_table[feature][left_boundary.4];
                }
            }
        }
    }
    if change < 0 {
        loop {
            match (left_boundary.3 - new_median.1).abs().partial_cmp(&(right_boundary.3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
                Ordering::Greater => {
                    right_boundary = sorted_rank_table[feature][right_boundary.0];
                    left_boundary = sorted_rank_table[feature][left_boundary.0];
                },
                Ordering::Less => break,
                Ordering::Equal =>{
                    right_boundary = sorted_rank_table[feature][right_boundary.0];
                    left_boundary = sorted_rank_table[feature][left_boundary.0];
                }
            }
        }
    }








    let mut median_distance_disordered = vec![(right_boundary.1,(right_boundary.3 - new_median.1).abs()),(left_boundary.1,(left_boundary.3 - new_median.1).abs())];
    median_distance_disordered.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
    new_median_distance = (median_distance_disordered[1].0,median_distance_disordered[1].1, median_distance_disordered[0].0, median_distance_disordered[0].1);

    println!("{}", sample_space);
    println!("{:?}", median_distance_disordered);

    if sample_space % 2 == 0 {

        println!("Even distances, computing split median!");

        let distance_to_outer_left = (sorted_rank_table[feature][left_boundary.0].0, (sorted_rank_table[feature][sorted_rank_table[feature][left_boundary.0].0].3 - new_median.1).abs());

        let distance_to_outer_right = (sorted_rank_table[feature][right_boundary.0].4, (sorted_rank_table[feature][sorted_rank_table[feature][right_boundary.0].4].3 - new_median.1).abs());

        println!("Outer distances: {:?},{:?}", distance_to_outer_left, distance_to_outer_right);

        let temp_outer_median = vec![distance_to_outer_right,distance_to_outer_left];
        let outer_median = temp_outer_median.iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or(&(0,0.)).clone();

        println!("Outer median: {:?}", outer_median);

        new_median_distance.1 = (new_median_distance.1 + outer_median.1)/2.;

    }


    println!("Done computing new median distance!");

    new_median_distance

}



fn median_distance(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature:usize, old_median_distance:(usize,f64,usize,f64), old_median:(usize,f64), new_median:(usize,f64), removed: (usize,usize,usize,f64,usize)) -> (usize,f64,usize,f64) {

    println!("Computing new median distance!");

    let mut new_median_distance = old_median_distance.clone();

    let change =  new_median.1 - old_median.1;

    let sample_space = sorted_rank_table[feature].len()-removed.1-1;

    let mut median_distance_ordered = vec![(old_median_distance.0,old_median_distance.1),(old_median_distance.2,old_median_distance.3)];
    median_distance_ordered.sort_unstable_by_key(|x| sorted_rank_table[feature][x.0].2);
    let (mut left_boundary, mut right_boundary) = (median_distance_ordered[0],median_distance_ordered[1]);


    let mut left_zone_samples: Vec<(usize,f64)> = Vec::new();
    let mut right_zone_samples: Vec<(usize,f64)> = Vec::new();


    for i in 0..10 {

        println!("Old median distance: {:?}", old_median_distance);
        println!("Left boundary: {:?}", left_boundary);
        println!("Change: {}", change);


        match left_boundary.1.partial_cmp(&(old_median_distance.1 - change)).unwrap_or(Ordering::Greater) {
            Ordering::Less => {
                println!("Less (L)");
                left_zone_samples.push(left_boundary);

                if left_boundary.0 == removed.1 {
                    left_boundary = (sorted_rank_table[feature][removed.0].0,sorted_rank_table[feature][sorted_rank_table[feature][removed.0].0].3-old_median.1);
                }
                else {
                    left_boundary = (sorted_rank_table[feature][left_boundary.0].0,sorted_rank_table[feature][sorted_rank_table[feature][left_boundary.0].0].3-old_median.1);
                }
            },
            Ordering::Greater => {println!("Greater and done! (L)"); break},
            Ordering::Equal => {
                println!("Equal (L)");
                left_zone_samples.push(left_boundary);

                if left_boundary.0 == removed.1 {
                    left_boundary = (removed.0,sorted_rank_table[feature][removed.0].3-old_median.1);
                }
                else {
                    left_boundary = (sorted_rank_table[feature][left_boundary.0].0,sorted_rank_table[feature][sorted_rank_table[feature][left_boundary.0].0].3-old_median.1);
                }
            }
        }
    }

    for i in 0..10 {


        println!("Old median distance: {:?}", old_median_distance);
        println!("Right boundary: {:?}", right_boundary);
        println!("Change: {}", change);

        match right_boundary.1.partial_cmp(&(old_median_distance.1 + change)).unwrap_or(Ordering::Greater) {
            Ordering::Less => {
                println!("Less (R)");
                right_zone_samples.push(right_boundary);
                if right_boundary.0 == removed.1 {
                    right_boundary = (removed.4,sorted_rank_table[feature][removed.4].3-old_median.1);
                }
                else {
                    right_boundary = (sorted_rank_table[feature][right_boundary.0].4,sorted_rank_table[feature][sorted_rank_table[feature][right_boundary.0].4].3-old_median.1);
                }
            },
            Ordering::Greater => {println!("Greater and done! (R)") ;break},
            Ordering::Equal => {
                println!("Equal (R)");
                right_zone_samples.push(right_boundary);
                if right_boundary.0 == removed.1 {
                    right_boundary = (removed.4,sorted_rank_table[feature][removed.4].3-old_median.1);
                }
                else {
                    right_boundary = (sorted_rank_table[feature][right_boundary.0].4,sorted_rank_table[feature][sorted_rank_table[feature][right_boundary.0].4].3-old_median.1);
                }
            }
        }
    }

    println!("Zones, (left, right)");
    println!("{:?}",left_zone_samples);
    println!("{:?}",right_zone_samples);



    let mut new_right_boundary = (right_boundary.0,(sorted_rank_table[feature][right_boundary.0].3-new_median.1).abs());

    for (i,sample) in left_zone_samples.iter().enumerate() {

        let mut new_distance_to_left_sample = (sorted_rank_table[feature][sample.0].3 - new_median.1).abs();

        let mut new_distance_to_right_bounadry= (sorted_rank_table[feature][new_right_boundary.0].3 - new_median.1).abs();

        println!("Left zone sample: {:?}", sample);
        println!("New distance to left sample: {}", new_distance_to_left_sample);
        println!("New distance to right boundary: {}", new_distance_to_right_bounadry);

        match new_distance_to_left_sample.partial_cmp(&new_distance_to_right_bounadry).unwrap_or(Ordering::Greater) {
            Ordering::Greater => {
                if new_right_boundary.0 == removed.1 {
                    new_right_boundary = (removed.4, (sorted_rank_table[feature][removed.4].3 - new_median.1).abs());
                }
                else {
                    new_right_boundary = (sorted_rank_table[feature][new_right_boundary.0].4, (sorted_rank_table[feature][sorted_rank_table[feature][new_right_boundary.0].4].3 - new_median.1).abs())
                }
            },
            Ordering::Less => break,
            Ordering::Equal => {
                if new_right_boundary.0 == removed.1 {
                    new_right_boundary = (removed.4, (sorted_rank_table[feature][removed.4].3 - new_median.1).abs());
                }
                else {
                    new_right_boundary = (sorted_rank_table[feature][new_right_boundary.0].4, (sorted_rank_table[feature][sorted_rank_table[feature][new_right_boundary.0].4].3 - new_median.1).abs())
                }
            }

        }


        // match (right_boundary.1 + change).partial_cmp(&sample.1).unwrap_or(Ordering::Greater) {
        //     Ordering::Greater => new_left_boundary = (sorted_rank_table[feature][sample].1
        // }
    }

    let mut new_left_boundary = (left_boundary.0,(sorted_rank_table[feature][left_boundary.0].3-new_median.1).abs());

    for (i,sample) in right_zone_samples.iter().enumerate() {

        let mut new_distance_to_right_sample = (sorted_rank_table[feature][sample.0].3 - new_median.1).abs();

        let mut new_distance_to_left_bounadry= (sorted_rank_table[feature][new_left_boundary.0].3 - new_median.1).abs();

        println!("Right zone sample: {:?}", sample);
        println!("New distance to right sample: {}", new_distance_to_right_sample);
        println!("New distance to left boundary: {}", new_distance_to_left_bounadry);

        match new_distance_to_right_sample.partial_cmp(&new_distance_to_left_bounadry).unwrap_or(Ordering::Greater) {
            Ordering::Greater => {
                if new_left_boundary.0 == removed.1 {
                    new_left_boundary = (removed.0, (sorted_rank_table[feature][removed.0].3 - new_median.1).abs());
                }
                else {
                    new_left_boundary = (sorted_rank_table[feature][new_left_boundary.0].0, (sorted_rank_table[feature][sorted_rank_table[feature][new_left_boundary.0].0].3 - new_median.1).abs())
                }
            },
            Ordering::Less => break,
            Ordering::Equal => {
                if new_left_boundary.0 == removed.1 {
                    new_left_boundary = (removed.0, (sorted_rank_table[feature][removed.0].3 - new_median.1).abs());
                }
                else {
                    new_left_boundary = (sorted_rank_table[feature][new_left_boundary.0].0, (sorted_rank_table[feature][sorted_rank_table[feature][new_left_boundary.0].0].3 - new_median.1).abs())
                }
            }
        }




        // match (right_boundary.1 + change).partial_cmp(&sample.1).unwrap_or(Ordering::Greater) {
        //     Ordering::Greater => new_left_boundary = (sorted_rank_table[feature][sample].1
        // }
    }



        let mut median_distance_disordered = vec![new_left_boundary,new_right_boundary];
        median_distance_disordered.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
        new_median_distance = (median_distance_disordered[0].0,median_distance_disordered[0].1, median_distance_disordered[1].0, median_distance_disordered[1].1);

    println!("{}", sample_space);
    println!("{:?}", median_distance_disordered);

    if sample_space % 2 == 0 {

        println!("Even distances, computing split median!");

        let distance_to_outer_left = (sorted_rank_table[feature][left_boundary.0].0, (sorted_rank_table[feature][sorted_rank_table[feature][left_boundary.0].0].3 - new_median.1).abs());

        let distance_to_outer_right = (sorted_rank_table[feature][right_boundary.0].4, (sorted_rank_table[feature][sorted_rank_table[feature][right_boundary.0].4].3 - new_median.1).abs());

        println!("Outer distances: {:?},{:?}", distance_to_outer_left, distance_to_outer_right);

        let temp_outer_median = vec![distance_to_outer_right,distance_to_outer_left];
        let outer_median = temp_outer_median.iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or(&(0,0.)).clone();

        println!("Outer median: {:?}", outer_median);

        new_median_distance.1 = (new_median_distance.1 + outer_median.1)/2.;

    }


    println!("Done computing new median distance!");

    new_median_distance

}
