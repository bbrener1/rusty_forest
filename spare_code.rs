//
// // fn sort_median(input: &Vec<f64> ) -> (usize,f64,Vec<(usize,f64)>) {
// //
// //     let mut index = 0;
// //     let mut value = 0.;
// //
// //
// //     let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
// //     intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
// //     let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
// //
// //     if intermediate1.len() % 2 == 0 {
// //         index = intermediate1.len()/2;
// //         value = (intermediate1[index].1 + intermediate1[index-1].1) / 2.
// //     }
// //     else {
// //         if intermediate1.len() % 2 == 1 {
// //             index = (intermediate1.len()-1)/2;
// //             value = *intermediate1[index].1;
// //         }
// //         else {
// //             panic!("Median failed!");
// //         }
// //     }
// //
// //     intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
// //     let out = intermediate2.iter().map(|x| (x.0,((x.1).1).clone())).collect();
// //
// //     (index,value,out)
// //
// // }
//
//
// // fn recompute_distances_from_median(input: &mut Vec<(usize,f64)>, index: usize, change: f64 ) {
// //     for i in input {
// //         match (i.0).cmp(&index){
// //             Ordering::Greater => i.1 -= change,
// //             Ordering::Less => i.1 += change,
// //             Ordering::Equal => i.1 = 0.
// //         }
// //     }
// // }
// //     fn find_split(&mut self, feature:usize) -> (usize,f64) {
// //
// //         // let tree: &Tree = self.tree.borrow();
// //
// //         // let counts: &Vec<Vec<f64>> = self.tree.counts.upgrade().expect("Dead tree").borrow();
// //
// //         println!("Splitting!");
// //
// //         let global_counts = self.counts.upgrade().expect("Missing counts?");
// //
// //         let node_counts: Vec<Vec<f64>> = self.indecies.iter().map(|x| self.output_features.iter().map(|y| global_counts[*x][*y]).collect()).collect();
// //         // let node_counts: Vec<Vec<f64>> = self.indecies.iter().map(|x| global_counts[x.clone()]).collect();
// //
// //
// //         // let mut local_variance = self.variance.clone();
// //         // let local_means = self.means.clone();
// //
// //         println!("Node counts cloned: {}, {}", node_counts.len(),node_counts[0].len());
// //
// //
// //         let dimensions: (usize,usize) = (node_counts.len(),node_counts[0].len());
// //
// //         let mut local_mean_upper_sum: Vec<f64> = vec![0f64;dimensions.1];
// //         let mut local_mean_lower_sum: Vec<f64> = inner_axis_sum(&node_counts);
// //
// //
// //         let mut local_variance_upper_sum: Vec<f64> = vec![0f64;dimensions.1];
// //         let mut local_variance_lower_sum: Vec<f64> = inner_axis_variance_sum(&node_counts, None);
// //
// //         let mut feature_vector: Vec<(usize,f64)> = self.indecies.iter().map(|x| global_counts[*x].clone()).enumerate().map(|x| (x.0,x.1[feature])).collect();
// //
// //         feature_vector.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
// //
// //         let mut agg_cv: Vec<(f64,f64)> = Vec::new();
// //         let mut cv_mat: Vec<(Vec<f64>,Vec<f64>)> = Vec::new();
// //         let mut svar_mat: Vec<(Vec<f64>,Vec<f64>)> = Vec::new();
// //         let mut sm_mat: Vec<(Vec<f64>,Vec<f64>)> = Vec::new();
// //
// //         for (si,sample) in feature_vector.iter().map(|x| &node_counts[x.0]).enumerate() {
// //             for (fi,feature) in sample.iter().enumerate() {
// //
// //                 if si % 300 == 0 {
// //                     println!("{:?}", local_mean_upper_sum.iter().take(10).collect::<Vec<&f64>>());
// //                     println!("{:?}", local_mean_lower_sum.iter().take(10).collect::<Vec<&f64>>());
// //                     println!("{:?}", local_variance_upper_sum.iter().take(10).collect::<Vec<&f64>>());
// //                     println!("{:?}", local_variance_lower_sum.iter().take(10).collect::<Vec<&f64>>());
// //                 }
// //
// //                 local_mean_upper_sum[fi] = local_mean_upper_sum[fi] + feature;
// //                 local_mean_lower_sum[fi] = local_mean_lower_sum[fi] - feature;
// //
// // // CHECK INDEXING HERE, BE VERY CAREFUL
// //
// // ////////////////////////////////// CALCULATE YOUR OWN DAMN MEANS //////////////////////////////////////
// // ////////////////////////////// VARIANCE IS SQUARED DIPSHIT //////////////////////////////////////////////
// //
// //                 // Computing running variance has to be done with two sums, one for positively-deviating features and one for negatively-deviating features.
// //
// //
// //
// //                 local_variance_upper_sum[fi] = local_variance_upper_sum[fi] + (local_mean_upper_sum[fi]/((si+1) as f64) - feature).powi(2);
// //                 local_variance_lower_sum[fi] = local_variance_lower_sum[fi] -
// //                 (local_mean_lower_sum[fi]/((dimensions.0 - si) as f64) - feature).powi(2);
// //
// //             }
// //
// //             let upper_split_variance: Vec<f64> = local_variance_upper_sum.iter().map(|x| x / ((si+1) as f64)).collect();
// //             let lower_split_variance: Vec<f64> = local_variance_lower_sum.iter().map(|x| x / ((dimensions.0 - si) as f64)).collect();
// //
// //             let upper_split_means: Vec<f64> = local_mean_upper_sum.iter().map(|x| x / ((si+1) as f64)).collect();
// //             let lower_split_means: Vec<f64> = local_mean_lower_sum.iter().map(|x| x / ((dimensions.0 -si) as f64)).collect();
// //
// //             let upper_split_cv: Vec<f64> = (0..dimensions.1).map(|x| upper_split_variance[x].sqrt()/upper_split_means[x]).collect();
// //             let lower_split_cv: Vec<f64> = (0..dimensions.1).map(|x| lower_split_variance[x].sqrt()/lower_split_means[x]).collect();
// //
// //             let upper_mean_cv: f64 = upper_split_cv.mean();
// //             let lower_mean_cv: f64 = lower_split_cv.mean();
// //
// //             if si % 300 == 0 {
// //                 println!("CVs:{},{}", upper_mean_cv,lower_mean_cv);
// //             }
// //
// //             agg_cv.push((upper_mean_cv,lower_mean_cv));
// //             cv_mat.push((upper_split_cv,lower_split_cv));
// //             svar_mat.push((upper_split_variance,lower_split_variance));
// //             sm_mat.push((upper_split_means,lower_split_means));
// //
// //             // for
// //             // node_counts[feature.0].map(|x| )
// //         }
// //
// //         println!("{:?}", sm_mat.iter().cloned().rev().take(20).collect::<Vec<(Vec<f64>,Vec<f64>)>>());
// //         println!("{:?}", svar_mat.iter().cloned().rev().take(20).collect::<Vec<(Vec<f64>,Vec<f64>)>>());
// //
// //         println!("{:?}", agg_cv.iter().cloned().rev().take(20).collect::<Vec<(f64,f64)>>());
// //
// //         println!("{:?}", &agg_cv[1..agg_cv.len()-1].iter().map(|x| ((x.0 + x.1) as f64) / 2.).collect::<Vec<f64>>());
// //
// //         argmin(&agg_cv[1..agg_cv.len()-1].iter().map(|x| ((x.0 + x.1) as f64) / 2.).collect())
// //
// //
// //
// //         // let result = Vec::new();
// //         //
// //         //
// //         // result
// //         // (0,0.)
// //     }
//
//
// fn median_distance(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature:usize, old_median_distance:(usize,f64,usize,f64), old_median:(usize,f64), new_median:(usize,f64), removed: (usize,usize,usize,f64,usize)) -> (usize,f64,usize,f64) {
//
//     println!("Computing new median distance!");
//
//     let mut new_median_distance = old_median_distance.clone();
//
//     let change =  new_median.1 - old_median.1;
//
//     let sample_space = sorted_rank_table[feature].len()-removed.1-1;
//
//     let mut median_distance_ordered = vec![(old_median_distance.0,old_median_distance.1),(old_median_distance.2,old_median_distance.3)];
//     median_distance_ordered.sort_unstable_by_key(|x| sorted_rank_table[feature][x.0].2);
//     let (mut left_boundary, mut right_boundary) = (sorted_rank_table[feature][median_distance_ordered[0].0,sorted_rank_table[feature][median_distance_ordered[1].0);
//
//     if right_boundary.1 == removed.1 {
//         right_boundary = removed;
//     }
//
//     if left_boundary.1 == removed.1 {
//         left_boundary = removed;
//     }
//
//
//     if change > 0 {
//         loop {
//             match (right_boundary.3 - new_median.1).abs().partial_cmp(&(left_boundary.3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
//                 Ordering::Greater => {
//                     right_boundary = sorted_rank_table[feature][right_boundary.4];
//                     left_boundary = sorted_rank_table[feature][left_boundary.4];
//                 },
//                 Ordering::Less => break,
//                 Ordering::Equal =>{
//                     right_boundary = sorted_rank_table[feature][right_boundary.4];
//                     left_boundary = sorted_rank_table[feature][left_boundary.4];
//                 }
//             }
//         }
//     }
//     if change < 0 {
//         loop {
//             match (left_boundary.3 - new_median.1).abs().partial_cmp(&(right_boundary.3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
//                 Ordering::Greater => {
//                     right_boundary = sorted_rank_table[feature][right_boundary.0];
//                     left_boundary = sorted_rank_table[feature][left_boundary.0];
//                 },
//                 Ordering::Less => break,
//                 Ordering::Equal =>{
//                     right_boundary = sorted_rank_table[feature][right_boundary.0];
//                     left_boundary = sorted_rank_table[feature][left_boundary.0];
//                 }
//             }
//         }
//     }
//
//
//
//
//
//
//
//
//     let mut median_distance_disordered = vec![(right_boundary.1,(right_boundary.3 - new_median.1).abs()),(left_boundary.1,(left_boundary.3 - new_median.1).abs())];
//     median_distance_disordered.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//     new_median_distance = (median_distance_disordered[1].0,median_distance_disordered[1].1, median_distance_disordered[0].0, median_distance_disordered[0].1);
//
//     println!("{}", sample_space);
//     println!("{:?}", median_distance_disordered);
//
//     if sample_space % 2 == 0 {
//
//         println!("Even distances, computing split median!");
//
//         let distance_to_outer_left = (sorted_rank_table[feature][left_boundary.0].0, (sorted_rank_table[feature][sorted_rank_table[feature][left_boundary.0].0].3 - new_median.1).abs());
//
//         let distance_to_outer_right = (sorted_rank_table[feature][right_boundary.0].4, (sorted_rank_table[feature][sorted_rank_table[feature][right_boundary.0].4].3 - new_median.1).abs());
//
//         println!("Outer distances: {:?},{:?}", distance_to_outer_left, distance_to_outer_right);
//
//         let temp_outer_median = vec![distance_to_outer_right,distance_to_outer_left];
//         let outer_median = temp_outer_median.iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or(&(0,0.)).clone();
//
//         println!("Outer median: {:?}", outer_median);
//
//         new_median_distance.1 = (new_median_distance.1 + outer_median.1)/2.;
//
//     }
//
//
//     println!("Done computing new median distance!");
//
//     new_median_distance
//
// }
//
//
//
// fn median_distance(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature:usize, old_median_distance:(usize,f64,usize,f64), old_median:(usize,f64), new_median:(usize,f64), removed: (usize,usize,usize,f64,usize)) -> (usize,f64,usize,f64) {
//
//     println!("Computing new median distance!");
//
//     let mut new_median_distance = old_median_distance.clone();
//
//     let change =  new_median.1 - old_median.1;
//
//     let sample_space = sorted_rank_table[feature].len()-removed.1-1;
//
//     let mut median_distance_ordered = vec![(old_median_distance.0,old_median_distance.1),(old_median_distance.2,old_median_distance.3)];
//     median_distance_ordered.sort_unstable_by_key(|x| sorted_rank_table[feature][x.0].2);
//     let (mut left_boundary, mut right_boundary) = (median_distance_ordered[0],median_distance_ordered[1]);
//
//
//     let mut left_zone_samples: Vec<(usize,f64)> = Vec::new();
//     let mut right_zone_samples: Vec<(usize,f64)> = Vec::new();
//
//
//     for i in 0..10 {
//
//         println!("Old median distance: {:?}", old_median_distance);
//         println!("Left boundary: {:?}", left_boundary);
//         println!("Change: {}", change);
//
//
//         match left_boundary.1.partial_cmp(&(old_median_distance.1 - change)).unwrap_or(Ordering::Greater) {
//             Ordering::Less => {
//                 println!("Less (L)");
//                 left_zone_samples.push(left_boundary);
//
//                 if left_boundary.0 == removed.1 {
//                     left_boundary = (sorted_rank_table[feature][removed.0].0,sorted_rank_table[feature][sorted_rank_table[feature][removed.0].0].3-old_median.1);
//                 }
//                 else {
//                     left_boundary = (sorted_rank_table[feature][left_boundary.0].0,sorted_rank_table[feature][sorted_rank_table[feature][left_boundary.0].0].3-old_median.1);
//                 }
//             },
//             Ordering::Greater => {println!("Greater and done! (L)"); break},
//             Ordering::Equal => {
//                 println!("Equal (L)");
//                 left_zone_samples.push(left_boundary);
//
//                 if left_boundary.0 == removed.1 {
//                     left_boundary = (removed.0,sorted_rank_table[feature][removed.0].3-old_median.1);
//                 }
//                 else {
//                     left_boundary = (sorted_rank_table[feature][left_boundary.0].0,sorted_rank_table[feature][sorted_rank_table[feature][left_boundary.0].0].3-old_median.1);
//                 }
//             }
//         }
//     }
//
//     for i in 0..10 {
//
//
//         println!("Old median distance: {:?}", old_median_distance);
//         println!("Right boundary: {:?}", right_boundary);
//         println!("Change: {}", change);
//
//         match right_boundary.1.partial_cmp(&(old_median_distance.1 + change)).unwrap_or(Ordering::Greater) {
//             Ordering::Less => {
//                 println!("Less (R)");
//                 right_zone_samples.push(right_boundary);
//                 if right_boundary.0 == removed.1 {
//                     right_boundary = (removed.4,sorted_rank_table[feature][removed.4].3-old_median.1);
//                 }
//                 else {
//                     right_boundary = (sorted_rank_table[feature][right_boundary.0].4,sorted_rank_table[feature][sorted_rank_table[feature][right_boundary.0].4].3-old_median.1);
//                 }
//             },
//             Ordering::Greater => {println!("Greater and done! (R)") ;break},
//             Ordering::Equal => {
//                 println!("Equal (R)");
//                 right_zone_samples.push(right_boundary);
//                 if right_boundary.0 == removed.1 {
//                     right_boundary = (removed.4,sorted_rank_table[feature][removed.4].3-old_median.1);
//                 }
//                 else {
//                     right_boundary = (sorted_rank_table[feature][right_boundary.0].4,sorted_rank_table[feature][sorted_rank_table[feature][right_boundary.0].4].3-old_median.1);
//                 }
//             }
//         }
//     }
//
//     println!("Zones, (left, right)");
//     println!("{:?}",left_zone_samples);
//     println!("{:?}",right_zone_samples);
//
//
//
//     let mut new_right_boundary = (right_boundary.0,(sorted_rank_table[feature][right_boundary.0].3-new_median.1).abs());
//
//     for (i,sample) in left_zone_samples.iter().enumerate() {
//
//         let mut new_distance_to_left_sample = (sorted_rank_table[feature][sample.0].3 - new_median.1).abs();
//
//         let mut new_distance_to_right_bounadry= (sorted_rank_table[feature][new_right_boundary.0].3 - new_median.1).abs();
//
//         println!("Left zone sample: {:?}", sample);
//         println!("New distance to left sample: {}", new_distance_to_left_sample);
//         println!("New distance to right boundary: {}", new_distance_to_right_bounadry);
//
//         match new_distance_to_left_sample.partial_cmp(&new_distance_to_right_bounadry).unwrap_or(Ordering::Greater) {
//             Ordering::Greater => {
//                 if new_right_boundary.0 == removed.1 {
//                     new_right_boundary = (removed.4, (sorted_rank_table[feature][removed.4].3 - new_median.1).abs());
//                 }
//                 else {
//                     new_right_boundary = (sorted_rank_table[feature][new_right_boundary.0].4, (sorted_rank_table[feature][sorted_rank_table[feature][new_right_boundary.0].4].3 - new_median.1).abs())
//                 }
//             },
//             Ordering::Less => break,
//             Ordering::Equal => {
//                 if new_right_boundary.0 == removed.1 {
//                     new_right_boundary = (removed.4, (sorted_rank_table[feature][removed.4].3 - new_median.1).abs());
//                 }
//                 else {
//                     new_right_boundary = (sorted_rank_table[feature][new_right_boundary.0].4, (sorted_rank_table[feature][sorted_rank_table[feature][new_right_boundary.0].4].3 - new_median.1).abs())
//                 }
//             }
//
//         }
//
//
//         // match (right_boundary.1 + change).partial_cmp(&sample.1).unwrap_or(Ordering::Greater) {
//         //     Ordering::Greater => new_left_boundary = (sorted_rank_table[feature][sample].1
//         // }
//     }
//
//     let mut new_left_boundary = (left_boundary.0,(sorted_rank_table[feature][left_boundary.0].3-new_median.1).abs());
//
//     for (i,sample) in right_zone_samples.iter().enumerate() {
//
//         let mut new_distance_to_right_sample = (sorted_rank_table[feature][sample.0].3 - new_median.1).abs();
//
//         let mut new_distance_to_left_bounadry= (sorted_rank_table[feature][new_left_boundary.0].3 - new_median.1).abs();
//
//         println!("Right zone sample: {:?}", sample);
//         println!("New distance to right sample: {}", new_distance_to_right_sample);
//         println!("New distance to left boundary: {}", new_distance_to_left_bounadry);
//
//         match new_distance_to_right_sample.partial_cmp(&new_distance_to_left_bounadry).unwrap_or(Ordering::Greater) {
//             Ordering::Greater => {
//                 if new_left_boundary.0 == removed.1 {
//                     new_left_boundary = (removed.0, (sorted_rank_table[feature][removed.0].3 - new_median.1).abs());
//                 }
//                 else {
//                     new_left_boundary = (sorted_rank_table[feature][new_left_boundary.0].0, (sorted_rank_table[feature][sorted_rank_table[feature][new_left_boundary.0].0].3 - new_median.1).abs())
//                 }
//             },
//             Ordering::Less => break,
//             Ordering::Equal => {
//                 if new_left_boundary.0 == removed.1 {
//                     new_left_boundary = (removed.0, (sorted_rank_table[feature][removed.0].3 - new_median.1).abs());
//                 }
//                 else {
//                     new_left_boundary = (sorted_rank_table[feature][new_left_boundary.0].0, (sorted_rank_table[feature][sorted_rank_table[feature][new_left_boundary.0].0].3 - new_median.1).abs())
//                 }
//             }
//         }
//
//
//
//
//         // match (right_boundary.1 + change).partial_cmp(&sample.1).unwrap_or(Ordering::Greater) {
//         //     Ordering::Greater => new_left_boundary = (sorted_rank_table[feature][sample].1
//         // }
//     }
//
//
//
//         let mut median_distance_disordered = vec![new_left_boundary,new_right_boundary];
//         median_distance_disordered.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//         new_median_distance = (median_distance_disordered[0].0,median_distance_disordered[0].1, median_distance_disordered[1].0, median_distance_disordered[1].1);
//
//     println!("{}", sample_space);
//     println!("{:?}", median_distance_disordered);
//
//     if sample_space % 2 == 0 {
//
//         println!("Even distances, computing split median!");
//
//         let distance_to_outer_left = (sorted_rank_table[feature][left_boundary.0].0, (sorted_rank_table[feature][sorted_rank_table[feature][left_boundary.0].0].3 - new_median.1).abs());
//
//         let distance_to_outer_right = (sorted_rank_table[feature][right_boundary.0].4, (sorted_rank_table[feature][sorted_rank_table[feature][right_boundary.0].4].3 - new_median.1).abs());
//
//         println!("Outer distances: {:?},{:?}", distance_to_outer_left, distance_to_outer_right);
//
//         let temp_outer_median = vec![distance_to_outer_right,distance_to_outer_left];
//         let outer_median = temp_outer_median.iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or(&(0,0.)).clone();
//
//         println!("Outer median: {:?}", outer_median);
//
//         new_median_distance.1 = (new_median_distance.1 + outer_median.1)/2.;
//
//     }
//
//
//     println!("Done computing new median distance!");
//
//     new_median_distance
//
// }
//
//
// // let mut distance = Vec::new();
// //
// // println!("Feature size: {}", sorted_rank_table.len());
// //
// // for sample in sorted_rank_table {
// //     distance.push((sample.1,(sample.3-current_median.1).abs()));
// // }
// //
// // distance.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
// //
// // let mut distance_rank_table = Vec::new();
// //
// // for (i,sample) in distance.iter().enumerate() {
// //     if i == 0 {
// //         distance_rank_table.push((0,sample.0,i,sample.1,distance[i+1].0));
// //     }
// //     if i == (distance.len()-1) {
// //         distance_rank_table.push((distance[i-1].0,sample.0,i,sample.1,distance.len()));
// //     }
// //
// //     if {i != 0} && {i < (distance.len()-1)} {
// //         distance_rank_table.push((distance[i-1].0,sample.0,i,sample.1,distance[i+1].0));
// //     }
// // }
// //
// // // let madm = median(&{distance_rank_table.iter().map(|x| x.3).collect()});
// //
// // let median_distance_sample = median(& distance_rank_table.iter().map(|x| x.3).collect());
// //
// // let mut madm = (median_distance_sample.0,median_distance_sample.1,current_median.0,current_median.1);
// //
// // madm.0 = distance_rank_table[madm.0].1;
// //
// // let opposite_madm_rank_steps = (sorted_rank_table[madm.0].2 as i32 - sorted_rank_table[current_median.0].2 as i32).abs();
// //
// // let direction = {
// //     if opposite_madm_rank_steps > 0 {
// //         1
// //     }
// //     else {
// //         if opposite_madm_rank_steps < 0 {
// //             4
// //         }
// //         else {
// //             1
// //         }
// //     }
// // };
// //
// // for i in 0..opposite_madm_rank_steps {
// //     match direction {
// //         4 => madm.2 = sorted_rank_table[madm.2].4,
// //         1 => madm.2 = sorted_rank_table[madm.2].1,
// //         x => madm.2 = sorted_rank_table[madm.2].0
// //     }
// // }
// //
// // madm.3 = sorted_rank_table[madm.2].3 - current_median.1;
// //
// // distance_rank_table.sort_unstable_by(|a,b| (a.1).cmp(&b.1));
// //
// // println!("Initial median distance: {:?}", madm);
// //
// // (distance_rank_table,madm)
//
//
// // fn next(&mut self) -> Option<(Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>)> {
// //
// //     if self.current_index == self.dimensions.0 {
// //         return None
// //     }
// //
// //     for i in 0..self.dimensions.1 {
// //
// //         let current_sample = pop_rank_table(&mut self.upper_sorted_rank_table,(i,self.current_index));
// //
// //         let new_median = rank_table_median(&self.upper_sorted_rank_table, &self.upper_sorted_samples, i, self.current_upper_median[i], current_sample);
// //
// //         let new_median_distance = median_distance(&self.upper_sorted_rank_table, i, self.upper_sorted_MSDM[i], self.current_upper_median[i], new_median, current_sample, &mut self.median_zone[i], &mut self.left_zone[i], &mut self.right_zone[i]);
// //
// //         println!("Current zones:{},{},{}", self.left_zone[i],self.median_zone[i],self.right_zone[i]);
// //
// //         println!("{:?}", current_sample);
// //         println!("{:?}", self.upper_sorted_rank_table);
// //         println!("{:?}", new_median);
// //         println!("{:?}", new_median_distance);
// //
// //         self.current_upper_median[i] = new_median;
// //         self.upper_sorted_MSDM[i] = new_median_distance;
// //
// //         self.current_index += 1;
// //         //
// //         // self.current_upper_median[i] = new_median;
// //
// //         // let where_to = insert_into_sorted(&mut self.lower_sorted_counts[i], (current_sample.1,current_sample.3), current_sample.2);
// //         //
// //         // insert_into_sorted(&mut self.lower_sorted_MSDM[i], (self.lower_sorted_counts[i][where_to].1,self.lower_sorted_counts[i][where_to].2-self.current_lower_median[i].1), self.lower_sorted_counts[i][where_to].0);
// //
// //
// //
// //
// //
// //         // self.lower_sorted_MSDM.insert(where_to,(self.lower_sorted_counts[i].1self.lower_sorted_counts[i][where_to].2 - self.current_lower_median[i].1));
// //
// //     }
// //
// //
// //     Some((self.current_upper_median.clone(),self.upper_sorted_MSDM.clone()))
// // }
//
// // for split in self.madm.take(2) {
// //
// // }
//
//
// // let global_counts = self.counts.upgrade().expect("Missing counts?");
// //
// // let node_counts: Vec<Vec<f64>> = self.indecies.iter().map(|x| self.output_features.iter().map(|y| global_counts[*x][*y]).collect()).collect();
//
// // println!("Testing Online MADM");
// //
// // let mut online = OnlineMADM::new(node_counts);
//
// // let count_subsets = matrix_flip(&matrix_flip(&self.counts.upgrade().unwrap()).iter().cloned().take(6).collect());
//
// // let counts_vectorized: Vec<Vec<f64>> = self.counts.upgrade().unwrap().iter().cloned().collect();
// //
// // let mut count_subsets: Vec<Vec<f64>> = counts_vectorized.iter().cloned().take(5).collect();
// //
// // count_subsets.push(counts_vectorized[6].clone());
// //
// // let mut gold_subset = OnlineMADM::new(count_subsets);
// //
// //
// // println!("Comparing subsetting:");
// // println!("{:?}", gold_subset);
// //
// // let mut subset = online.derive_subset(vec![0,1,2,3,4,6]);
// //
// // println!("{:?}", subset);
// //
// // for i in 0..6 {
// //     println!("{:?}", gold_subset.next());
// //     println!("{:?}", subset.next());
// // }
//
// // online.test();
//
//
// // fn derive_node(&mut self, mut prnt_arc: Arc<Node>, samples: Vec<usize>) {
// //
// //     let pointer_clone = prnt_arc.clone();
// //
// //     let mut parent = prnt_arc.get_mut();
// //
// //     let mut weights = vec![1.;parent.counts.upgrade().expect("Empty counts at node creation!")[0].len()];
// //
// //     let fmadm = parent.madm.0.derive_subset(samples.clone());
// //
// //     let medians = fmadm.median_history[0].clone();
// //     let dispersion = fmadm.dispersion_history[0].iter().enumerate().map(|(i,x)| {
// //         x.1/medians[i].1
// //     }).collect();
// //
// //     let rmadm = parent.madm.1.derive_subset(samples.clone());
// //
// //     let madm = (fmadm,rmadm);
// //
// //     let child = Node{
// //         selfreference: None,
// //         feature: None,
// //         split: None,
// //         output_features: parent.output_features.clone(),
// //         input_features: parent.input_features.clone(),
// //         indecies: samples,
// //         medians: medians,
// //         weights: weights,
// //         dispersion: dispersion,
// //         children: Vec::new(),
// //         parent: Some(pointer_clone),
// //         counts: parent.counts.clone(),
// //         madm: madm
// //     };
// //
// //     parent.children.push(Arc::new(child));
// //
// // }
// //
// // fn first_node(&mut self, counts:Weak<Vec<Vec<f64>>>, samples:Vec<usize>, input_features:Vec<usize>, output_features:Vec<usize> ) {
// //
// //     let out_f_set : HashSet<usize> = output_features.iter().cloned().collect();
// //
// //     let mut weights = vec![1.;counts.upgrade().expect("Empty counts at node creation!")[0].len()];
// //
// //     let mut loc_counts: Vec<Vec<f64>> = samples.iter().cloned().map(|x| counts.upgrade().expect("Dead tree!")[x].clone()).collect();
// //
// //     loc_counts = matrix_flip(&loc_counts).iter().cloned().enumerate().filter(|x| out_f_set.contains(&x.0)).map(|y| y.1).collect();
// //
// //     loc_counts = matrix_flip(&loc_counts);
// //
// //     let fmadm = OnlineMADM::new(loc_counts.clone(),true);
// //
// //     let medians = fmadm.median_history[0].clone();
// //     let dispersion = fmadm.dispersion_history[0].iter().enumerate().map(|(i,x)| {
// //         x.1/medians[i].1
// //     }).collect();
// //
// //     let rmadm = fmadm.reverse();
// //
// //     let madm = (fmadm,rmadm);
// //     // let means: Vec<f64> = loc_counts.iter().fold(vec![0f64;loc_counts[0].len()], |acc, x|
// //     //     {
// //     //         x.iter().zip(acc.iter()).map(|y| y.0 + y.1).collect::<Vec<f64>>()
// //     //     }).iter().map(|z| z/(loc_counts.len() as f64)).collect();
// //
// //     // let variance = loc_counts.iter().fold(vec![0f64;loc_counts[0].len()], |acc,x| {
// //     //         x.iter().enumerate().zip(acc.iter()).map(|y| ((y.0).1 - medians[(y.0).0]).powi(2) + y.1).collect()
// //     //     }
// //     //     ).iter().map(|z| z/(loc_counts.len() as f64)).collect();
// //
// //
// //
// //     let mut result = Node {
// //         selfreference: None,
// //         feature:None,
// //         split: None,
// //         medians: medians,
// //         output_features: output_features,
// //         input_features: input_features,
// //         indecies: samples,
// //         dispersion:dispersion,
// //         weights: weights,
// //         children:Vec::new(),
// //         parent:None,
// //         counts:counts,
// //         madm:madm
// //     };
// //
// //     self.nodes.push(Arc::new(result));
// //
// // }
//
// while let Some(x) = self.madm.1.next() {
//
//     if self.dropout && self.madm.1.sorted_rank_table[feature][x.2].3 == 0.
//     {
//         continue
//     }
//
//     let mut individual_dispersions = Vec::new();
//
//     for (i,(med,disp)) in x.0.iter().zip(x.1.iter()).enumerate() {
//         individual_dispersions.push(((disp.1/med.1)+1.).ln()*self.weights[self.weights.len()-(i+1)]);
//     }
//
//     // println!("{:?}",individual_dispersions);
//
//     reverse_dispersions.push((individual_dispersions.iter().sum::<f64>() / self.weights.iter().sum::<f64>()).exp()-1.);
//
// }
//
// println!("Reverse split found");
//
// reverse_dispersions = reverse_dispersions.iter().cloned().rev().collect();
//
// for (i,(fw,rv)) in forward_dispersions.iter().zip(reverse_dispersions.clone()).enumerate() {
//     if i%100 == 0 {println!("fw/rv: {},{}",fw,rv);}
// }
//
// let mut minimum = (0,f64::INFINITY);
//
// for (i,(fw,rv)) in forward_dispersions.iter().zip(reverse_dispersions.clone()).enumerate() {
//
//     if self.madm.0.sorted_rank_table[feature][drawn_samples[i]].3 == 0. {
//         continue
//     }
//
//     // let proportion = (0.5 - ((i as f64) / (forward_dispersions.len() as f64))).abs();
//     let proportion = (i as f64) / (forward_dispersions.len() as f64);
//     // let proportion = 1.;
//
//     // println!("{}", (fw+rv)*proportion);
//     //
//     // if (0 < i) && (i < (forward_dispersions.len()-1)) {
//     //     if minimum.1 > (fw+rv)*proportion {
//     //         minimum = (i,(fw+rv)*proportion);
//     //     }
//     // }
//
//     let f_adj_disp = fw * (1. - proportion);
//     let r_adj_disp = rv * proportion;
//
//     // println!("{},{}",f_adj_disp,r_adj_disp);
//     // println!("{}", f_adj_disp + r_adj_disp);
//
//     if (0 < i) && (i < (forward_dispersions.len()-1)) {
//         if minimum.1 > f_adj_disp + r_adj_disp {
//             minimum = (drawn_samples[i],f_adj_disp + r_adj_disp);
//         }
//     }
// }
//
// //
// // while let Some(x) = self.madm.1.next() {
// //
// //     if self.dropout && self.madm.1.sorted_rank_table[feature][x.2].3 == 0.
// //     {
// //         continue
// //     }
// //
// //     let mut individual_dispersions = Vec::new();
// //
// //     for (i,(med,disp)) in x.0.iter().zip(x.1.iter()).enumerate() {
// //         individual_dispersions.push(((disp.1/med.1)+1.).ln()*self.weights[self.weights.len()-(i+1)]);
// //     }
// //
// //     // println!("{:?}",individual_dispersions);
// //
// //     reverse_dispersions.push(individual_dispersions);
// //
// // }
// //
// // println!("Reverse split found");
// //
// // reverse_dispersions = reverse_dispersions.iter().cloned().rev().collect();
//
// // for (i,(fw,rv)) in forward_dispersions.iter().zip(reverse_dispersions.clone()).enumerate() {
// //     if i%100 == 0 {println!("fw/rv: {},{}",fw,rv);}
// // }
// //
// // let mut minimum = (0,f64::INFINITY);
// // let mut individual_minima = vec![(0,f64::INFINITY);forward_dispersions[0].len()];
// //
// // for (i,(fw,rv)) in forward_dispersions.iter().zip(reverse_dispersions.clone()).enumerate() {
// //
// //     // if self.madm.0.sorted_rank_table[feature][drawn_samples[i]].3 == 0. {
// //     //     continue
// //     // }
// //
// //     let proportion = (i as f64) / (forward_dispersions.len() as f64);
// //
// //     for (j,(f_el,r_el)) in fw.iter().zip(rv.iter()).enumerate() {
// //
// //         let f_adj_disp = f_el * (1. - proportion);
// //         let r_adj_disp = r_el * proportion;
// //
// //
// //         if individual_minima[j].1 > f_adj_disp + r_adj_disp {
// //             individual_minima[j] = (i,f_adj_disp + r_adj_disp);
// //         }
// //     }
// // }
// //
// // println!("Individual minima computed");
// //
// // // println!("{:?}", individual_minima);
// //
// //     // println!("{},{}",f_adj_disp,r_adj_disp);
// //     // println!("{}", f_adj_disp + r_adj_disp);
// // for i in 0..forward_dispersions.len() {
// //     let mut less = 0;
// //     for element in &individual_minima {
// //         if element.0 < i {
// //             less += 1;
// //         }
// //     }
// //     if less > forward_dispersions.len()/2 {
// //         minimum.0 = i;
// //         minimum.1 = (forward_dispersions[i].mean() + reverse_dispersions[i].mean())/2.;
// //         break
// //     }
// // }
//
// fn find_split(&mut self, feature:usize) -> (usize,f64) {
//
//     println!("Finding split: {}", feature);
//
//     let weight_backup = self.weights[feature];
//     self.weights[feature] = 0.;
//
//     let draw_order = self.madm.0.sort_by_feature(feature);
//     self.madm.1.sort_by_feature(feature);
//     self.madm.1.reverse_draw_order();
//
//     let mut forward_dispersions = Vec::new();
//
//     let mut drawn_samples = Vec::new();
//
//     // println!("{:?}" ,self.madm.0);
//     // println!("{:?}" ,self.madm.1);
//
//     while let Some(x) = self.madm.0.next() {
//
//         if self.dropout && self.madm.1.sorted_rank_table[feature][x.2].3 == 0.
//         {
//             continue
//         }
//
//         let mut individual_dispersions = Vec::new();
//
//         // for (i,(med,disp)) in x.0.iter().zip(x.1.iter()).enumerate() {
//         //     individual_dispersions.push((disp.1/med.1)*self.weights[i]);
//         // }
//         //
//         // forward_dispersions.push(individual_dispersions.iter().sum::<f64>() / self.weights.iter().sum::<f64>());
//
//         for (i,(med,disp)) in x.0.iter().zip(x.1.iter()).enumerate() {
//             individual_dispersions.push((disp.1/med.1)*self.weights[i]);
//         }
//
//         forward_dispersions.push(individual_dispersions.sum() / self.weights.sum());
//
//
//         if forward_dispersions.len()%150 == 0 {
//             println!("{}", forward_dispersions.len());
//         }
//
//         drawn_samples.push(x.2);
//
//     }
//
//     println!("Forward split found");
//
//     let mut reverse_dispersions = Vec::new();
//
//     // println!("{:?}" ,self.madm.1);
//
//     while let Some(x) = self.madm.1.next() {
//
//         if self.dropout && self.madm.1.sorted_rank_table[feature][x.2].3 == 0.
//         {
//             continue
//         }
//
//         let mut individual_dispersions = Vec::new();
//
//         for (i,(med,disp)) in x.0.iter().zip(x.1.iter()).enumerate() {
//             individual_dispersions.push((disp.1/med.1)*self.weights[self.weights.len()-(i+1)]);
//         }
//
//         // println!("{:?}",individual_dispersions);
//
//         reverse_dispersions.push(individual_dispersions.sum() / self.weights.sum());
//
//     }
//
//     println!("Reverse split found");
//
//     reverse_dispersions = reverse_dispersions.iter().cloned().rev().collect();
//
//     // for (i,(fw,rv)) in forward_dispersions.iter().zip(reverse_dispersions.clone()).enumerate() {
//     //     if i%100 == 0 {println!("fw/rv: {},{}",fw,rv);}
//     // }
//
//     let mut minimum = (0,f64::INFINITY, 0,0);
//     // let mut individual_minima = vec![(0,f64::INFINITY);forward_dispersions[0].len()];
//
//     for (i,(fw,rv)) in forward_dispersions.iter().zip(reverse_dispersions.clone()).enumerate() {
//
//         if self.madm.0.sorted_rank_table[feature][drawn_samples[i]].3 == 0. {
//             continue
//         }
//
//         minimum.3 += 1;
//
//         let proportion = (i as f64) / (forward_dispersions.len() as f64);
//
//
//         let f_adj_disp = fw * (1. - proportion);
//         let r_adj_disp = rv * proportion;
//
//         if (0 < i) && (i < (forward_dispersions.len()-1)) {
//             if minimum.1 > f_adj_disp + r_adj_disp {
//                 minimum = (drawn_samples[i],f_adj_disp + r_adj_disp, i, minimum.3);
//             }
//         }
//
//         // for (j,(f_el,r_el)) in fw.iter().zip(rv.iter()).enumerate() {
//         //
//         //     let f_adj_disp = f_el * (1. - proportion);
//         //     let r_adj_disp = r_el * proportion;
//         //
//         //
//         //     if individual_minima[j].1 > f_adj_disp + r_adj_disp {
//         //         individual_minima[j] = (i,f_adj_disp + r_adj_disp);
//         //     }
//         // }
//     }
//
//     // println!("Individual minima computed");
//
//     // println!("{:?}", individual_minima);
//
//         // println!("{},{}",f_adj_disp,r_adj_disp);
//         // println!("{}", f_adj_disp + r_adj_disp);
//     // for i in 0..forward_dispersions.len() {
//     //     let mut less = 0;
//     //     for element in &individual_minima {
//     //         if element.0 < i {
//     //             less += 1;
//     //         }
//     //     }
//     //     if less > forward_dispersions.len()/2 {
//     //         minimum.0 = i;
//     //         minimum.1 = (forward_dispersions[i].mean() + reverse_dispersions[i].mean())/2.;
//     //         break
//     //     }
//     // }
//
//
//     // {
//     //     if (0 < i) && (i < (forward_dispersions.len()-1)) {
//     //         if minimum.1 > f_adj_disp + r_adj_disp {
//     //             minimum = (drawn_samples[i],f_adj_disp + r_adj_disp);
//     //         }
//     //     }
//     // }
//
//     // println!("{:?}", self.madm.0.counts[0]);
//
//     println!("Feature: {}", feature);
//     println!("{:?}", minimum);
//     println!("Split rank: {}, Split value: {}", self.madm.0.sorted_rank_table[feature][minimum.0].2, self.madm.0.sorted_rank_table[feature][minimum.0].3);
//
//
//     self.madm.0.reset();
//     self.madm.1.reset();
//
//     self.weights[feature] = weight_backup;
//
//
//     (minimum.0, minimum.1)
// }
//
// let set_left_split : HashSet<usize> = left_split.iter().cloned().collect();
//
// let mut left_counts = Vec::new();
//
// let loc_counts: Vec<Vec<f64>> = self.counts.upgrade().unwrap().iter().cloned().collect();
//
// for (i,line) in loc_counts.iter().enumerate() {
//     if set_left_split.contains(&i) {
//         left_counts.append(&mut line.clone());
//     }
// }
//
// let set_right_split : HashSet<usize> = right_split.iter().cloned().collect();
//
// let mut right_counts = Vec::new();
//
// for (i,line) in loc_counts.iter().enumerate() {
//     if set_right_split.contains(&i) {
//         right_counts.append(&mut line.clone());
//     }
// }

// impl<'a> Iterator for RightVectCrawler<'a> {
//     type Item = &'a (usize,usize,usize,f64,usize);
//
//     fn next(&mut self) -> Option<&'a (usize,usize,usize,f64,usize)> {
//
//         if self.edge {
//             return None
//         }
//
//         let current = &self.vector.vector[self.index];
//         if self.index == self.vector.right(self.index) {
//             self.edge = true;
//         }
//         self.index = self.vector.right(self.index);
//
//
//         Some(&self.vector.vector[0])
//     }
// }

// impl<'a,'b,'c> RawVectDropSkip<'a,'b,'c> {
//     fn new(input: & RawVectIter<'a,'b>, drop_index : &'c HashSet<usize>) -> RawVectDropSkip<'a,'b,'c> {
//         RawVectDropSkip{draw: input, index:0, drop_set: drop_index}
//     }
// }
//
// impl<'a,'b,'c> Iterator for RawVectDropSkip<'a,'b,'c> {
//     type Item = &'a (usize,usize,usize,f64,usize);
//
//     fn next(&mut self) -> Option<&'a (usize,usize,usize,f64,usize)> {
//
//     }
// }
//
//
// pub struct RawVectDropSkip<'a,'b,'c> {
//     draw: & RawVectIter<'a,'b>,
//     drop_set: &'c HashSet<usize>,
//     index: usize
// }

// impl<'a,'b> RawVectDropSkip<'a,'b> {
//     fn new(input: &'a Vec<(usize,usize,usize,f64,usize)>, drop_index : &'b HashSet<usize>) -> RawVectDropSkip<'a,'b> {
//         RawVectDropSkip{vector: input, index:0, drop_set: drop_index}
//     }
// }
//
// impl<'a,'b> Iterator for RawVectDropSkip<'a,'b> {
//     type Item = &'a (usize,usize,usize,f64,usize);
//
//     fn next(&mut self) -> Option<&'a (usize,usize,usize,f64,usize)> {
//
//         if self.index >= self.vector.len() {
//             return None
//         }
//         loop {
//             self.index += 1;
//             if self.drop_set.contains(&self.index) || self.index > self.vector.len(){
//                 break
//             }
//         }
//         Some(& self.vector[self.index-1])
//     }
// }
//
//
// pub struct RawVectDropSkip<'a,'b> {
//     vector: &'a Vec<(usize,usize,usize,f64,usize)>,
//     drop_set: &'b HashSet<usize>,
//     index: usize
// }

//
// impl<'a,'b> RawVectDropNone<'a,'b> {
//     fn new(input: &'a Vec<(usize,usize,usize,f64,usize)>, drop_index : &'b HashSet<usize>) -> RawVectDropNone<'a,'b> {
//         RawVectDropNone{vector: input, index:0, drop_set: drop_index}
//     }
// }
//
// impl<'a,'b> Iterator for RawVectDropNone<'a,'b> {
//     type Item = (usize,usize,usize,Option<f64>,usize);
//
//     fn next(&mut self) -> Option<(usize,usize,usize,Option<f64>,usize)> {
//
//         if self.index >= self.vector.len(){
//             return None
//         }
//
//         let current = self.vector[self.index].clone();
//
//         let mut result: (usize,usize,usize,Option<f64>,usize);
//
//         if self.drop_set.contains(&self.index) {
//             result = (current.0,current.1,current.2,None,current.4);
//         }
//         else {
//             result = (current.0,current.1,current.2,Some(current.3),current.4);
//         }
//
//         Some(result)
//     }
//
// }
//
// pub struct RawVectDropNone<'a,'b> {
//     vector: &'a Vec<(usize,usize,usize,f64,usize)>,
//     drop_set: &'b HashSet<usize>,
//     index: usize
// }
// impl<'a,'b> RawVectIter<'a,'b> {
//     fn new(input: &'a Vec<(usize,usize,usize,f64,usize)>, draw_order: &'b Vec<usize>) -> RawVectIter<'a,'b> {
//         RawVectIter{vector: input, index:0,draw_order:draw_order}
//     }
// }
//
// impl<'a,'b> Iterator for RawVectIter<'a,'b> {
//     type Item = &'a (usize,usize,usize,f64,usize);
//
//     fn next(&mut self) -> Option<&'a (usize,usize,usize,f64,usize)> {
//
//         self.index += 1;
//         if self.index < self.vector.len() {
//             Some(& self.vector[self.draw_order[self.index-1]])
//         }
//         else {
//             None
//         }
//     }
// }
//
//
// pub struct RawVectIter<'a,'b> {
//     vector: &'a Vec<(usize,usize,usize,f64,usize)>,
//     index: usize,
//     draw_order: &'b Vec<usize>
// }
// impl DeadCenter {
//     pub fn center(raw :&RawVector) -> DeadCenter {
//         let length = raw.len();
//
//         let mut left_zone:i32 = -1;
//         let mut right_zone:i32 = (length +1) as i32;
//
//         let mut left = None;
//         let mut center = None;
//         let mut right = None;
//
//         for sample in raw.left_to_right() {
//
//             println!("Center debug: {},{}", left_zone, right_zone);
//
//             center = right;
//             right = Some(sample.clone());
//             right_zone -= 1;
//
//             if left_zone == right_zone {
//                 break
//             }
//
//             left = center;
//             center = None;
//             left_zone += 1;
//
//             if left_zone == right_zone {
//                 break
//             }
//
//         }
//
//         DeadCenter {
//             left: left,
//             center: center,
//             right: right,
//         }
//     }
//
//     pub fn re_center(&mut self, target: usize, raw_vector: &RawVector) {
//
//         let removed = raw_vector.vector[target].clone();
//
//         if raw_vector.len() < 2 {
//             self.center = Some(raw_vector.first().clone());
//             self.left = None;
//             self.right = None;
//             return
//         }
//
//         if let Some(center) = self.center {
//             match removed.2.cmp(&center.2) {
//                 Ordering::Greater => {
//                     self.right = self.center;
//                     self.center = None;
//                     self.left = Some(raw_vector[raw_vector.left(self.right.unwrap().1).unwrap()]);
//                 },
//                 Ordering::Less => {
//                     self.left = self.center;
//                     self.center = None;
//                     self.right = Some(raw_vector[raw_vector.right(self.left.unwrap().1).unwrap()]);
//                 },
//                 Ordering::Equal => {}
//             }
//         }
//         else {
//             if removed.2 > self.left.unwrap().2 {
//                 self.center = self.left;
//                 self.left = Some(raw_vector[raw_vector.left(self.center.unwrap().1).unwrap()]);
//                 self.right = Some(raw_vector[raw_vector.right(self.center.unwrap().1).unwrap()]);
//             }
//             if removed.2 < self.right.unwrap().2 {
//                 self.center = self.right;
//                 self.left = Some(raw_vector[raw_vector.left(self.center.unwrap().1).unwrap()]);
//                 self.right = Some(raw_vector[raw_vector.right(self.center.unwrap().1).unwrap()]);
//             }
//
//         }
//     }
//
//     pub fn median(&self) -> f64 {
//         if self.center.is_some(){
//             return self.center.unwrap().3
//         }
//         else {
//             return (self.left.unwrap().3 + self.right.unwrap().3)/2.
//         }
//     }
// }
//
// #[derive(Debug)]
// pub struct DeadCenter {
//     left: Option<(usize,usize,usize,f64,usize)>,
//     center: Option<(usize,usize,usize,f64,usize)>,
//     right: Option<(usize,usize,usize,f64,usize)>
// }

// fn initialize(raw_vector:& RawVector, dead_center:DeadCenter) -> MedianZone {
//     let mut leftward = raw_vector.crawl_left(dead_center.left.unwrap().1);
//     let mut rightward = raw_vector.crawl_right(dead_center.right.unwrap().1);
//
//     let mut left = leftward.next().unwrap();
//     let mut right = rightward.next().unwrap();
//
//     let mut outer_left = leftward.cloned().next();
//     let mut outer_right = rightward.cloned().next();
//
//     let median = dead_center.median();
//
//     let mut left_zone = 0;
//     let mut median_zone = 0;
//     let mut right_zone = 0;
//
//     let mut left_set: HashSet<usize> = HashSet::new();
//     let mut middle_set: HashSet<usize> = HashSet::new();
//     let mut right_set: HashSet<usize> = HashSet::new();
//
//     let left_object: LeftZone;
//     let median_object: MedianZone;
//     let right_object: RightZone;
//
//     if left == right {
//         middle_set.insert(left.1);
//         median_zone = 1;
//         median_object = MedianZone{ size:1 ,dead_center:dead_center,left:Some(left.clone()),right:Some(right.clone()), index_set: middle_set};
//     }
//     else {
//         middle_set.insert(left.1);
//         middle_set.insert(right.1);
//         // median_zone = 2;
//         median_object = MedianZone{ size:2 ,dead_center:dead_center,left:Some(left.clone()),right:Some(right.clone()), index_set: middle_set};
//     }
//
//     for sample in leftward {
//         left_set.insert(sample.1);
//         left_zone += 1;
//     }
//     for sample in rightward {
//         right_set.insert(sample.1);
//         right_zone += 1;
//     }
//
//     left_object = LeftZone{size: left_zone, right:outer_left, index_set: left_set};
//     right_object = RightZone{size: right_zone, left: outer_right, index_set: right_set};
//
//     median_object
//
// }
