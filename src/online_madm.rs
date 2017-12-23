use std;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::collections::HashMap;

extern crate rand;
use rand::Rng;


fn median(input: &Vec<f64>) -> (usize,f64) {
    let mut index = 0;
    let mut value = 0.;

    let mut sorted_input = input.clone();
    sorted_input.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

    if sorted_input.len() % 2 == 0 {
        index = sorted_input.len()/2;
        value = (sorted_input[index-1] + sorted_input[index]) / 2.
    }
    else {
        if sorted_input.len() % 2 == 1 {
            index = (sorted_input.len()-1)/2;
            value = sorted_input[index]
        }
        else {
            panic!("Median failed!");
        }
    }
    (index,value)
}

fn sort_upper(input: &Vec<f64>, drop_zeroes: bool) -> (Vec<usize>, Vec<(usize,usize,usize,f64,usize)>,usize,f64) {

    let mut median_index = 0;
    let mut median_value = 0.;

    // eprintln!("Sorting upper, trying to find median");

    // eprintln!("Feature size: {}", input.len());

    let mut sorted_rank_table: Vec<(usize,usize,usize,f64,usize)> = Vec::new();

    let mut intermediate = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));

    // eprintln!("Feature sorted!");


    if drop_zeroes {
        let mut zeroes = 0;
        for sample in intermediate.clone() {
            if *sample.1 == 0. {
                zeroes += 1;
            }
        }
        let mut non_zero_samples = 0;
        let mut previous_sample = 0.;
        let target = intermediate.len()-zeroes;
        for (i,sample) in intermediate.iter().enumerate() {
            if *sample.1 != 0. {
                non_zero_samples += 1;
            }
            if non_zero_samples > (target - (target%2))/2 {
                median_index = i;
                if target%2 == 0 {
                    median_value = (sample.1+previous_sample)/2.;
                }
                else {
                    median_value = *sample.1;
                }
                break
            }
            if *sample.1 != 0. {
                previous_sample = *sample.1;
            }
        }
        // if (intermediate.len() - zero_boundary) % 2 == 0 {
        //     median_index = zeroes + ((intermediate.len()-zeroes)/2);
        //     median_value = (intermediate[median_index].1 + intermediate[median_index-1].1) / 2.
        // }
        // else {
        //     median_index = zeroes + (((intermediate.len()-zeroes)-1)/2);
        //     median_value = *intermediate[median_index].1;
        // }
    }
    else {
        if intermediate.len() % 2 == 0 {
            median_index = intermediate.len()/2;
            median_value = (intermediate[median_index].1 + intermediate[median_index-1].1) / 2.
        }
        else {
            if intermediate.len() % 2 == 1 {
                median_index = (intermediate.len()-1)/2;
                median_value = *intermediate[median_index].1;
            }
            else {
                panic!("Median failed!");
            }
        }
    }

    median_index = intermediate[median_index].0;

    // eprintln!("Median computed! {}", median_value);

    for (i,sample) in intermediate.iter().enumerate() {

        // eprintln!("{}", i);

        if i == 0 {
            sorted_rank_table.push((sample.0,sample.0,i,*sample.1,intermediate[i+1].0));
        }
        if i == (intermediate.len() - 1) {
            sorted_rank_table.push((intermediate[i-1].0,sample.0,i,*sample.1,sample.0));
        }
        if {i != 0} && {i < (intermediate.len()-1)} {
            sorted_rank_table.push((intermediate[i-1].0,sample.0,i,*sample.1,intermediate[i+1].0));
        }
    }

    // eprintln!("Sorting back into correct order!");

    sorted_rank_table.sort_unstable_by(|a,b| a.1.cmp(&b.1));

    let order_of_samples = intermediate.iter().map(|x| x.0).collect();

    // eprintln!("Returning ranking table:");

    (order_of_samples,sorted_rank_table,median_index,median_value)
}

// fn madm_ranking(sorted_rank_table: &Vec<(usize,usize,usize,f64,usize)>, sorted_samples: &Vec<usize>, current_median:&(usize,f64)) -> (Vec<(usize,usize,usize,f64,usize)>,(usize,f64,usize,f64)) {

fn madm_ranking(sorted_rank_table: &Vec<Vec<(usize,usize,usize,f64,usize)>>,feature:usize, current_median:&(usize,f64),drop_zeroes:bool) -> ((usize,f64,usize,f64),(usize,usize,usize)) {

    // eprintln!("Computing MADM rankings!");

    let mut new_median_distance = (0,0.,0,0.);


    let mut left_sample = sorted_rank_table[feature][sorted_rank_table[feature][current_median.0].0];
    let mut right_sample = sorted_rank_table[feature][sorted_rank_table[feature][current_median.0].4];


    let mut size = sorted_rank_table[feature].len();
    let mut zeroes = 0;

    if drop_zeroes{
        for sample in sorted_rank_table[feature].iter().cloned() {
            if sample.3 == 0. {
                size -=1;
                zeroes += 1;
            }
        }
    }

    let mut current_samples = 1;

    let target_samples = ((size as f64 / 2.).trunc() + 1.) as i32;

    let mut closest_left_sample = sorted_rank_table[feature][current_median.0];
    let mut closest_right_sample = sorted_rank_table[feature][current_median.0];

    // eprintln!("Target samples: {}", target_samples);

    loop {

        if current_samples >= target_samples {
            break
        }

        match (left_sample.3 - current_median.1).abs().partial_cmp(&(right_sample.3 - current_median.1).abs()).unwrap_or(Ordering::Greater) {
            Ordering::Less => {closest_left_sample = left_sample; left_sample = sorted_rank_table[feature][left_sample.0]},
            Ordering::Greater => {closest_right_sample = right_sample; right_sample = sorted_rank_table[feature][right_sample.4]},
            Ordering::Equal => {closest_right_sample = right_sample; right_sample = sorted_rank_table[feature][right_sample.4]}
        }

        current_samples += 1;

    }

    if size%2 == 0 {
        let mut closer = closer_sample(closest_left_sample, closest_right_sample, *current_median, feature, sorted_rank_table);
        closer.reverse();

        // eprintln!("Even samples: {:?}", closer);

        new_median_distance.0 = closer[0].1;
        new_median_distance.1 = (((closer[0].3 - current_median.1).abs() + (closer[1].3 - current_median.1).abs())  / 2.).abs();

        let opposite_ranking = vec![closest_left_sample,closest_right_sample];
        let opposite_closest = opposite_ranking.iter().min_by(|a,b| (a.3 - current_median.1).abs().partial_cmp(&(b.3 - current_median.1).abs()).unwrap_or(Ordering::Greater)).unwrap();

        new_median_distance.2 = opposite_closest.1;
        new_median_distance.3 = (opposite_closest.3 - current_median.1).abs()

    }
    else {

        let mut closer = closer_sample(closest_left_sample, closest_right_sample, *current_median, feature, sorted_rank_table);
        closer.reverse();

        // eprintln!("Odd samples: {:?}", closer);

        new_median_distance.0 = closer[0].1;
        new_median_distance.1 = (closer[0].3 - current_median.1).abs();

        let opposite_ranking = vec![closest_left_sample,closest_right_sample];
        let opposite_closest = opposite_ranking.iter().min_by(|a,b| (a.3 - current_median.1).abs().partial_cmp(&(b.3 - current_median.1).abs()).unwrap_or(Ordering::Greater)).unwrap();

        new_median_distance.2 = opposite_closest.1;
        new_median_distance.3 = (opposite_closest.3 - current_median.1).abs()

    }

    let mut median_zone = current_samples as usize;
    let mut left_zone = 0;
    let mut right_zone = 0;

    // eprintln!("Computing new zones:{},{},{}",median_zone,left_zone,right_zone);
    // eprintln!("{:?},{:?}",closest_left_sample,closest_right_sample);

    loop {

        // eprintln!("Computing new zones:{},{},{}",median_zone,left_zone,right_zone);
        // eprintln!("{:?},{:?}",closest_left_sample,closest_right_sample);

        if closest_left_sample.0 == closest_left_sample.1 {
            break
        }
        else {
            closest_left_sample = sorted_rank_table[feature][closest_left_sample.0];
            left_zone += 1
        }
    }

    loop {

        // eprintln!("Computing new zones:{},{},{}",median_zone,left_zone,right_zone);
        // eprintln!("{:?},{:?}",closest_left_sample,closest_right_sample);

        if closest_right_sample.4 == closest_right_sample.1 {
            break
        }
        else {
            closest_right_sample = sorted_rank_table[feature][closest_right_sample.4];
            right_zone += 1
        }
    }

    // eprintln!("Computed new median distance: {:?}", new_median_distance);
    //
    // new_median_distance



    (new_median_distance,(left_zone,median_zone,right_zone))



}


fn insert_into_sorted(sorted: &mut Vec<(usize,usize,f64)> , insertion: (usize,f64), old_index: usize) -> usize {
    let index = sorted.binary_search_by(|x| (x.2).partial_cmp(&insertion.1).unwrap_or(Ordering::Greater));
    let mut returned_index = 0;
    match index {
        Ok(index) => {sorted.insert(index, (old_index, insertion.0, insertion.1)); returned_index = index},
        Err(index) => {sorted.insert(index, (old_index, insertion.0, insertion.1)); returned_index = index}
    }
    returned_index
}

fn pop_rank_table(rank_table: &mut Vec<Vec<(usize,usize,usize,f64,usize)>>, pop: (usize,usize)) -> (usize,usize,usize,f64,usize) {

    // eprintln!("Popping a sample!");

    // assert!(rank_table[pop.0].len() > 1, "Popped an empty feature!");

    // let target = rank_table[pop.0].remove(pop.1);
    let target = rank_table[pop.0][pop.1];

    let mut not_edge = true;

    let mut greater_than = target.0;
    if target.2 != rank_table[pop.0][target.0].2 {
        greater_than = target.0;
    }
    else {
        greater_than = target.4;
        not_edge = false;
    }

    let mut less_than = target.4;
    if rank_table[pop.0][target.4].2 != target.2 {
        less_than = target.4;
    }
    else {
        less_than = target.0;
        not_edge = false;
    }

    if not_edge {
        rank_table[pop.0][greater_than].4 = less_than;
        rank_table[pop.0][less_than].0 = greater_than;
    }
    else {
        // eprintln!("Popping an edge: {},{}", greater_than, less_than);
        if target.4 == target.1 {
            rank_table[pop.0][target.0].4 = rank_table[pop.0][target.0].1;
        }
        if target.0 == target.1 {
            rank_table[pop.0][target.4].0 = rank_table[pop.0][target.4].1;
        }
    }

    rank_table[pop.0][pop.1].0 = target.1;
    rank_table[pop.0][pop.1].4 = target.1;

    // eprintln!("Sample popped: {:?}", target);

    target

}

fn rank_table_median(rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, right_zone: &usize, median_zone:&usize, left_zone:&usize, feature:usize, old_median:(usize,f64), removed: (usize,usize,usize,f64,usize)) -> (usize,f64) {

    // eprintln!("Computing new median!");

    let mut new_median: (usize,f64) = (0,0.);

    if (right_zone+left_zone+median_zone) < 3 {
        return (removed.4, rank_table[feature][removed.4].3)
    }
////////###### Warning table size hack, check against current index may be better!

    // if (right_zone+left_zone+median_zone -1 ) % 2 != 0 {
    if (rank_table[feature].len() - removed.1) %2 != 0 {

        // eprintln!("Even median!");

        match (rank_table[feature][old_median.0].2).cmp(&removed.2) {
            Ordering::Less => new_median = {
                let new_index = old_median.0;
                let left_index = rank_table[feature][new_index].0;
                let new_value = (rank_table[feature][new_index].3 + rank_table[feature][left_index].3) / 2.;
                (new_index,new_value)
            },
            Ordering::Greater => new_median = {
                let new_index = rank_table[feature][old_median.0].4;
                let left_index = rank_table[feature][new_index].0;
                let new_value = (rank_table[feature][new_index].3 + rank_table[feature][left_index].3)/2.;
                (new_index,new_value)
            },
            Ordering::Equal => new_median = {
                let new_index = removed.4;
                let left_index = removed.0;
                let new_value = (rank_table[feature][new_index].3 + rank_table[feature][left_index].3)/2.;
                (new_index,new_value)
            }
        }
    }
    else {

        // eprintln!("Odd median!");
        match (rank_table[feature][old_median.0].2).partial_cmp(&removed.2).unwrap_or(Ordering::Greater) {
            Ordering::Less => new_median = {
                let new_index = rank_table[feature][old_median.0].0;
                let new_value = rank_table[feature][new_index].3;
                (new_index,new_value)
            },
            Ordering::Greater => new_median = {
                let new_index = old_median.0;
                let new_value = rank_table[feature][new_index].3;
                (new_index,new_value)
            },
            Ordering::Equal => new_median = {
                let new_index = removed.0;
                let new_value = rank_table[feature][new_index].3;
                (new_index,new_value)

            }
        }
    }

    // eprintln!("Computed new median!");
    // eprintln!("{:?}", new_median);

    new_median

}

fn closer_sample(left: (usize,usize,usize,f64,usize), right: (usize,usize,usize,f64,usize), median: (usize, f64), feature: usize, sorted_rank_table: & Vec<Vec<(usize,usize,usize,f64,usize)>>) -> Vec<(usize,usize,usize,f64,usize)> {

    // eprintln!("Closer sample debug: {:?},{:?}",left,right);

    let inner_left = sorted_rank_table[feature][left.4];
    let inner_right = sorted_rank_table[feature][right.0];

    // eprintln!("Closer sample debug: {:?},{:?}", inner_left,inner_right);

    let mut possibilities = vec![inner_left,inner_right,left,right];

    possibilities.sort_by(|a,b| (a.3 - median.1).abs().partial_cmp(&(b.3 - median.1).abs()).unwrap_or(Ordering::Greater));



    possibilities
}

fn expand_by_1(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature:usize, left: (usize,usize,usize,f64,usize) ,right: (usize,usize,usize,f64,usize), right_zone: &mut usize, median_zone: &mut usize, left_zone: &mut usize, old_median: (usize,f64)) -> ((usize,usize,usize,f64,usize),(usize,usize,usize,f64,usize)) {

    let mut left_boundary = left.clone();
    let mut right_boundary = right.clone();

    let outer_right = sorted_rank_table[feature][right_boundary.4];
    let outer_left = sorted_rank_table[feature][left_boundary.0];

    if (right_boundary.1 == right_boundary.4) || (left_boundary.1 == left_boundary.0) {
        if left_boundary.1 == left_boundary.0 {
            right_boundary = outer_right;
            *right_zone -= 1;
            *median_zone += 1;
        }
        if right_boundary.1 == right_boundary.4 {
            left_boundary = outer_left;
            *left_zone -= 1;
            *median_zone += 1;
        }
    }

    else {
        if (outer_right.3 - old_median.1).abs() > (outer_left.3 - old_median.1).abs() {
            left_boundary = outer_left;
            *left_zone -= 1;
            *median_zone += 1;

        }
        else {
            right_boundary = outer_right;
            *right_zone -= 1;
            *median_zone += 1;
        }
    }

    // eprintln!("Expanding by 1:");
    // eprintln!("{:?},{:?}",left,right);
    // eprintln!("{:?},{:?}", left_boundary,right_boundary);

    (left_boundary,right_boundary)

}

fn contract_by_1 (sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature:usize, left: (usize,usize,usize,f64,usize) ,right: (usize,usize,usize,f64,usize), right_zone: &mut usize, median_zone: &mut usize, left_zone: &mut usize, old_median: (usize,f64)) -> ((usize,usize,usize,f64,usize),(usize,usize,usize,f64,usize)) {

    let mut left_boundary = left.clone();
    let mut right_boundary = right.clone();


    if (right_boundary.3 - old_median.1).abs() > (left_boundary.3 - old_median.1).abs() {
        right_boundary = sorted_rank_table[feature][right_boundary.0];
        *right_zone += 1;
        *median_zone -= 1;
    }
    else {
        left_boundary = sorted_rank_table[feature][left_boundary.4];
        *left_zone += 1;
        *median_zone -=1;
    }

    (left_boundary,right_boundary)
}

fn small_median_distance(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature: usize ,new_median:(usize,f64)) -> (usize,f64,usize,f64) {
    let median_entry = sorted_rank_table[feature][new_median.0];
    if median_entry.1 == median_entry.4 && median_entry.1 != median_entry.0 {
        let distance_value = (median_entry.3 - sorted_rank_table[feature][median_entry.0].3).abs()/2.;
        return (median_entry.0,distance_value,median_entry.1,distance_value)
    }
    else if median_entry.1 == median_entry.0 && median_entry.1 != median_entry.4 {
        let distance_value = (median_entry.3 - sorted_rank_table[feature][median_entry.4].3).abs()/2.;
        return (median_entry.1,distance_value,median_entry.4,distance_value)
    }
    else {
        return (median_entry.1,0.,median_entry.1,0.)
    }
}

fn median_distance(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature:usize, old_median_distance:(usize,f64,usize,f64), old_median:(usize,f64), new_median:(usize,f64), removed: (usize,usize,usize,f64,usize), median_zone: &mut usize,left_zone: &mut usize, right_zone: &mut usize) -> (usize,f64,usize,f64) {

    if (*left_zone + *median_zone + *right_zone) < 4 {
        return small_median_distance(sorted_rank_table,feature,new_median)
    }


    // eprintln!("Computing new median distance!");

    let mut new_median_distance = old_median_distance.clone();

    let change =  new_median.1 - old_median.1;

    let mut median_distance_ordered = vec![(old_median_distance.0,old_median_distance.1),(old_median_distance.2,old_median_distance.3)];
    median_distance_ordered.sort_unstable_by_key(|x| sorted_rank_table[feature][x.0].2);
    let (mut left_boundary, mut right_boundary) =
    (sorted_rank_table[feature][median_distance_ordered[0].0],sorted_rank_table[feature][median_distance_ordered[1].0]);

    // eprintln!("Before recomputation:");
    // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    // eprintln!("Removed: {:?}", removed);
    // eprintln!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);


    if removed.2 < left_boundary.2 {
        *left_zone -= 1;
    }
    else if removed.2 > right_boundary.2 {
        *right_zone -= 1;
    }
    else if left_boundary.2 <= removed.2 && removed.2 <= right_boundary.2 {
        *median_zone -= 1;
    }


    // eprintln!("Subtracted removed sample!");
    //
    // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    // eprintln!("Removed: {:?}", removed);
    // eprintln!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);


    if removed.1 == left_boundary.1 {
        left_boundary = sorted_rank_table[feature][removed.4];
    }

    if removed.1 == right_boundary.1 {
        right_boundary = sorted_rank_table[feature][removed.0];
    }
    //
    // eprintln!("Handled removed boundary");
    //
    // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    // eprintln!("Removed: {:?}", removed);
    // eprintln!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);



    if (*median_zone as i32 - 2) > (*left_zone + *right_zone) as i32 {
        let contracted = contract_by_1(sorted_rank_table, feature, left_boundary, right_boundary, right_zone, median_zone, left_zone, old_median);
        left_boundary = contracted.0;
        right_boundary = contracted.1;
    }

    // eprintln!("Handled median overflow");
    //
    // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    // eprintln!("Removed: {:?}", removed);
    // eprintln!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);


    if (*median_zone as i32) <= ((*left_zone+*right_zone) as i32) {
        let expanded = expand_by_1(sorted_rank_table, feature, left_boundary, right_boundary, right_zone, median_zone, left_zone, old_median);
        left_boundary = expanded.0;
        right_boundary = expanded.1;
    }


    // eprintln!("Zones fixed:");
    // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    // eprintln!("Removed: {:?}", removed);
    // eprintln!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);

    if change > 0. {
        // eprintln!("Moving right!");
        for i in 0..sorted_rank_table[feature].len() {
            // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            // eprintln!("New median: {:?}", new_median);
            if right_boundary.1 == right_boundary.4 {
                break
            }

            // eprintln!("Comparison: {},{}",(sorted_rank_table[feature][right_boundary.4].3 - new_median.1).abs(),(sorted_rank_table[feature][left_boundary.4].3 - new_median.1).abs());

            match (sorted_rank_table[feature][right_boundary.4].3 - new_median.1).abs().partial_cmp(&(left_boundary.3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
                Ordering::Less => {
                    right_boundary = sorted_rank_table[feature][right_boundary.4];
                    left_boundary = sorted_rank_table[feature][left_boundary.4];
                    // eprintln!("Moved right!")
                },
                Ordering::Greater => break,
                Ordering::Equal =>{
                    right_boundary = sorted_rank_table[feature][right_boundary.4];
                    left_boundary = sorted_rank_table[feature][left_boundary.4];
                }
            }
            *left_zone += 1;
            *right_zone -= 1;
            // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
        }
    }
    if change < 0. {
        // eprintln!("Moving left!");
        for i in 0..sorted_rank_table[feature].len() {
            // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            // eprintln!("New median: {:?}", new_median);
            if left_boundary.1 == left_boundary.0 {
                break
            }
            match (sorted_rank_table[feature][left_boundary.0].3 - new_median.1).abs().partial_cmp(&(right_boundary.3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
                Ordering::Less => {
                    right_boundary = sorted_rank_table[feature][right_boundary.0];
                    left_boundary = sorted_rank_table[feature][left_boundary.0];
                },
                Ordering::Greater => break,
                Ordering::Equal =>{
                    right_boundary = sorted_rank_table[feature][right_boundary.0];
                    left_boundary = sorted_rank_table[feature][left_boundary.0];
                }
            }
            *right_zone += 1;
            *left_zone -= 1;
            // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
        }
    }


    let mut median_distance_disordered = vec![(right_boundary.1,(right_boundary.3 - new_median.1).abs()),(left_boundary.1,(left_boundary.3 - new_median.1).abs())];
    median_distance_disordered.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
    new_median_distance = (median_distance_disordered[1].0,median_distance_disordered[1].1, median_distance_disordered[0].0, median_distance_disordered[0].1);

    let sample_space = *median_zone+*left_zone+*right_zone;

    // eprintln!("{}", sample_space);
    // eprintln!("{:?}", median_distance_disordered);

    if sample_space % 2 == 0 {

        // eprintln!("Even samples, computing split median!");

        let mut distances = closer_sample(left_boundary, right_boundary, new_median, feature, sorted_rank_table);
        distances.reverse();

        // eprintln!("{:?}",distances);

        new_median_distance.1 = (new_median_distance.1 + (distances[1].3 - new_median.1).abs())/2.;

        // eprintln!("Even distances, computing split median!");
        //
        // let distance_to_outer_left = (left_boundary.0, (sorted_rank_table[feature][left_boundary.0].3 - new_median.1).abs());
        //
        // let distance_to_outer_right = (right_boundary.4, (sorted_rank_table[feature][right_boundary.4].3 - new_median.1).abs());
        //
        // eprintln!("Outer distances: {:?},{:?}", distance_to_outer_left, distance_to_outer_right);
        //
        // let temp_outer_median = vec![distance_to_outer_right,distance_to_outer_left];
        // let outer_median = temp_outer_median.iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or(&(0,0.)).clone();
        //
        // eprintln!("Outer median: {:?}", outer_median);
        //
        // new_median_distance.1 = (new_median_distance.1 + outer_median.1)/2.;

    }


    // eprintln!("Done computing new median distance!");
    // eprintln!("{:?}", new_median_distance);

    new_median_distance

}

fn rank_table_subset<T:std::clone::Clone>(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, upper_sorted_samples: & Vec<Vec<usize>>, indecies: & Vec<usize>, sample_identity: & Vec<T>, drop_zeroes: bool) -> (Vec<Vec<(usize,usize,usize,f64,usize)>>,Vec<(usize,f64)>,Vec<T>,Vec<Vec<usize>>) {

    eprintln!("Rank table: {},{}", sorted_rank_table.len(),sorted_rank_table[0].len());

    let sample_set: HashSet<usize> = indecies.iter().cloned().collect();

    eprintln!("Built a sample hash");

    let mut index_dictionary: HashMap<usize,usize> = HashMap::new();

    let mut new_sample_ids: Vec<T> = Vec::new();

    let mut new_indecies = 0;

    for (i,sample) in sorted_rank_table[0].iter().enumerate() {
        if sample_set.contains(&i) {
            index_dictionary.insert(i,new_indecies);
            new_sample_ids.push(sample_identity[i].clone());
            new_indecies += 1;

        }


    }

    eprintln!("Built the sample index dictionary");

    // eprintln!("{:?}", sample_set);
    // eprintln!("{:?}", index_dictionary);

    let mut new_sorted_samples = Vec::new();

    let mut new_rank_table: Vec<Vec<(usize,usize,usize,f64,usize)>> = Vec::new();

    let mut new_medians = Vec::new();

    for (i,feature) in sorted_rank_table.iter().enumerate() {

        new_rank_table.push(Vec::new());

        let mut current_feature = new_rank_table.last_mut().unwrap();

        let mut first_sample:usize = sample_set.iter().next().unwrap().clone();

        for (j,sample) in feature.iter().enumerate() {

            // eprintln!("i: {}, j: {}", i,j);

            if sample_set.contains(&sample.1) {

                let mut current_sample = sample.1;

                // eprintln!("Current sample: {}", current_sample);
                // eprintln!("{:?}", sorted_rank_table[i][current_sample]);
                //
                // eprintln!("Finding previous sample:");

                let mut previous_sample = sample.0;

                for k in 0..feature.len()-1 {
                    if sample_set.contains(&previous_sample) {
                        break;
                    }
                    else {
                        previous_sample = sorted_rank_table[i][previous_sample].0;
                    }
                    if previous_sample == sorted_rank_table[i][previous_sample].0 {
                        if sample_set.contains(&previous_sample) {
                            break;
                        }
                        else {
                            previous_sample = current_sample;
                            break;

                        }
                    }

                }

                // eprintln!("Previous sample: {}", previous_sample);


                let mut next_sample = sample.4;

                for k in 0..feature.len()-1 {
                    if sample_set.contains(&next_sample) {
                        break;
                    }
                    else {
                        next_sample = sorted_rank_table[i][next_sample].4;
                    }
                    if next_sample == sorted_rank_table[i][next_sample].4 {
                        if sample_set.contains(&next_sample) {
                            break
                        }
                        else {
                            next_sample = current_sample;
                            break;
                        }
                    }
                }

                if drop_zeroes {

                    if sorted_rank_table[i][current_sample].3 < sorted_rank_table[i][first_sample].3 && sorted_rank_table[i][current_sample].3 != 0. {
                        first_sample = current_sample;
                        // eprintln!("Updated first sample");
                        // eprintln!("{:?}", sample);
                    }
                }
                else {
                    if sorted_rank_table[i][current_sample].3 < sorted_rank_table[i][first_sample].3 {
                        first_sample = current_sample;
                        // eprintln!("Updated first sample");
                        // eprintln!("{:?}", sample);
                    }
                }


                current_sample = index_dictionary[&sample.1];
                previous_sample = index_dictionary[&previous_sample];
                next_sample = index_dictionary[&next_sample];

                // eprintln!("Next sample: {}", next_sample);



                current_feature.push((previous_sample,current_sample,0,sample.3,next_sample));


            }

        }

        println!("Derived, but is the first sample correct?");
        println!("{:?}", first_sample);

        first_sample = index_dictionary[&first_sample];

        eprintln!("Ranking derived rank table");
        eprintln!("First sample: {}", first_sample);

        let mut current_sorted_samples = Vec::new();

        let mut current_index = first_sample.clone();
        let mut current_sample = 0;

        let mut zeroes = 0;

        if drop_zeroes {
            for sample in current_feature.iter_mut() {
                if sample.3 == 0. {
                    sample.2 = current_sample;
                    current_sample += 1;
                    zeroes += 1;
                }
            }
        }

        for i in 0..current_feature.len() {

            // eprintln!("Before rank: {:?}", current_feature[current_index]);

            current_feature[current_index].2 = current_sample;

            // eprintln!("Index: {:?}", current_index);
            // eprintln!("Sample: {:?}",current_sample);
            // eprintln!("{:?}", current_feature[current_index].2);

            current_sorted_samples.push(current_index);

            if (current_feature.len()-zeroes)%2 == 0 {
                if current_sample == (current_feature.len()-zeroes)/2 {
                    new_medians.push((current_index,(current_feature[current_index].3 + current_feature[current_sorted_samples[current_sorted_samples.len()-2]].3)/2.))
                }
            }
            else {
                if current_sample == ((current_feature.len()-zeroes)-1)/2 {
                    new_medians.push((current_index,current_feature[current_index].3))
                }
            }

            current_sample += 1;

            if current_index == current_feature[current_index].4 {
                break
            }

            current_index = current_feature[current_index].4;


        }

        if i%400 == 0 {
            eprintln!("Finished subsetting {}",i);
        }

        new_sorted_samples.push(current_sorted_samples);

    }


    eprintln!("{:?}",new_rank_table);

    (new_rank_table,new_medians,new_sample_ids,new_sorted_samples)

}


#[allow(dead_code)]
impl OnlineMADM {

    pub fn new(counts: Vec<Vec<f64>>,drop_zeroes:bool) -> OnlineMADM {

        eprintln!("Computing MADM initial conditions");

        let mut local_counts = counts;
        let dimensions = (local_counts.len(),local_counts[0].len());
        let mut upper_sorted_rank_table = Vec::new();
        let mut upper_sorted_samples = Vec::new();
        // for feature in matrix_flip(&local_counts) {
        //     upper_sorted_counts.push(argsort(&feature));
        // }
        let mut current_upper_median = Vec::new();
        for (i,feature) in matrix_flip(&local_counts).iter().enumerate() {
            // eprintln!("Computing feature median");
            if i%100 == 0 {
                eprintln!("Feature median, feature: {}", i);
            }
            let description = sort_upper(&feature,drop_zeroes);
            // current_upper_median.push(median(&feature.iter().map(|x| x.1).collect()));
            upper_sorted_samples.push(description.0);
            upper_sorted_rank_table.push(description.1);
            current_upper_median.push((description.2,description.3));
        }
        if drop_zeroes {
            for (i,feature) in upper_sorted_rank_table.clone().iter().enumerate() {
                for (j, sample) in feature.iter().enumerate() {
                    if sample.3 == 0. {
                        pop_rank_table(&mut upper_sorted_rank_table, (i,j));
                    }
                }
            }
        }
        let mut upper_sorted_MSDM = Vec::new();

        let mut median_zone = Vec::new();
        let mut left_zone = Vec::new();
        let mut right_zone = Vec::new();

        for (i,(feature,median)) in upper_sorted_rank_table.iter().zip(current_upper_median.iter()).enumerate() {

            // eprintln!("Sorted rank table: {:?}", upper_sorted_rank_table[i]);

            let (madm,zones) = madm_ranking(&upper_sorted_rank_table,i, &current_upper_median[i],drop_zeroes);
            upper_sorted_MSDM.push(madm);

            if i%100 == 0 {
                eprintln!("Rankings computed for feature {}", i);
            }

            let boundaries = vec![(madm.0,madm.1),(madm.2,madm.3)];

            let left_boundary = boundaries.iter().min_by(|a,b| upper_sorted_rank_table[i][a.0].2.cmp(& upper_sorted_rank_table[i][b.0].2)).unwrap();
            let right_boundary = boundaries.iter().max_by(|a,b| upper_sorted_rank_table[i][a.0].2.cmp(& upper_sorted_rank_table[i][b.0].2)).unwrap();

            left_zone.push(zones.0);
            median_zone.push(zones.1);
            right_zone.push(zones.2);
            // median_zone.push(upper_sorted_rank_table[i][right_boundary.0].2 - upper_sorted_rank_table[i][left_boundary.0].2 + 1);
            // right_zone.push(upper_sorted_rank_table[i].len() - upper_sorted_rank_table[i][right_boundary.0].2 - 1);
            // left_zone.push(upper_sorted_rank_table[i][left_boundary.0].2);

            // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            // eprintln!("Zones: {},{},{}", left_zone[i],median_zone[i],right_zone[i]);

        }

        eprintln!("Finished computing rankings");

        // let sample_identity = Vec::from(0..dimensions.1);
        // let draw_order = Vec::from(0..dimensions.1);

        let sample_identity = (0..dimensions.0).collect();
        let draw_order: Vec<usize> = (0..dimensions.0).rev().collect();

        let mut median_history = Vec::new();
        median_history.push(current_upper_median.clone());

        let mut dispersion_history = Vec::new();
        dispersion_history.push(upper_sorted_MSDM.clone());


        OnlineMADM {
            counts : local_counts,
            dimensions: dimensions,
            drop_zeroes: drop_zeroes,
            current_index: 0,
            current_upper_median : current_upper_median.clone(),

            upper_sorted_samples: upper_sorted_samples,

            median_zone : median_zone.clone(),
            left_zone : left_zone.clone(),
            right_zone : right_zone.clone(),

            sample_identity: sample_identity,
            draw_order: draw_order.clone(),

            draw_backup: draw_order,
            zone_backup : (left_zone,median_zone,right_zone),
            sorted_median: current_upper_median,
            sorted_rank_table: upper_sorted_rank_table.clone(),
            sorted_MSDM: upper_sorted_MSDM.clone(),

            upper_sorted_rank_table : upper_sorted_rank_table,
            upper_sorted_MSDM : upper_sorted_MSDM,

            median_history: median_history,
            dispersion_history: dispersion_history

        }
    }

    pub fn reverse(&self) -> OnlineMADM {
        let mut reversed = self.clone();
        reversed.draw_order.reverse();
        reversed.draw_backup.reverse();
        reversed
    }

    pub fn reverse_draw_order(&mut self) {
        self.draw_order.reverse();
        self.draw_backup.reverse();
    }


    pub fn reset(&mut self) {
        // for (i,feature) in self.sorted_rank_table.iter().enumerate() {
        //     for (j, element) in feature.iter().enumerate() {
        //         self.upper_sorted_rank_table[i][j] = *element
        //     }
        // }

        self.upper_sorted_rank_table = self.sorted_rank_table.clone();
        self.upper_sorted_MSDM = self.sorted_MSDM.clone();

        self.current_index = 0;

        self.current_upper_median = self.sorted_median.clone();

        self.left_zone = self.zone_backup.0.clone();
        self.median_zone = self.zone_backup.1.clone();
        self.right_zone = self.zone_backup.2.clone();

        self.draw_order = self.draw_backup.clone();

        self.median_history.truncate(1);
        self.dispersion_history.truncate(1);

    }


    pub fn sort_by_feature(&mut self, key_feature:usize) -> Vec<usize> {

        // let key_vector: = self.upper_sorted_rank_table[key_feature].iter().map(|x| x.2).collect();

        let mut new_draw_order = vec![0;self.draw_order.len()];

        for sample in 0..self.dimensions.0 {
            let rank = self.upper_sorted_rank_table[key_feature][sample].2;
            new_draw_order[rank] = self.upper_sorted_rank_table[key_feature][sample].1;
        }

        // eprintln!("Sort by feature debug, new order done!");

        self.draw_order = new_draw_order.clone();

        new_draw_order
    }

    pub fn derive_subset(&self, samples: Vec<usize>) -> OnlineMADM {

        eprintln!("Deriving a subset!");

        let (new_rank_table,new_medians,new_identities,new_sorted_samples) = rank_table_subset(&self.upper_sorted_rank_table, &self.upper_sorted_samples, &samples, &self.sample_identity,self.drop_zeroes);


        let mut local_counts = Vec::new();
        // let flipped_counts = matrix_flip(&self.counts);
        for sample in samples {
            local_counts.push(self.counts[sample].clone())
        }
        // local_counts = matrix_flip(&local_counts);


        let dimensions = (local_counts.len(),local_counts[0].len());

        let mut upper_sorted_MSDM = Vec::new();

        let mut median_zone = Vec::new();
        let mut left_zone = Vec::new();
        let mut right_zone = Vec::new();

        for (i,(feature,median)) in new_rank_table.iter().zip(new_medians.iter()).enumerate() {

            // eprintln!("Sorted rank table: {:?}", new_rank_table[i]);

            let (madm,zones) = madm_ranking(&new_rank_table,i, &new_medians[i],true);
            upper_sorted_MSDM.push(madm);

            let boundaries = vec![(madm.0,madm.1),(madm.2,madm.3)];

            let left_boundary = boundaries.iter().min_by(|a,b| new_rank_table[i][a.0].2.cmp(& new_rank_table[i][b.0].2)).unwrap();
            let right_boundary = boundaries.iter().max_by(|a,b| new_rank_table[i][a.0].2.cmp(& new_rank_table[i][b.0].2)).unwrap();

            left_zone.push(zones.0);
            median_zone.push(zones.1);
            right_zone.push(zones.2);

            // median_zone.push(new_rank_table[i][right_boundary.0].2 - new_rank_table[i][left_boundary.0].2 + 1);
            // right_zone.push(new_rank_table[i].len() - new_rank_table[i][right_boundary.0].2 - 1);
            // left_zone.push(new_rank_table[i][left_boundary.0].2);

            // eprintln!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            // eprintln!("Zones: {},{},{}", left_zone[i],median_zone[i],right_zone[i]);

        }

        let mut median_history = Vec::new();
        median_history.push(new_medians.clone());

        let mut dispersion_history = Vec::new();
        dispersion_history.push(upper_sorted_MSDM.clone());

        // let sample_identity = Vec::from(0..dimensions.1);
        // let draw_order = Vec::from(0..dimensions.1);

        let draw_order: Vec<usize> = (0..dimensions.0).collect();


        OnlineMADM {
            counts : local_counts,
            dimensions: dimensions,
            drop_zeroes: true,
            current_index: 0,
            current_upper_median : new_medians.clone(),

            upper_sorted_samples: new_sorted_samples,

            median_zone : median_zone.clone(),
            left_zone : left_zone.clone(),
            right_zone : right_zone.clone(),

            sample_identity: new_identities,
            draw_order: draw_order.clone(),

            draw_backup: draw_order,
            zone_backup: (left_zone,median_zone,right_zone),
            sorted_median: new_medians,
            sorted_rank_table: new_rank_table.clone(),
            sorted_MSDM: upper_sorted_MSDM.clone(),

            upper_sorted_rank_table : new_rank_table,
            upper_sorted_MSDM : upper_sorted_MSDM,

            median_history: median_history,
            dispersion_history: dispersion_history
        }


    }

    fn dispersion_by_feature(&mut self, sorted_feature: usize) -> Vec<Vec<(f64,f64)>> {

        // let reordered_rank_table: Vec<usize> = self.upper_sorted_rank_table[sorted_feature].iter().map(|x| x.2).collect();
        //
        // for (i,feature) in self.upper_sorted_rank_table.iter_mut().enumerate() {
        //     feature.sort_unstable_by_key(|x| reordered_rank_table[x.1]);
        // }


        let mut output: Vec<Vec<(f64,f64)>> = Vec::new();

        self.sort_by_feature(sorted_feature);

        output.push(self.current_upper_median.iter().map(|x| x.1).zip(self.upper_sorted_MSDM.iter().map(|x| x.1)).collect());

        for sample in self {

            let sample_medians = sample.0.iter().map(|x| x.1);
            let sample_MADs = sample.1.iter().map(|x| x.1);

            output.push(sample_medians.zip(sample_MADs).collect());
        }

        output

    }

    pub fn test(&mut self) {

        eprintln!("Testing the online madm!");
        eprintln!("{:?}", self.current_upper_median);

        eprintln!("{:?}", self.dimensions);
        eprintln!("{:?}", self.upper_sorted_rank_table);
        eprintln!("{:?}", self.upper_sorted_MSDM);

        eprintln!("Computing a step!");
        self.next();
        eprintln!("Exited next function");
        eprintln!("{:?}", self.current_upper_median);
        eprintln!("{:?}", self.upper_sorted_MSDM);
        for i in 0..6 {
            self.next();
            eprintln!("Iteration {}", i);
            eprintln!("{:?}", self.current_upper_median);
            eprintln!("{:?}", self.upper_sorted_MSDM);

        }

        // eprintln!("{:?}", self.upper_sorted_MSDM);
    }
}


impl Iterator for OnlineMADM {

    type Item = (Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>,usize);

    fn next(&mut self) -> Option<(Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>,usize)> {

        // eprintln!("####################################################################################################################################################################");

        self.current_index = self.draw_order.pop().unwrap();

        // eprintln!("Current index: {}", self.current_index);

        if self.median_history.len() >= self.dimensions.0 {
            return None
        }

        for i in 0..self.dimensions.1 {

            // eprintln!("Processing: {},{}", self.current_index, i);

            let current_sample = pop_rank_table(&mut self.upper_sorted_rank_table,(i,self.current_index));

            // print!("{:?},",current_sample);

            if self.drop_zeroes {
                if current_sample.3 == 0. {
                    continue
                }
            }

            let new_median = rank_table_median(&self.upper_sorted_rank_table, &self.median_zone[i], &self.left_zone[i], &self.right_zone[i], i, self.current_upper_median[i], current_sample);

            let new_median_distance = median_distance(&self.upper_sorted_rank_table, i, self.upper_sorted_MSDM[i], self.current_upper_median[i], new_median, current_sample, &mut self.median_zone[i], &mut self.left_zone[i], &mut self.right_zone[i]);

            // eprintln!("Current zones:{},{},{}", self.left_zone[i],self.median_zone[i],self.right_zone[i]);
            // //
            // // eprintln!("Current index: {}", self.current_index);
            // eprintln!("{:?}", current_sample);
            // eprintln!("{:?}", self.upper_sorted_rank_table);
            // eprintln!("{:?}", new_median);
            // eprintln!("{:?}", new_median_distance);

            self.current_upper_median[i] = new_median;
            self.upper_sorted_MSDM[i] = new_median_distance;

        }

        // print!("\n");


        self.median_history.push(self.current_upper_median.clone());
        self.dispersion_history.push(self.upper_sorted_MSDM.clone());

        Some((self.current_upper_median.clone(),self.upper_sorted_MSDM.clone(),self.current_index.clone()))
    }

}

// impl<'a> Iterator for &'a OnlineMADM {
//
//     type Item = &'a (Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>);
//
//     fn next(&mut self) -> Option<&'a (Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>)> {
//
//         let result = self.next();
//         match result {
//             Some(x) => Some(&'a x),
//             None => None
//         }
//         // match self.next() {
//         //     Some(result) => Some(&'a result),
//         //     None => None
//         // }
//     }
// }
// impl<'a> Iterator for OnlineMADM<'a> {
//
//     type Item = (Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>);
//
//     fn next(&mut self) -> Option<(Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>)> {
//
//         let result = self.next();
//         match result {
//             Some(x) => Some(x),
//             None => None
//         }
//         // match self.next() {
//         //     Some(result) => Some(&'a result),
//         //     None => None
//         // }
//     }
// }


#[derive(Clone)]
#[derive(Debug)]
pub struct OnlineMADM {
    pub counts: Vec<Vec<f64>>,
    dimensions: (usize,usize),
    drop_zeroes: bool,
    current_index : usize,
    current_upper_median: Vec<(usize,f64)>,
    pub median_zone : Vec<usize>,
    pub left_zone : Vec<usize>,
    pub right_zone : Vec<usize>,

    sample_identity: Vec<usize>,
    draw_order: Vec<usize>,

    draw_backup: Vec<usize>,
    zone_backup: (Vec<usize>,Vec<usize>,Vec<usize>),
    sorted_median: Vec<(usize,f64)>,
    pub sorted_rank_table: Vec<Vec<(usize,usize,usize,f64,usize)>>,
    sorted_MSDM: Vec<(usize,f64,usize,f64)>,

    pub upper_sorted_rank_table: Vec<Vec<(usize,usize,usize,f64,usize)>>,

    pub upper_sorted_MSDM: Vec<(usize,f64,usize,f64)>,

    upper_sorted_samples: Vec<Vec<usize>>,

    pub median_history: Vec<Vec<(usize,f64)>>,
    pub dispersion_history: Vec<Vec<(usize,f64,usize,f64)>>
}


pub fn slow_description_test() {
    let test_vector_1: Vec<f64> = vec![-2.,-1.,0.,1.,1000.];
    let test_vector_2: Vec<f64> = vec![0.,0.,0.,0.,0.,1.,2.];
    let test_vector_3: Vec<f64> = vec![2.,3.,1.,0.,4.,1.,3.,-1.]; //1.5,1.5

    let description_1 = slow_description(&test_vector_1);
    let description_2 = slow_description(&test_vector_2);
    let description_3 = slow_description(&test_vector_3);

    eprintln!("{:?}", description_1);
    eprintln!("{:?}", description_2);
    eprintln!("{:?}", description_3);

    assert_eq!(description_1,(0.,1.0));
    assert_eq!(description_2,(0.,0.));
    assert_eq!(description_3,(1.5,1.5));
}

pub fn slow_vs_fast () {

    /// Generate some random feature vectors

    let mut thr_rng = rand::thread_rng();

    let mut counts = Vec::new();

    for feature in 0..10 {

        let mut rng = thr_rng.gen_iter::<f64>();

        counts.push(rng.take(100).collect());
    }

    let matrix = matrix_flip(&counts);

    let mut slow_medians: Vec<Vec<(f64,f64)>> = Vec::new();

    for sample in counts[0].clone() {
        slow_medians.push(Vec::new());
    }
    for (i,feature) in counts.clone().iter().enumerate() {
        for (j, sample) in feature.iter().enumerate() {
            let append = slow_description(&counts[i][j..].iter().cloned().collect::<Vec<f64>>());
            slow_medians[j].push(append);
        }
    }


    // We have generated feature medians and median distances using the slow function, checked above.
    //


    let mut model = OnlineMADM::new(matrix.clone(),false);

    let fast_medians: Vec<Vec<(f64,f64)>> = model.dispersion_by_feature(0);

    eprintln!("Target data:");
    eprintln!("{:?}", counts);

    assert_eq!(slow_medians,fast_medians);

}


pub fn slow_description(feature: &Vec<f64>) -> (f64,f64) {
    let feature_median = median(feature);
    let distances = feature.iter().map(|x| (x - feature_median.1).abs()).collect();
    let median_distance = median(&distances);
    (feature_median.1,median_distance.1)

}

fn matrix_flip(in_mat: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    let mut out = Vec::new();

    for _ in &in_mat[0] {
        out.push(vec![in_mat[0][0];in_mat.len()]);
    }

    for (i,iv) in in_mat.iter().enumerate() {
        for (j,jv) in iv.iter().enumerate() {
            out[j][i] = *jv;
        }
    }

    out
}

fn argsort(input: &Vec<f64>) -> Vec<(usize,f64)> {
    let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate1.sort_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    let out = intermediate2.iter().map(|x| (x.0,((x.1).1).clone())).collect();
    out
}
