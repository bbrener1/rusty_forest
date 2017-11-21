#![allow(dead_code)]
use std::env;
use std::io;
use std::sync::Arc;
// use std::rc::Weak;
use std::sync::Weak;
use std::fs::File;
use std::io::prelude::*;
use std::borrow::BorrowMut;
use std::borrow::Borrow;
use std::ops::DerefMut;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use rustml::math::*;
use std::thread::sleep;
use std::time;
#[macro_use] extern crate rustml;
extern crate rand;
use rand::Rng;
extern crate gnuplot;



fn main() {

    // let filename = "/Users/boris/taylor/vision/rust_prototype/raw_data/counts.txt";
    //
    // println!("Reading data");
    //
    // let count_array_file = File::open(filename).expect("File error!");
    // let mut count_array_lines = io::BufReader::new(&count_array_file).lines();
    //
    // let mut count_array: Vec<Vec<f64>> = Vec::new();
    //
    // for line in count_array_lines.by_ref().enumerate() {

        // count_array.push(Vec::new());

        // for gene in line.1.expect("Readline error").split_whitespace().enumerate() {
        //
        //     // println!("{}", gene.0);
        //     //
        //     // println!("{}", gene.1.parse::<f64>().unwrap() );
        //
        //     match gene.1.parse::<f64>() {
        //         Ok(exp_val) => {
        //             match count_array.last_mut() {
        //                 Some(last_vec) => last_vec.push(exp_val),
        //                 None => {
        //                     println!("0th Dimension of count array empty! ln 31");
        //                     panic!("Check count array creation rules or input file!")
        //                 }
        //             }
        //
        //         },
        //         Err(msg) => {
        //             println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
        //             match count_array.last_mut() {
        //                 Some(last_vec) => last_vec.push(0f64),
        //                 None => {
        //                     println!("0th Dimension of count array empty! ln 31");
        //                     panic!("Check count array creation rules or input file!")
        //                 }
        //             }
        //         }
        //     }
        //
        //     // if let Result::Ok(exp_val) = gene.1.parse::<f64>() {
        //     //     match count_array.last() {
        //     //         Some(last_vec) => last_vec.push(exp_val),
        //     //         None => None
        //     //     }
        //     }
        // if line.0 % 100 == 0 {
        //     println!("{}", line.0);
        // };
        //
        // count_array.push(
        //     line.1.expect("Readline error!").split_whitespace()
        //         .map(|x| {x.parse::<f64>().unwrap()}).collect::<Vec<f64>>());



        // if line.0%3 == 0 && line.0 > 0 {
        //
        //     println!("===========");
        //     println!("{}",line.0);
        //     // println!("{:?}", &count_array[..3].iter().map(|x| {x[..3]}));
        //
        // };
        //
        // }



    println!("##############################################################################################################");
    println!("##############################################################################################################");
    println!("##############################################################################################################");

    // let temp = tree {nodes: Vec::new(),counts: Vec::new()};
    // for cell in &mut count_array.into_iter() {
    //     for gene in &mut cell.into_iter() {
    //         println!("Test? {}", gene);
    //     }
    // }

    // let mut forest = Forest::grow_forest(count_array, 10, 1000);
    // forest.test();

    println!("Argmin test: {},{}", argmin(&vec![1.,3.,0.,5.]).0,argmin(&vec![1.,3.,0.,5.]).1);

    println!("Argsort test: {:?}", argsort(&vec![1.,3.,0.,5.]));

    let mut axis_sum_test: Vec<Vec<f64>> = Vec::new();

    // axis_sum_test.push(vec![1.,2.,3.]);
    // axis_sum_test.push(vec![4.,5.,6.]);
    // axis_sum_test.push(vec![0.,1.,0.]);
    let temp: [f64;7] = [-3.,-2.,-1.,0.,10.,15.,20.];
    let temp2 = temp.into_iter().cloned().collect();
    let temp3 = vec![temp2];
    let temp4 = matrix_flip(&temp3);

    let mut thr_rng = rand::thread_rng();
    let rng = thr_rng.gen_iter::<f64>();
    let temp5: Vec<f64> = rng.take(49).collect();
    let temp6 = matrix_flip(&(vec![temp5.clone()]));

    // axis_sum_test.push(vec![1.,2.,3.]);
    // axis_sum_test.push(vec![4.,5.,6.]);
    // axis_sum_test.push(vec![7.,8.,9.]);

    println!("Source floats: {:?}", temp5);

    let mut forest = Forest::grow_forest(temp6,10,1);
    forest.test();

    // println!("Inner axist test! {:?}", inner_axis_mean(&axis_sum_test));
    // println!("Matrix flip test! {:?}", matrix_flip(&axis_sum_test));

    slow_description_test();
    slow_vs_fast();

}

fn slow_description(feature: &Vec<f64>) -> (f64,f64) {
    let feature_median = median(feature);
    let distances = feature.iter().map(|x| (x - feature_median.1).abs()).collect();
    let median_distance = median(&distances);
    (feature_median.1,median_distance.1)

}


impl Forest {



    fn grow_forest(counts : Vec<Vec<f64>>, forest_size: usize, tree_features: usize) -> Forest {

        let count_size1:usize = counts.len();
        let count_size2 = {&counts[0]}.len();

        let mut forest = Forest{trees:Vec::new(),counts: Arc::new(counts),count_size:(count_size1,count_size2),tree_features:tree_features};

        for n in 0..forest_size {
            let in_features = rand::sample(&mut rand::thread_rng(), 0..count_size2, tree_features);
            let out_features = {0..count_size2}.collect::<Vec<usize>>();
            let tree_vector : &mut Vec<Tree> = forest.trees.borrow_mut();
            tree_vector.push(Tree::plant_tree(in_features,out_features,forest.counts.clone()))
        }
        forest
    }

    fn test(&mut self) {
        println!("Printing {} trees!",self.trees.len());
        for (i,tree) in self.trees.iter().by_ref().enumerate() {
            println!("Printing tree {}",i);
            println!("{}", tree.input_features.len());
            println!("{}", tree.output_features.len());
            println!("Selection of input features");
            for feature in tree.input_features.iter().take(10).by_ref(){
                print!("{},",feature);
            }
            print!("\n");
            // println!("Contains {} nodes", tree.nodes.len());
            //     for node in &tree.nodes {
            //         node.test()
            //         // println!("{}", node.feature.unwrap_or(0))
            //     }


        };
        // self.trees.push(Tree::plant_tree(vec![627],vec![3964],self.counts.clone()));
        let split = self.trees[2].nodes[0].find_split(0);
        println!("Best split for feature 627 is found to be {},{}", split.0,split.1);
    }
}

struct Forest {
    trees: Vec<Tree>,
    count_size: (usize,usize),
    counts: Arc<Vec<Vec<f64>>>,
    tree_features: usize
}


impl Tree {

    fn plant_tree(in_f:Vec<usize>,out_f:Vec<usize>,f:Arc<Vec<Vec<f64>>>) -> Tree {
        let mut tree = Tree {
            nodes: Vec::new(),
            input_features:Arc::new(in_f),
            // input_samples:Arc::new(Vec::new()),
            input_samples:(0..f.len()).collect(),
            output_features:Arc::new(out_f),
            weights:Arc::new(Vec::new()),
            counts:Arc::downgrade(&f)};
        tree.nodes.push(Node::first(tree.counts.clone(),tree.input_samples.clone(),tree.output_features.to_vec()));

        tree
    }


}

struct Tree {
    nodes: Vec<Node>,
    input_features: Arc<Vec<usize>>,
    input_samples: Vec<usize>,
    output_features: Arc<Vec<usize>>,
    weights: Arc<Vec<f64>>,
    counts: Weak<Vec<Vec<f64>>>
}

impl Node {

    fn first(counts:Weak<Vec<Vec<f64>>>, samples:Vec<usize>, output_features:Vec<usize>) -> Node {

        let loc_counts: Vec<Vec<f64>> = samples.iter().map(|x| counts.upgrade().expect("Dead tree!")[x.clone()].clone()).collect();

        let means: Vec<f64> = loc_counts.iter().fold(vec![0f64;loc_counts[0].len()], |acc, x|
            {
                x.iter().zip(acc.iter()).map(|y| y.0 + y.1).collect::<Vec<f64>>()
            }).iter().map(|z| z/(loc_counts.len() as f64)).collect();

        let variance = loc_counts.iter().fold(vec![0f64;loc_counts[0].len()], |acc,x| {
                x.iter().enumerate().zip(acc.iter()).map(|y| ((y.0).1 - means[(y.0).0]).powi(2) + y.1).collect()
            }
            ).iter().map(|z| z/(loc_counts.len() as f64)).collect();



        let mut result = Node {
            feature:None,
            split: None,
            means: means,
            output_features: output_features,
            indecies: samples,
            variance:variance,
            weights:vec![1.;loc_counts[0].len()],
            children:Vec::new(),
            parent:None,
            counts:counts
        };

        result.means.push(0.);
        result.weights.push(0.);
        result.variance.push(0.);

        result
    }

    fn test(&self) {
        println!("Node test (feature, split,indecies)");
        println!("{}", self.feature.unwrap_or(0));
        println!("{}", self.split.unwrap_or(0.0));
        println!("{}", self.indecies.len());
        println!("Weights");
        for weight in self.weights.iter().by_ref() {
            println!("{}", weight);
        }

    }

    // fn first(tree: &Tree) -> Node {
    //     Node {
    //         feature:None,
    //         indecies:Vec::new(),
    //         means:Vec::new(), weights:Vec::new()
    //         ,std_devs:Vec::new(),children:Vec::new(), parent:None, counts:tree.counts.clone()}




    fn find_split(&mut self, feature:usize) -> (usize,f64) {

        let global_counts = self.counts.upgrade().expect("Missing counts?");

        let node_counts: Vec<Vec<f64>> = self.indecies.iter().map(|x| self.output_features.iter().map(|y| global_counts[*x][*y]).collect()).collect();

        println!("Testing Online MADM");

        let mut online = OnlineMADM::new(node_counts);

        online.test();

        (0,0.)
    }

}

struct Node {
    feature: Option<usize>,
    split: Option<f64>,
    output_features: Vec<usize>,
    indecies: Vec<usize>,
    means: Vec<f64>,
    weights: Vec<f64>,
    variance: Vec<f64>,
    children: Vec<Node>,
    parent: Option<Arc<Node>>,
    counts: Weak<Vec<Vec<f64>>>
}

fn find_variance(samples: Vec<&Vec<f64>>) -> Vec<f64> {


    for sample in samples {

    }

    Vec::new()
}

fn argmin(in_vec: &Vec<f64>) -> (usize,f64) {
    let mut min_ind = 0;
    let mut min_val: f64 = 1./0.;
    for (i,val) in in_vec.iter().enumerate() {
        // println!("Argmin debug:{},{},{}",i,val,min_val);
        // match val.partial_cmp(&min_val).unwrap_or(Ordering::Less) {
        //     Ordering::Less => println!("Less"),
        //     Ordering::Equal => println!("Equal"),
        //     Ordering::Greater => println!("Greater")
        // }
        match val.partial_cmp(&min_val).unwrap_or(Ordering::Less) {
            Ordering::Less => {min_val = val.clone(); min_ind = i.clone()},
            Ordering::Equal => {},
            Ordering::Greater => {}
        }
    }
    (min_ind,min_val)
}

fn inner_axis_sum(in_mat: &Vec<Vec<f64>>) -> Vec<f64> {

    let mut s = vec![0f64;in_mat[0].len()];

    for i in in_mat {
        for j in i.iter().enumerate() {
            s[j.0] += *j.1;
        }
    }
    // println!("Inner axis being computed: {:?}", in_mat.iter().map(|x| x[0]).take(10).collect::<Vec<_>>());
    // println!("{}", in_mat[0].len());
    // println!("Inner axis sum: {}", s[0]);
    s
}

fn inner_axis_mean(in_mat: &Vec<Vec<f64>>) -> Vec<f64> {

    let mut s = vec![0f64;in_mat[0].len()];

    for i in in_mat {
        for j in i.iter().enumerate() {
            s[j.0] += *j.1/(in_mat.len() as f64);
        }
    }

    s
}

fn inner_axis_variance_sum(in_mat: &Vec<Vec<f64>>, in_means: Option<Vec<f64>>) -> Vec<f64> {

    let m: Vec<f64>;

    match in_means {
        Option::Some(input) => m = input,
        Option::None => m = inner_axis_mean(in_mat)
    }

    println!("Inner axis mean: {:?}", m);

    let mut vs = vec![0f64;in_mat[0].len()];

    for i in in_mat {
        for j in i.iter().enumerate() {
            // println!("Variance sum compute");
            // println!("{}",*j.1);
            // println!("{}", m[j.0]);
            vs[j.0] += (*j.1 - m[j.0]).powi(2);
            // println!("{}", vs[j.0]);
        }
    }
    // println!("Inner_axis being computed: {:?}", in_mat.iter().map(|x| x[0]).take(10).collect::<Vec<_>>());
    // println!("{}", in_mat.len());
    // println!("Inner axis variance sum: {}", vs[0]);
    vs
}

fn inner_axis_stats(in_mat: &Vec<Vec<f64>>) -> (Vec<f64>,Vec<f64>) {

    let m = inner_axis_mean(in_mat);

    let mut v = vec![0f64;in_mat[0].len()];

    for i in in_mat {
        for j in i.iter().enumerate() {
            v[j.0] += (*j.1 - m[j.0]).powi(2)/(v.len() as f64);
        }
    }

    (m,v)
}

// impl Owned {
//     fn mutate(&mut self, into: i32) {
//         self.element = into
//     }
// }
//
// struct Owned {
//     element: i32
// }


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
    intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    let out = intermediate2.iter().map(|x| (x.0,((x.1).1).clone())).collect();
    out
}

// fn argsort(input: &Vec<f64>) -> Vec<(usize,f64)> {
//     let mut out = input.iter().cloned().enumerate().collect::<Vec<(usize,f64)>>();
//     out.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//     out
// }


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

fn sort_upper(input: &Vec<f64>) -> (Vec<usize>, Vec<(usize,usize,usize,f64,usize)>,usize,f64) {

    let mut median_index = 0;
    let mut median_value = 0.;

    println!("Sorting upper, trying to find median");

    println!("Feature size: {}", input.len());

    let mut sorted_rank_table: Vec<(usize,usize,usize,f64,usize)> = Vec::new();

    let mut intermediate = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));

    println!("Feature sorted!");

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

    median_index = intermediate[median_index].0;

    println!("Median computed! {}", median_value);

    for (i,sample) in intermediate.iter().enumerate() {

        // println!("{}", i);

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

    println!("Sorting back into correct order!");

    sorted_rank_table.sort_unstable_by(|a,b| a.1.cmp(&b.1));

    let order_of_samples = intermediate.iter().map(|x| x.0).collect();

    println!("Returning ranking table:");

    (order_of_samples,sorted_rank_table,median_index,median_value)
}

// fn madm_ranking(sorted_rank_table: &Vec<(usize,usize,usize,f64,usize)>, sorted_samples: &Vec<usize>, current_median:&(usize,f64)) -> (Vec<(usize,usize,usize,f64,usize)>,(usize,f64,usize,f64)) {

fn madm_ranking(sorted_rank_table: &Vec<Vec<(usize,usize,usize,f64,usize)>>,feature:usize, current_median:&(usize,f64)) -> (usize,f64,usize,f64) {

    println!("Computing MADM rankings!");

    let mut new_median_distance = (0,0.,0,0.);


    let mut left_sample = sorted_rank_table[feature][sorted_rank_table[feature][current_median.0].0];
    let mut right_sample = sorted_rank_table[feature][sorted_rank_table[feature][current_median.0].4];

    let size = sorted_rank_table[feature].len();

    let mut current_samples = 1;

    let target_samples = ((size as f64 / 2.).trunc() + 1.) as i32;

    let mut closest_left_sample = sorted_rank_table[feature][current_median.0];
    let mut closest_right_sample = sorted_rank_table[feature][current_median.0];

    println!("Target samples: {}", target_samples);

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

        println!("{},{:?},{:?},{},{}", current_samples, left_sample,right_sample, left_sample.3-current_median.1,right_sample.3-current_median.1);
        println!("{:?},{:?},{},{}",closest_left_sample,closest_right_sample,closest_left_sample.3-current_median.1,closest_right_sample.3-current_median.1);

    }

    if size%2 == 0 {
        let mut closer = closer_sample(closest_left_sample, closest_right_sample, *current_median, feature, sorted_rank_table);
        closer.reverse();

        println!("Even samples: {:?}", closer);

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

        println!("Odd samples: {:?}", closer);

        new_median_distance.0 = closer[0].1;
        new_median_distance.1 = (closer[0].3 - current_median.1).abs();

        let opposite_ranking = vec![closest_left_sample,closest_right_sample];
        let opposite_closest = opposite_ranking.iter().min_by(|a,b| (a.3 - current_median.1).abs().partial_cmp(&(b.3 - current_median.1).abs()).unwrap_or(Ordering::Greater)).unwrap();

        new_median_distance.2 = opposite_closest.1;
        new_median_distance.3 = (opposite_closest.3 - current_median.1).abs()

    }

    println!("Computed new median distance: {:?}", new_median_distance);
    //
    // new_median_distance



    new_median_distance



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

    println!("Popping a sample!");

    assert!(rank_table[pop.0].len() > 1, "Popped an empty feature!");

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
        println!("Popping an edge: {},{}", greater_than, less_than);
        if target.4 == target.1 {
            rank_table[pop.0][target.0].4 = rank_table[pop.0][target.0].1;
        }
        if target.0 == target.1 {
            rank_table[pop.0][target.4].0 = rank_table[pop.0][target.4].1;
        }
    }

    rank_table[pop.0][pop.1].0 = target.1;
    rank_table[pop.0][pop.1].4 = target.1;

    println!("Sample popped: {:?}", target);

    target

}

fn rank_table_median(rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, right_zone: &usize, median_zone:&usize, left_zone:&usize, feature:usize, old_median:(usize,f64), removed: (usize,usize,usize,f64,usize)) -> (usize,f64) {

    println!("Computing new median!");

    let mut new_median: (usize,f64) = (0,0.);

    if (right_zone+left_zone+median_zone) < 3 {
        return (removed.4, rank_table[feature][removed.4].3)
    }
////////###### Warning table size hack, check against current index may be better!

    // if (right_zone+left_zone+median_zone -1 ) % 2 != 0 {
    if (rank_table[feature].len() - removed.1) %2 != 0 {

        println!("Even median!");

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

        println!("Odd median!");
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

    println!("Computed new median!");
    println!("{:?}", new_median);

    new_median

}

fn closer_sample(left: (usize,usize,usize,f64,usize), right: (usize,usize,usize,f64,usize), median: (usize, f64), feature: usize, sorted_rank_table: & Vec<Vec<(usize,usize,usize,f64,usize)>>) -> Vec<(usize,usize,usize,f64,usize)> {

    // println!("Closer sample debug: {:?},{:?}",left,right);

    let inner_left = sorted_rank_table[feature][left.4];
    let inner_right = sorted_rank_table[feature][right.0];

    // println!("Closer sample debug: {:?},{:?}", inner_left,inner_right);

    let mut possibilities = vec![left,right,inner_right,inner_left];

    possibilities.sort_unstable_by(|a,b| (a.3 - median.1).abs().partial_cmp(&(b.3 - median.1).abs()).unwrap_or(Ordering::Greater));

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

    println!("Expanding by 1:");
    println!("{:?},{:?}",left,right);
    println!("{:?},{:?}", left_boundary,right_boundary);

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


    println!("Computing new median distance!");

    let mut new_median_distance = old_median_distance.clone();

    let change =  new_median.1 - old_median.1;

    let mut median_distance_ordered = vec![(old_median_distance.0,old_median_distance.1),(old_median_distance.2,old_median_distance.3)];
    median_distance_ordered.sort_unstable_by_key(|x| sorted_rank_table[feature][x.0].2);
    let (mut left_boundary, mut right_boundary) =
    (sorted_rank_table[feature][median_distance_ordered[0].0],sorted_rank_table[feature][median_distance_ordered[1].0]);

    println!("Before recomputation:");
    println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    println!("Removed: {:?}", removed);
    println!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);


    if removed.2 < left_boundary.2 {
        *left_zone -= 1;
    }
    else if removed.2 > right_boundary.2 {
        *right_zone -= 1;
    }
    else if left_boundary.2 <= removed.2 && removed.2 <= right_boundary.2 {
        *median_zone -= 1;
    }


    println!("Subtracted removed sample!");

    println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    println!("Removed: {:?}", removed);
    println!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);


    if removed.1 == left_boundary.1 {
        left_boundary = sorted_rank_table[feature][removed.4];
    }

    if removed.1 == right_boundary.1 {
        right_boundary = sorted_rank_table[feature][removed.0];
    }

    println!("Handled removed boundary");

    println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    println!("Removed: {:?}", removed);
    println!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);



    if (*median_zone as i32 - 2) > (*left_zone + *right_zone) as i32 {
        let contracted = contract_by_1(sorted_rank_table, feature, left_boundary, right_boundary, right_zone, median_zone, left_zone, old_median);
        left_boundary = contracted.0;
        right_boundary = contracted.1;
    }

    println!("Handled median overflow");

    println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    println!("Removed: {:?}", removed);
    println!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);


    if (*median_zone as i32) <= ((*left_zone+*right_zone) as i32) {
        let expanded = expand_by_1(sorted_rank_table, feature, left_boundary, right_boundary, right_zone, median_zone, left_zone, old_median);
        left_boundary = expanded.0;
        right_boundary = expanded.1;
    }


    println!("Zones fixed:");
    println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
    println!("Removed: {:?}", removed);
    println!("Zones: {},{},{}", *left_zone,*median_zone,*right_zone);

    if change > 0. {
        println!("Moving right!");
        for i in 0..sorted_rank_table[feature].len() {
            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            println!("New median: {:?}", new_median);
            if right_boundary.1 == right_boundary.4 {
                break
            }

            println!("Comparison: {},{}",(sorted_rank_table[feature][right_boundary.4].3 - new_median.1).abs(),(sorted_rank_table[feature][left_boundary.4].3 - new_median.1).abs());

            match (sorted_rank_table[feature][right_boundary.4].3 - new_median.1).abs().partial_cmp(&(left_boundary.3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
                Ordering::Less => {
                    right_boundary = sorted_rank_table[feature][right_boundary.4];
                    left_boundary = sorted_rank_table[feature][left_boundary.4];
                    println!("Moved right!")
                },
                Ordering::Greater => break,
                Ordering::Equal =>{
                    right_boundary = sorted_rank_table[feature][right_boundary.4];
                    left_boundary = sorted_rank_table[feature][left_boundary.4];
                }
            }
            *left_zone += 1;
            *right_zone -= 1;
            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
        }
    }
    if change < 0. {
        println!("Moving left!");
        for i in 0..sorted_rank_table[feature].len() {
            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            println!("New median: {:?}", new_median);
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
            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
        }
    }


    let mut median_distance_disordered = vec![(right_boundary.1,(right_boundary.3 - new_median.1).abs()),(left_boundary.1,(left_boundary.3 - new_median.1).abs())];
    median_distance_disordered.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
    new_median_distance = (median_distance_disordered[1].0,median_distance_disordered[1].1, median_distance_disordered[0].0, median_distance_disordered[0].1);

    let sample_space = *median_zone+*left_zone+*right_zone;

    println!("{}", sample_space);
    println!("{:?}", median_distance_disordered);

    if sample_space % 2 == 0 {

        println!("Even samples, computing split median!");

        let mut distances = closer_sample(left_boundary, right_boundary, new_median, feature, sorted_rank_table);
        distances.reverse();

        println!("{:?}",distances);

        new_median_distance.1 = (new_median_distance.1 + (distances[1].3 - new_median.1).abs())/2.;

        // println!("Even distances, computing split median!");
        //
        // let distance_to_outer_left = (left_boundary.0, (sorted_rank_table[feature][left_boundary.0].3 - new_median.1).abs());
        //
        // let distance_to_outer_right = (right_boundary.4, (sorted_rank_table[feature][right_boundary.4].3 - new_median.1).abs());
        //
        // println!("Outer distances: {:?},{:?}", distance_to_outer_left, distance_to_outer_right);
        //
        // let temp_outer_median = vec![distance_to_outer_right,distance_to_outer_left];
        // let outer_median = temp_outer_median.iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or(&(0,0.)).clone();
        //
        // println!("Outer median: {:?}", outer_median);
        //
        // new_median_distance.1 = (new_median_distance.1 + outer_median.1)/2.;

    }


    println!("Done computing new median distance!");
    println!("{:?}", new_median_distance);

    new_median_distance

}


#[allow(dead_code)]
impl OnlineMADM {

    fn new(counts: Vec<Vec<f64>>) -> OnlineMADM {

        println!("Computing MADM initial conditions");

        let mut local_counts = counts;
        let dimensions = (local_counts.len(),local_counts[0].len());
        let mut upper_sorted_samples = Vec::new();
        let mut upper_sorted_rank_table = Vec::new();
        // for feature in matrix_flip(&local_counts) {
        //     upper_sorted_counts.push(argsort(&feature));
        // }
        let mut current_upper_median = Vec::new();
        for feature in matrix_flip(&local_counts) {
            println!("Computing feature median");
            let description = sort_upper(&feature);
            // current_upper_median.push(median(&feature.iter().map(|x| x.1).collect()));
            upper_sorted_samples.push(description.0);
            upper_sorted_rank_table.push(description.1);
            current_upper_median.push((description.2,description.3));
        }
        let mut upper_sorted_MSDM = Vec::new();

        let mut median_zone = Vec::new();
        let mut left_zone = Vec::new();
        let mut right_zone = Vec::new();

        for (i,(feature,median)) in upper_sorted_rank_table.iter().zip(current_upper_median.iter()).enumerate() {

            println!("Sorted rank table: {:?}", upper_sorted_rank_table[i]);

            let madm = madm_ranking(&upper_sorted_rank_table,i, &current_upper_median[i]);
            upper_sorted_MSDM.push(madm);

            let boundaries = vec![(madm.0,madm.1),(madm.2,madm.3)];

            let left_boundary = boundaries.iter().min_by(|a,b| upper_sorted_rank_table[i][a.0].2.cmp(& upper_sorted_rank_table[i][b.0].2)).unwrap();
            let right_boundary = boundaries.iter().max_by(|a,b| upper_sorted_rank_table[i][a.0].2.cmp(& upper_sorted_rank_table[i][b.0].2)).unwrap();

            median_zone.push(upper_sorted_rank_table[i][right_boundary.0].2 - upper_sorted_rank_table[i][left_boundary.0].2 + 1);
            right_zone.push(upper_sorted_rank_table[i].len() - upper_sorted_rank_table[i][right_boundary.0].2 - 1);
            left_zone.push(upper_sorted_rank_table[i][left_boundary.0].2);

            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            println!("Zones: {},{},{}", left_zone[i],median_zone[i],right_zone[i]);

        }





        OnlineMADM {
            counts : local_counts,
            dimensions: dimensions,
            current_index: 0,
            current_upper_median : current_upper_median,

            median_zone : median_zone,
            left_zone : left_zone,
            right_zone : right_zone,


            upper_sorted_rank_table : upper_sorted_rank_table,
            upper_sorted_samples : upper_sorted_samples,
            upper_sorted_MSDM : upper_sorted_MSDM,
        }



    }

    fn variance_by_feature(&mut self, sorted_feature: usize) -> Vec<Vec<(f64,f64)>> {

        // let reordered_rank_table: Vec<usize> = self.upper_sorted_rank_table[sorted_feature].iter().map(|x| x.2).collect();
        //
        // for (i,feature) in self.upper_sorted_rank_table.iter_mut().enumerate() {
        //     feature.sort_unstable_by_key(|x| reordered_rank_table[x.1]);
        // }


        let mut output = Vec::new();

        for sample in self {

            let sample_medians = sample.0.iter().map(|x| x.1);
            let sample_MADs = sample.1.iter().map(|x| x.1);

            output.push(sample_medians.zip(sample_MADs).collect());
        }

        output

    }

    fn test(&mut self) {

        println!("Testing the online madm!");
        println!("{:?}", self.current_upper_median);

        println!("{:?}", self.dimensions);
        println!("{:?}", self.upper_sorted_rank_table);
        println!("{:?}", self.upper_sorted_MSDM);

        println!("Computing a step!");
        self.next();
        println!("Exited next function");
        println!("{:?}", self.current_upper_median);
        println!("{:?}", self.upper_sorted_MSDM);
        for i in 0..6 {
            self.next();
            println!("Iteration {}", i);
            println!("{:?}", self.current_upper_median);
            println!("{:?}", self.upper_sorted_MSDM);

        }

        // println!("{:?}", self.upper_sorted_MSDM);
    }

    // fn next(&mut self) -> Option<(Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>)> {
    //
    //     if self.current_index == self.dimensions.0 {
    //         return None
    //     }
    //
    //     for i in 0..self.dimensions.1 {
    //
    //         let current_sample = pop_rank_table(&mut self.upper_sorted_rank_table,(i,self.current_index));
    //
    //         let new_median = rank_table_median(&self.upper_sorted_rank_table, &self.upper_sorted_samples, i, self.current_upper_median[i], current_sample);
    //
    //         let new_median_distance = median_distance(&self.upper_sorted_rank_table, i, self.upper_sorted_MSDM[i], self.current_upper_median[i], new_median, current_sample, &mut self.median_zone[i], &mut self.left_zone[i], &mut self.right_zone[i]);
    //
    //         println!("Current zones:{},{},{}", self.left_zone[i],self.median_zone[i],self.right_zone[i]);
    //
    //         println!("{:?}", current_sample);
    //         println!("{:?}", self.upper_sorted_rank_table);
    //         println!("{:?}", new_median);
    //         println!("{:?}", new_median_distance);
    //
    //         self.current_upper_median[i] = new_median;
    //         self.upper_sorted_MSDM[i] = new_median_distance;
    //
    //         self.current_index += 1;
    //         //
    //         // self.current_upper_median[i] = new_median;
    //
    //         // let where_to = insert_into_sorted(&mut self.lower_sorted_counts[i], (current_sample.1,current_sample.3), current_sample.2);
    //         //
    //         // insert_into_sorted(&mut self.lower_sorted_MSDM[i], (self.lower_sorted_counts[i][where_to].1,self.lower_sorted_counts[i][where_to].2-self.current_lower_median[i].1), self.lower_sorted_counts[i][where_to].0);
    //
    //
    //
    //
    //
    //         // self.lower_sorted_MSDM.insert(where_to,(self.lower_sorted_counts[i].1self.lower_sorted_counts[i][where_to].2 - self.current_lower_median[i].1));
    //
    //     }
    //
    //
    //     Some((self.current_upper_median.clone(),self.upper_sorted_MSDM.clone()))
    // }

}

impl Iterator for OnlineMADM {

    type Item = (Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>);

    fn next(&mut self) -> Option<(Vec<(usize,f64)>,Vec<(usize,f64,usize,f64)>)> {

        println!("####################################################################################################################################################################");

        if self.current_index >= self.dimensions.0-1 {
            return None
        }

        for i in 0..self.dimensions.1 {

            println!("Processing: {},{}", self.current_index, i);

            let current_sample = pop_rank_table(&mut self.upper_sorted_rank_table,(i,self.current_index));

            let new_median = rank_table_median(&self.upper_sorted_rank_table, &self.median_zone[i], &self.left_zone[i], &self.right_zone[i], i, self.current_upper_median[i], current_sample);

            let new_median_distance = median_distance(&self.upper_sorted_rank_table, i, self.upper_sorted_MSDM[i], self.current_upper_median[i], new_median, current_sample, &mut self.median_zone[i], &mut self.left_zone[i], &mut self.right_zone[i]);

            println!("Current zones:{},{},{}", self.left_zone[i],self.median_zone[i],self.right_zone[i]);

            println!("Current index: {}", self.current_index);
            println!("{:?}", current_sample);
            println!("{:?}", self.upper_sorted_rank_table);
            println!("{:?}", new_median);
            println!("{:?}", new_median_distance);

            self.current_upper_median[i] = new_median;
            self.upper_sorted_MSDM[i] = new_median_distance;


            //
            // self.current_upper_median[i] = new_median;

            // let where_to = insert_into_sorted(&mut self.lower_sorted_counts[i], (current_sample.1,current_sample.3), current_sample.2);
            //
            // insert_into_sorted(&mut self.lower_sorted_MSDM[i], (self.lower_sorted_counts[i][where_to].1,self.lower_sorted_counts[i][where_to].2-self.current_lower_median[i].1), self.lower_sorted_counts[i][where_to].0);





            // self.lower_sorted_MSDM.insert(where_to,(self.lower_sorted_counts[i].1self.lower_sorted_counts[i][where_to].2 - self.current_lower_median[i].1));

        }

        self.current_index += 1;

        Some((self.current_upper_median.clone(),self.upper_sorted_MSDM.clone()))
    }

}

struct OnlineMADM {
    counts: Vec<Vec<f64>>,
    dimensions: (usize,usize),
    current_index : usize,
    current_upper_median: Vec<(usize,f64)>,
    // upper_sorted_counts: Vec<Vec<(usize,f64)>>,
    median_zone : Vec<usize>,
    left_zone : Vec<usize>,
    right_zone : Vec<usize>,

    upper_sorted_samples: Vec<Vec<usize>>,
    upper_sorted_rank_table: Vec<Vec<(usize,usize,usize,f64,usize)>>,

    upper_sorted_MSDM: Vec<(usize,f64,usize,f64)>,
}


fn slow_description_test() {
    let test_vector_1: Vec<f64> = vec![-2.,-1.,0.,1.,1000.];
    let test_vector_2: Vec<f64> = vec![0.,0.,0.,0.,0.,1.,2.];
    let test_vector_3: Vec<f64> = vec![2.,3.,1.,0.,4.,1.,3.,-1.]; //1.5,1.5

    let description_1 = slow_description(&test_vector_1);
    let description_2 = slow_description(&test_vector_2);
    let description_3 = slow_description(&test_vector_3);

    println!("{:?}", description_1);
    println!("{:?}", description_2);
    println!("{:?}", description_3);

    assert_eq!(description_1,(0.,1.0));
    assert_eq!(description_2,(0.,0.));
    assert_eq!(description_3,(1.5,1.5));
}

fn slow_vs_fast () {

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


    let mut model = OnlineMADM::new(matrix.clone());

    let fast_medians: Vec<Vec<(f64,f64)>> = model.variance_by_feature(0);

    println!("Target data:");
    println!("{:?}", counts);

    assert_eq!(slow_medians[1..],fast_medians[..]);

}
