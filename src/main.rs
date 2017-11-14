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

    // axis_sum_test.push(vec![1.,2.,3.]);
    // axis_sum_test.push(vec![4.,5.,6.]);
    // axis_sum_test.push(vec![7.,8.,9.]);

    let mut forest = Forest::grow_forest(temp4,10,1);
    forest.test();

    // println!("Inner axist test! {:?}", inner_axis_mean(&axis_sum_test));
    // println!("Matrix flip test! {:?}", matrix_flip(&axis_sum_test));

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

    if input.len() % 2 == 0 {
        index = input.len()/2;
        value = (input[index] + input[index+1]) / 2.
    }
    else {
        if input.len() % 2 == 1 {
            index = (input.len()-1)/2;
            value = input[index]
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

    println!("Median computed!");

    for (i,sample) in intermediate.iter().enumerate() {

        println!("{}", i);

        if i == 0 {
            sorted_rank_table.push((0,sample.0,i,*sample.1,intermediate[i+1].0));
        }
        if i == (intermediate.len() - 1) {
            sorted_rank_table.push((intermediate[i-1].0,sample.0,i,*sample.1,intermediate.len()-1));
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

fn madm_ranking(sorted_rank_table: &Vec<(usize,usize,usize,f64,usize)>, sorted_samples: &Vec<usize>, current_median:&(usize,f64)) -> (Vec<(usize,usize,usize,f64,usize)>,(usize,f64,usize,f64)) {

    println!("Computing MADM rankings!");

    let mut positive_median_index = 0;
    let mut negative_median_index = 0;

    let mut distance = Vec::new();

    println!("Feature size: {}", sorted_rank_table.len());

    for sample in sorted_rank_table {
        distance.push((sample.1,(sample.3-current_median.1).abs()));
    }

    distance.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));

    let mut distance_rank_table = Vec::new();

    for (i,sample) in distance.iter().enumerate() {
        if i == 0 {
            distance_rank_table.push((0,sample.0,i,sample.1,distance[i+1].0));
        }
        if i == (distance.len()-1) {
            distance_rank_table.push((distance[i-1].0,sample.0,i,sample.1,distance.len()));
        }

        if {i != 0} && {i < (distance.len()-1)} {
            distance_rank_table.push((distance[i-1].0,sample.0,i,sample.1,distance[i+1].0));
        }
    }

    // let madm = median(&{distance_rank_table.iter().map(|x| x.3).collect()});

    let median_distance_sample = median(& distance_rank_table.iter().map(|x| x.3).collect());

    let mut madm = (median_distance_sample.0,median_distance_sample.1,current_median.0,current_median.1);

    madm.0 = distance_rank_table[madm.0].1;

    let opposite_madm_rank_steps = (sorted_rank_table[madm.0].2 as i32 - sorted_rank_table[current_median.0].2 as i32).abs();

    let direction = {
        if opposite_madm_rank_steps > 0 {
            1
        }
        else {
            if opposite_madm_rank_steps < 0 {
                4
            }
            else {
                1
            }
        }
    };

    for i in 0..opposite_madm_rank_steps {
        match direction {
            4 => madm.2 = sorted_rank_table[madm.2].4,
            1 => madm.2 = sorted_rank_table[madm.2].1,
            x => madm.2 = sorted_rank_table[madm.2].0
        }
    }

    madm.3 = sorted_rank_table[madm.2].3 - current_median.1;


    distance_rank_table.sort_unstable_by(|a,b| (a.1).cmp(&b.1));

    (distance_rank_table,madm)
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
    if target.1 > target.0 {
        greater_than = target.0;
    }
    else {
        greater_than = target.4;
        not_edge = false;
    }

    let mut less_than = target.4;
    if target.4 > target.1 {
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

    println!("Sample popped!");

    target

}

fn rank_table_median(rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, sorted_samples: &Vec<Vec<usize>>, feature:usize, old_median:(usize,f64), removed: (usize,usize,usize,f64,usize)) -> (usize,f64) {

    println!("Computing new median!");

    let mut new_median: (usize,f64) = (0,0.);


////////###### Warning table size hack, check against current index may be better!

    if (rank_table[feature].len() - removed.1) % 2 != 0 {

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
                let new_index = rank_table[feature][old_median.0].0;
                let new_value = rank_table[feature][new_index].3;
                (new_index,new_value)

            }
        }
    }

    println!("Computed new median!");
    println!("{:?}", new_median);

    new_median

}

fn closer_sample(left: (usize,usize,usize,f64,usize), right: (usize,usize,usize,f64,usize), median: (usize, f64), feature: usize, sorted_rank_table: & Vec<Vec<(usize,usize,usize,f64,usize)>>) -> (usize,usize,usize,f64,usize) {

    let inner_left = sorted_rank_table[feature][left.4];
    let inner_right = sorted_rank_table[feature][right.0];

    let mut possibilities = vec![left,right,inner_right,inner_left];

    possibilities.sort_unstable_by(|a,b| (a.3 - median.1).abs().partial_cmp(&(b.3 - median.1)).unwrap_or(Ordering::Greater));

    let closer = possibilities.pop().unwrap();

    closer
}

fn median_distance(sorted_rank_table:& Vec<Vec<(usize,usize,usize,f64,usize)>>, feature:usize, old_median_distance:(usize,f64,usize,f64), old_median:(usize,f64), new_median:(usize,f64), removed: (usize,usize,usize,f64,usize)) -> (usize,f64,usize,f64) {

    println!("Computing new median distance!");

    let mut new_median_distance = old_median_distance.clone();

    let change =  new_median.1 - old_median.1;

    let sample_space = sorted_rank_table[feature].len()-removed.1-1;

    let mut median_distance_ordered = vec![(old_median_distance.0,old_median_distance.1),(old_median_distance.2,old_median_distance.3)];
    median_distance_ordered.sort_unstable_by_key(|x| sorted_rank_table[feature][x.0].2);
    let (mut left_boundary, mut right_boundary) = (sorted_rank_table[feature][median_distance_ordered[0].0],sorted_rank_table[feature][median_distance_ordered[1].0]);

    if right_boundary.1 == removed.1 {
        right_boundary = closer_sample(left_boundary, removed, new_median, feature, sorted_rank_table);
    }

    if left_boundary.1 == removed.1 {
        left_boundary = closer_sample(removed, right_boundary, new_median, feature, sorted_rank_table);
    }


    if change > 0. {
        println!("Moving right!");
        for i in 0..10 {
            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            println!("New median: {:?}", new_median);
            match (sorted_rank_table[feature][right_boundary.4].3 - new_median.1).abs().partial_cmp(&(sorted_rank_table[feature][left_boundary.4].3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
                Ordering::Less => {
                    right_boundary = sorted_rank_table[feature][right_boundary.4];
                    left_boundary = sorted_rank_table[feature][left_boundary.4];
                },
                Ordering::Greater => break,
                Ordering::Equal =>{
                    right_boundary = sorted_rank_table[feature][right_boundary.4];
                    left_boundary = sorted_rank_table[feature][left_boundary.4];
                }
            }
            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
        }
    }
    if change < 0. {
        println!("Moving left!");
        for i in 0..10 {
            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
            println!("New median: {:?}", new_median);
            match (sorted_rank_table[feature][left_boundary.0].3 - new_median.1).abs().partial_cmp(&(sorted_rank_table[feature][right_boundary.0].3 - new_median.1).abs()).unwrap_or(Ordering::Greater) {
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
            println!("Boundaries: {:?},{:?}", left_boundary,right_boundary);
        }
    }


    let mut median_distance_disordered = vec![(right_boundary.1,(right_boundary.3 - new_median.1).abs()),(left_boundary.1,(left_boundary.3 - new_median.1).abs())];
    median_distance_disordered.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
    new_median_distance = (median_distance_disordered[1].0,median_distance_disordered[1].1, median_distance_disordered[0].0, median_distance_disordered[0].1);

    println!("{}", sample_space);
    println!("{:?}", median_distance_disordered);

    if sample_space % 2 == 0 {

        println!("Even distances, computing split median!");

        let distance_to_outer_left = (left_boundary.0, (sorted_rank_table[feature][left_boundary.0].3 - new_median.1).abs());

        let distance_to_outer_right = (right_boundary.4, (sorted_rank_table[feature][right_boundary.4].3 - new_median.1).abs());

        println!("Outer distances: {:?},{:?}", distance_to_outer_left, distance_to_outer_right);

        let temp_outer_median = vec![distance_to_outer_right,distance_to_outer_left];
        let outer_median = temp_outer_median.iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or(&(0,0.)).clone();

        println!("Outer median: {:?}", outer_median);

        new_median_distance.1 = (new_median_distance.1 + outer_median.1)/2.;

    }


    println!("Done computing new median distance!");

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
        let mut upper_distance_rank_table = Vec::new();
        for (i,(feature,median)) in upper_sorted_rank_table.iter().zip(current_upper_median.iter()).enumerate() {
            // upper_sorted_MSDM.push(feature.iter().map(|x|(x.1, x.3 - median.1)).collect());
            let description = madm_ranking(&upper_sorted_rank_table[i],&upper_sorted_samples[i], &current_upper_median[i]);
            upper_distance_rank_table.push(description.0);
            upper_sorted_MSDM.push(description.1);
        }




        let mut lower_sorted_counts = Vec::new();
        for feature in &upper_sorted_rank_table {
            lower_sorted_counts.push(Vec::new());
        }
        let mut lower_sorted_MSDM = Vec::new();
        for feature in &upper_sorted_MSDM {
            lower_sorted_MSDM.push(Vec::new());
        }
        let mut current_lower_median = Vec::new();
        for feature in &lower_sorted_counts {
            current_lower_median.push((0,0.));
        }

        OnlineMADM {
            counts : local_counts,
            dimensions: dimensions,
            current_index: 0,
            current_upper_median : current_upper_median,
            current_lower_median : current_lower_median,
            upper_sorted_rank_table : upper_sorted_rank_table,
            upper_sorted_samples : upper_sorted_samples,
            upper_distance_rank_table: upper_distance_rank_table,
            upper_sorted_MSDM : upper_sorted_MSDM,
            lower_sorted_counts : lower_sorted_counts,
            lower_sorted_MSDM : lower_sorted_MSDM
        }



    }

    fn test(&mut self) {

        println!("Testing the online madm!");
        println!("{:?}", self.current_upper_median);
        println!("{:?}", self.current_lower_median);
        println!("{:?}", self.dimensions);
        println!("{:?}", self.upper_sorted_rank_table);
        println!("{:?}", self.upper_distance_rank_table);
        println!("{:?}", self.upper_sorted_MSDM);
        println!("Computing a step!");
        self.next();
        println!("Exited next function");
        println!("{:?}", self.current_upper_median);
        println!("{:?}", self.upper_sorted_MSDM);
        for i in 0..6 {
            self.next();
            println!("{:?}", self.current_upper_median);
            println!("{:?}", self.upper_sorted_MSDM);

        }

        // println!("{:?}", self.upper_sorted_MSDM);
    }

    fn next(&mut self) -> ((Vec<f64>,Vec<f64>),(Vec<f64>,Vec<f64>)) {



        for i in 0..self.dimensions.1 {

            let current_sample = pop_rank_table(&mut self.upper_sorted_rank_table,(i,self.current_index));

            let new_median = rank_table_median(&self.upper_sorted_rank_table, &self.upper_sorted_samples, i, self.current_upper_median[i], current_sample);

            let new_median_distance = median_distance(&self.upper_sorted_rank_table, i, self.upper_sorted_MSDM[i], self.current_upper_median[i], new_median, current_sample);

            println!("{:?}", current_sample);
            println!("{:?}", self.upper_sorted_rank_table);
            println!("{:?}", new_median);
            println!("{:?}", new_median_distance);

            self.current_upper_median[i] = new_median;
            self.upper_sorted_MSDM[i] = new_median_distance;

            self.current_index += 1;
            //
            // self.current_upper_median[i] = new_median;

            // let where_to = insert_into_sorted(&mut self.lower_sorted_counts[i], (current_sample.1,current_sample.3), current_sample.2);
            //
            // insert_into_sorted(&mut self.lower_sorted_MSDM[i], (self.lower_sorted_counts[i][where_to].1,self.lower_sorted_counts[i][where_to].2-self.current_lower_median[i].1), self.lower_sorted_counts[i][where_to].0);





            // self.lower_sorted_MSDM.insert(where_to,(self.lower_sorted_counts[i].1self.lower_sorted_counts[i][where_to].2 - self.current_lower_median[i].1));

        }

        ((Vec::new(),Vec::new()),(Vec::new(),Vec::new()))
    }

}

struct OnlineMADM {
    counts: Vec<Vec<f64>>,
    dimensions: (usize,usize),
    current_index : usize,
    current_upper_median: Vec<(usize,f64)>,
    current_lower_median: Vec<(usize,f64)>,
    // upper_sorted_counts: Vec<Vec<(usize,f64)>>,

    upper_sorted_samples: Vec<Vec<usize>>,
    upper_sorted_rank_table: Vec<Vec<(usize,usize,usize,f64,usize)>>,

    upper_distance_rank_table: Vec<Vec<(usize,usize,usize,f64,usize)>>,

    upper_sorted_MSDM: Vec<(usize,f64,usize,f64)>,
    lower_sorted_counts: Vec<Vec<(usize,usize,f64)>>,
    lower_sorted_MSDM: Vec<Vec<(usize,usize,f64)>>,
}
