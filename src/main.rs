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

mod online_madm;
use online_madm::OnlineMADM;
use online_madm::slow_vs_fast;
use online_madm::slow_description;
use online_madm::slow_description_test;

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
