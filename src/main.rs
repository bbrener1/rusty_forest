#![allow(dead_code)]
use std::env;
use std::io;
use std::sync::Arc;
use std::f64;
// use std::rc::Weak;
use std::cmp::min;
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
use std::cell::Cell;
#[macro_use] extern crate rustml;
extern crate rand;
use rand::Rng;
use std::collections::HashSet;


extern crate gnuplot;

mod online_madm;
use online_madm::OnlineMADM;
use online_madm::slow_vs_fast;
use online_madm::slow_description;
use online_madm::slow_description_test;

fn main() {

    let filename = "/Users/boris/taylor/vision/rust_prototype/raw_data/counts.txt";

    println!("Reading data");

    let count_array_file = File::open(filename).expect("File error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    let mut count_array: Vec<Vec<f64>> = Vec::new();

    for line in count_array_lines.by_ref().enumerate() {

        count_array.push(Vec::new());

        let gene_line = line.1.expect("Readline error");

        for gene in gene_line.split_whitespace().enumerate() {

            // println!("{}", gene.0);
            //
            // println!("{}", gene.1.parse::<f64>().unwrap() );

            if !((gene.0 == 1686) || (gene.0 == 4660)) {
                continue
            }

            match gene.1.parse::<f64>() {
                Ok(exp_val) => {
                    match count_array.last_mut() {
                        Some(last_vec) => last_vec.push(exp_val),
                        None => {
                            println!("0th Dimension of count array empty!");
                            panic!("Check count array creation rules or input file!")
                        }
                    }

                },
                Err(msg) => {
                    println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                    match count_array.last_mut() {
                        Some(last_vec) => last_vec.push(0f64),
                        None => {
                            println!("0th Dimension of count array empty! ln 31");
                            panic!("Check count array creation rules or input file!")
                        }
                    }
                }
            }
        }

            // if let Result::Ok(exp_val) = gene.1.parse::<f64>() {
            //     match count_array.last() {
            //         Some(last_vec) => last_vec.push(exp_val),
            //         None => None
            //     }

        if line.0 % 100 == 0 {
            println!("{}", line.0);
        }

        // count_array.push(
        //     gene_line.split_whitespace()
        //         .map(|x| {x.parse::<f64>().unwrap()}).collect::<Vec<f64>>());



        if line.0%100 == 0 && line.0 > 0 {

        //     println!("===========");
            println!("{}",line.0);
        //     println!("{:?}", &count_array.iter().take(3).map(|x| x.iter().take(3)));
        }

    };

    println!("===========");
    //
    // for i in 0..100 {
    //     for j in 0..100 {
    //         print!("{}, ", count_array[i][j]);
    //     }
    //     print!("\n");
    // }
    //
    // let model = OnlineMADM::new(count_array);
    //
    // for i in 0..100 {
    //     for j in 0..100 {
    //         print!("{}, ", model.counts[i][j]);
    //     }
    //     print!("\n");
    // }
    //
    // for i in 0..10 {
    //     for j in 0..10 {
    //         print!("{:?}, ", model.upper_sorted_rank_table[i][j]);
    //     }
    //     print!("\n");
    // }
    //
    // for i in 0..10 {
    //     for j in 0..10 {
    //         print!("{:?}, ", model.dispersion_history[i][j]);
    //     }
    //     print!("\n");
    // }


    // let temp = tree {nodes: Vec::new(),counts: Vec::new()};
    // for cell in &mut count_array.into_iter() {
    //     for gene in &mut cell.into_iter() {
    //         println!("Test? {}", gene);
    //     }
    // }

    // let mut forest = Forest::grow_forest(count_array, 10, 1000);
    // forest.test();
    //
    // println!("Argmin test: {},{}", argmin(&vec![1.,3.,0.,5.]).0,argmin(&vec![1.,3.,0.,5.]).1);
    //
    // println!("Argsort test: {:?}", argsort(&vec![1.,3.,0.,5.]));
    //
    // let mut axis_sum_test: Vec<Vec<f64>> = Vec::new();

    println!("##############################################################################################################");
    println!("##############################################################################################################");
    println!("##############################################################################################################");


    // axis_sum_test.push(vec![1.,2.,3.]);
    // axis_sum_test.push(vec![4.,5.,6.]);
    // axis_sum_test.push(vec![0.,1.,0.]);
    // let temp: [f64;7] = [-3.,-2.,-1.,0.,10.,15.,20.];
    // let temp2 = temp.into_iter().cloned().collect();
    // let temp3 = vec![temp2];
    // let temp4 = matrix_flip(&temp3);
    //
    //
    // let mut thr_rng = rand::thread_rng();
    //
    // let mut counts = Vec::new();
    //
    // let mut rng = thr_rng.gen_iter::<f64>();
    //
    // for feature in 0..10 {
    //
    //     println!("{:?}",counts);
    //
    //     counts.push(Vec::new());
    //
    //     let loc_f = counts.last_mut().unwrap();
    //
    //     for sample in 0..100 {
    //         if rng.next().unwrap() < 0.7 {
    //             loc_f.push(rng.next().unwrap());
    //         }
    //         else {
    //             loc_f.push(0.0);
    //         }
    //     }
    //         // counts.push(rng.take(6).collect());
    //
    // }

    // println!("{:?}",counts);
    //
    //
    // counts = matrix_flip(&counts);
    //
    // let model = OnlineMADM::new(counts.clone(),true);
    //
    // let second_model = OnlineMADM::new(counts.clone(),false);
    //
    // println!("{:?}", model);
    // println!("{:?}", second_model);

    // let temp6 = matrix_flip(&counts);

    // let mut thr_rng = rand::thread_rng();
    // let rng = thr_rng.gen_iter::<f64>();
    // let temp5: Vec<f64> = rng.take(49).collect();
    // let temp6 = matrix_flip(&(vec![temp5.clone()]));

    // axis_sum_test.push(vec![1.,2.,3.]);
    // axis_sum_test.push(vec![4.,5.,6.]);
    // axis_sum_test.push(vec![7.,8.,9.]);

    // println!("Source floats: {:?}", matrix_flip(&counts));

    println!("{:?}", count_array);

    let mut forest = Forest::grow_forest(count_array, 1, 2, true);
    forest.test();

    // println!("Inner axist test! {:?}", inner_axis_mean(&axis_sum_test));
    // println!("Matrix flip test! {:?}", matrix_flip(&axis_sum_test));

    // slow_description_test();
    // slow_vs_fast();

}




impl Forest {



    fn grow_forest(counts : Vec<Vec<f64>>, forest_size: usize, tree_features: usize, dropout: bool) -> Forest {

        let count_size1:usize = counts.len();
        let count_size2 = {&counts[0]}.len();

        let mut forest = Forest{dropout: dropout, trees:Vec::new(),counts: Arc::new(counts),count_size:(count_size1,count_size2),tree_features:tree_features};

        for n in 0..forest_size {
            let in_features = rand::sample(&mut rand::thread_rng(), 0..count_size2, tree_features);
            let out_features = {0..count_size2}.collect::<Vec<usize>>();
            let tree_vector : &mut Vec<Tree> = forest.trees.borrow_mut();
            tree_vector.push(Tree::plant_tree(in_features,out_features,forest.counts.clone(),forest.dropout))
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

        println!("Dumping the root node:");

        let mut backup_debug_file = File::create("root_node_debug.txt").unwrap();
        backup_debug_file.write_fmt(format_args!("{:?}", self.trees[0].root.madm.0));

        // self.trees.push(Tree::plant_tree(vec![627],vec![3964],self.counts.clone()));

        // let split_feature = 627;
        //
        // let split = self.trees[0].root.find_split(split_feature);
        // println!("Best split for feature {} is found to be {},{}", split_feature, split.0,split.1);

        self.trees[0].root.best_split();
        // for child in &mut self.trees[0].root.children {
        //     child.best_split();
        // }

        // let split = self.trees[0].nodes[0].find_split(split_feature);
        // println!("Trying to find split again: {},{}", split.0,split.1);
        //
        // let split = self.trees[0].nodes[0].find_split(1);
        // println!("Splitting by the second feature: {},{}", split.0,split.1);
        //
        // let split = self.trees[0].nodes[0].find_split(2);
        // println!("Splitting by the third feature: {},{}", split.0,split.1);
        //
        // let split = self.trees[0].nodes[0].find_split(split_feature);
        // println!("Best split for feature {} is found to be {},{}", split_feature, split.0,split.1);


    }
}

struct Forest {
    dropout: bool,
    trees: Vec<Tree>,
    count_size: (usize,usize),
    counts: Arc<Vec<Vec<f64>>>,
    tree_features: usize
}


impl Tree {

    fn plant_tree(in_f:Vec<usize>,out_f:Vec<usize>,f:Arc<Vec<Vec<f64>>>, dropout: bool) -> Tree {

        let root = Node::first(Arc::downgrade(&f),(0..f.len()).collect(),in_f.clone(),out_f.clone(),dropout);

        println!("Exited the root node function, planting tree!");

        let mut tree = Tree {
            dropout: dropout,
            nodes: Vec::new(),
            root: root.0,
            input_features:Arc::new(in_f),
            // input_samples:Arc::new(Vec::new()),
            input_samples:(0..f.len()).collect(),
            output_features:Arc::new(out_f),
            weights:Arc::new(Vec::new()),
            counts:Arc::downgrade(&f)};
        tree.nodes.push(root.1);

        println!("Tree planted!");

        tree
    }


}

struct Tree {
    dropout: bool,
    nodes: Vec<Weak<Node>>,
    root: Node,
    input_features: Arc<Vec<usize>>,
    input_samples: Vec<usize>,
    output_features: Arc<Vec<usize>>,
    weights: Arc<Vec<f64>>,
    counts: Weak<Vec<Vec<f64>>>
}

impl Node {

    fn first(counts:Weak<Vec<Vec<f64>>>, samples:Vec<usize>, input_features:Vec<usize>, output_features:Vec<usize>, dropout: bool) -> (Node,Weak<Node>) {

        let out_f_set : HashSet<usize> = output_features.iter().cloned().collect();

        let mut weights = vec![1.;counts.upgrade().expect("Empty counts at node creation!")[0].len()];

        let mut loc_counts: Vec<Vec<f64>> = samples.iter().cloned().map(|x| counts.upgrade().expect("Dead tree!")[x].clone()).collect();

        loc_counts = matrix_flip(&loc_counts).iter().cloned().enumerate().filter(|x| out_f_set.contains(&x.0)).map(|y| y.1).collect();

        loc_counts = matrix_flip(&loc_counts);

        let fmadm = OnlineMADM::new(loc_counts.clone(),true);

        let medians = fmadm.median_history[0].clone();
        let dispersion = fmadm.dispersion_history[0].iter().enumerate().map(|(i,x)| {
            x.1/medians[i].1
        }).collect();

        let rmadm = fmadm.reverse();

        let madm = (fmadm,rmadm);

        println!("Finished computing a root node!");

        // let means: Vec<f64> = loc_counts.iter().fold(vec![0f64;loc_counts[0].len()], |acc, x|
        //     {
        //         x.iter().zip(acc.iter()).map(|y| y.0 + y.1).collect::<Vec<f64>>()
        //     }).iter().map(|z| z/(loc_counts.len() as f64)).collect();

        // let variance = loc_counts.iter().fold(vec![0f64;loc_counts[0].len()], |acc,x| {
        //         x.iter().enumerate().zip(acc.iter()).map(|y| ((y.0).1 - medians[(y.0).0]).powi(2) + y.1).collect()
        //     }
        //     ).iter().map(|z| z/(loc_counts.len() as f64)).collect();



        let mut result = Node {
            dropout: dropout,
            selfreference: Cell::new(None),
            feature:None,
            split: None,
            medians: medians,
            output_features: output_features,
            input_features: input_features,
            indecies: samples,
            dispersion:dispersion,
            weights: weights,
            children:Vec::new(),
            parent: Cell::new(None),
            counts:counts,
            madm:madm
        };

        // println!("Dumping feature information to disk");
        //
        // let mut backup_debug_file = File::create("root_node_debug.txt").unwrap();
        // backup_debug_file.write_fmt(format_args!("{:?}", result.madm.0));

        println!("Root node object complete, setting up references!");

        let mut res_arc = Arc::new(result);

        let  res_weak = Arc::downgrade(&res_arc);

        // let res_ref: &Node = res_arc.borrow();

        res_arc.selfreference.set(Some(res_weak.clone()));

        match Arc::try_unwrap(res_arc) {
            Ok(node) => return (node,res_weak),
            Err(arc) => panic!("Failed to unwrap a root node, something went wrong at tree construction")
        }

    }

    fn derive(&mut self, samples: Vec<usize>) -> Weak<Node> {

        let mut weights = vec![1.;self.counts.upgrade().expect("Empty counts at node creation!")[0].len()];

        let fmadm = self.madm.0.derive_subset(samples.clone());

        let medians = fmadm.median_history[0].clone();
        let dispersion = fmadm.dispersion_history[0].iter().enumerate().map(|(i,x)| {
            x.1/medians[i].1
        }).collect();

        let rmadm = self.madm.1.derive_subset(samples.clone());

        let madm = (fmadm,rmadm);

        println!("Derived rank tables!");

        let child = Node{
            dropout: self.dropout,
            selfreference: Cell::new(None),
            feature: None,
            split: None,
            output_features: self.output_features.clone(),
            input_features: self.input_features.clone(),
            indecies: samples,
            medians: medians,
            weights: weights,
            dispersion: dispersion,
            children: Vec::new(),
            parent: Cell::new(None),
            counts: self.counts.clone(),
            madm: madm
        };

        println!("Constructed a child object, trying to insert labels!");
        // println!("{:?}", self.selfreference);

        let self_weak = self.selfreference.take().unwrap();

        child.parent.set(Some(self_weak.clone()));

        self.selfreference.set(Some(self_weak.clone()));

        let arc_child = Arc::new(child);

        let weak_child = Arc::downgrade(&arc_child.clone());

        match Arc::try_unwrap(arc_child) {
            Ok(child) => {self.children.push(child); return weak_child},
            Err(error) => panic!("Failed to derive a node, pointers broken!")
        }

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

        println!("Finding split: {}", feature);

        println!("{:?}", self.madm.0.sorted_rank_table[feature]);
        println!("{},{},{},{}", self.madm.0.left_zone[feature],self.madm.0.median_zone[feature],self.madm.0.right_zone[feature],self.indecies.len());

        let weight_backup = self.weights[feature];
        self.weights[feature] = 0.;

        let draw_order = self.madm.0.sort_by_feature(feature);
        self.madm.1.sort_by_feature(feature);
        self.madm.1.reverse_draw_order();

        let mut forward_dispersions = Vec::new();

        let mut drawn_samples = Vec::new();

        // println!("{:?}" ,self.madm.0);
        // println!("{:?}" ,self.madm.1);

        while let Some(x) = self.madm.0.next() {

            if self.dropout && self.madm.1.sorted_rank_table[feature][x.2].3 == 0.
            {
                continue
            }

            let mut individual_dispersions = Vec::new();

            // for (i,(med,disp)) in x.0.iter().zip(x.1.iter()).enumerate() {
            //     individual_dispersions.push((disp.1/med.1)*self.weights[i]);
            // }
            //
            // forward_dispersions.push(individual_dispersions.iter().sum::<f64>() / self.weights.iter().sum::<f64>());

            for (i,(med,disp)) in x.0.iter().zip(x.1.iter()).enumerate() {
                individual_dispersions.push((disp.1/med.1).powi(2)*self.weights[i]);
            }

            forward_dispersions.push(individual_dispersions.sum().sqrt());


            if forward_dispersions.len()%150 == 0 {
                println!("{}", forward_dispersions.len());
            }

            drawn_samples.push(x.2);

        }

        println!("Forward split found");

        let mut reverse_dispersions = Vec::new();

        // println!("{:?}" ,self.madm.1);

        while let Some(x) = self.madm.1.next() {

            if self.dropout && self.madm.1.sorted_rank_table[feature][x.2].3 == 0.
            {
                continue
            }

            let mut individual_dispersions = Vec::new();

            for (i,(med,disp)) in x.0.iter().zip(x.1.iter()).enumerate() {
                individual_dispersions.push((disp.1/med.1).powi(2)*self.weights[self.weights.len()-(i+1)]);
            }

            // println!("{:?}",individual_dispersions);

            reverse_dispersions.push(individual_dispersions.sum().sqrt());

        }

        println!("Reverse split found");

        reverse_dispersions = reverse_dispersions.iter().cloned().rev().collect();

        // for (i,(fw,rv)) in forward_dispersions.iter().zip(reverse_dispersions.clone()).enumerate() {
        //     if i%100 == 0 {println!("fw/rv: {},{}",fw,rv);}
        // }

        let mut minimum = (0,f64::INFINITY, 0,0);
        // let mut individual_minima = vec![(0,f64::INFINITY);forward_dispersions[0].len()];

        for (i,(fw,rv)) in forward_dispersions.iter().zip(reverse_dispersions.clone()).enumerate() {

            if self.madm.0.sorted_rank_table[feature][drawn_samples[i]].3 == 0. {
                continue
            }

            minimum.3 += 1;

            let proportion = (i as f64) / (forward_dispersions.len() as f64);


            let f_adj_disp = fw * (1. - proportion);
            let r_adj_disp = rv * proportion;

            if (0 < i) && (i < (forward_dispersions.len()-1)) {
                if minimum.1 > f_adj_disp + r_adj_disp {
                    minimum = (drawn_samples[i],f_adj_disp + r_adj_disp, i, minimum.3);
                }
            }

        }


        println!("Feature: {}", feature);
        println!("{:?}", minimum);
        println!("Split rank: {}, Split value: {}", self.madm.0.sorted_rank_table[feature][minimum.0].2, self.madm.0.sorted_rank_table[feature][minimum.0].3);


        self.madm.0.reset();
        self.madm.1.reset();

        self.weights[feature] = weight_backup;


        (minimum.0, minimum.1)
    }

    fn best_split(&mut self) -> (usize,(usize,f64)) {

        println!("Trying to find the best split!");

        let mut minima = Vec::new();

        for feature in self.input_features.clone() {
            minima.push((feature,self.find_split(feature)));
        }

        let best_split = minima.iter().min_by(|x,y| (x.1).1.partial_cmp(&(y.1).1).unwrap_or(Ordering::Greater)).unwrap();

        println!("Best split: {:?}", best_split);

        // best_split.unwrap().clone();

        let mut left_split = Vec::new();
        let mut right_split = Vec::new();

        println!("Deriving child nodes:");

        let comparator = self.madm.0.sorted_rank_table[best_split.0][(best_split.1).0];

        for sample in self.madm.0.sorted_rank_table[best_split.0].iter().cloned() {

            println!("{:?}", comparator);
            println!("{:?}", sample);

            if (self.dropout && sample.3 != 0.) || !self.dropout
            {
                if sample.3 < comparator.3 {
                    println!("Left");
                    left_split.push(sample.1);
                    // println!("{:?}",left_split.len());
                }
                else {
                    println!("Right");
                    right_split.push(sample.1);
                    // println!("{:?}",right_split.len());
                }
            }
            else {
                println!("Drop");
            }
        }

        self.feature = Some(best_split.0);
        self.split = Some((best_split.1).1);

        println!("Finished computing child indecies:");

        println!("{:?}", left_split);
        println!("{:?}", right_split);

        println!("Synthesizing new nodes raw:");

        let left_raw_node = Node::first(self.counts.clone(), left_split.clone(), self.input_features.clone(), self.output_features.clone(), self.dropout.clone());

        println!("Dumping the left raw node:");

        let mut left_raw_file = File::create("left_raw_node.txt").unwrap();
        left_raw_file.write_fmt(format_args!("{:?}", left_raw_node.0.madm.0));


        let right_raw_node = Node::first(self.counts.clone(), right_split.clone(), self.input_features.clone() , self.output_features.clone(), self.dropout.clone());

        println!("Dumping the right raw node:");

        let mut right_raw_file = File::create("right_raw_node.txt").unwrap();
        right_raw_file.write_fmt(format_args!("{:?}", right_raw_node.0.madm.0));

        println!("Done dumping raw nodes");

        self.derive(left_split);
        self.derive(right_split);

        println!("Dumping the derived nodes:");

        let mut left_derived_file = File::create("left_derived_node.txt").unwrap();
        left_derived_file.write_fmt(format_args!("{:?}", self.children[0].madm.0));

        let mut right_derived_file = File::create("right_derived_node.txt").unwrap();
        right_derived_file.write_fmt(format_args!("{:?}", self.children[1].madm.0));


        println!("Child indecies:");

        for child in &self.children {
            println!("{:?}" , child.indecies)
        }

        println!("Found best split, derived children");
        println!("{:?}",best_split);

        *best_split
    }

}

struct Node {
    dropout: bool,
    selfreference: Cell<Option<Weak<Node>>>,
    feature: Option<usize>,
    split: Option<f64>,
    output_features: Vec<usize>,
    input_features: Vec<usize>,
    indecies: Vec<usize>,
    medians: Vec<(usize,f64)>,
    weights: Vec<f64>,
    dispersion: Vec<f64>,
    children: Vec<Node>,
    parent: Cell<Option<Weak<Node>>>,
    counts: Weak<Vec<Vec<f64>>>,
    madm: (OnlineMADM,OnlineMADM)
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
