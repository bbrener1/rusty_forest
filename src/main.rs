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
use std::thread::sleep;
use std::time;
use std::cell::Cell;
extern crate rand;
use rand::Rng;
use rand::seq;
use std::collections::HashSet;



mod online_madm;
mod raw_vector;
mod rank_vector;
mod rank_table;
mod node;
mod tree;
mod thread_pool;
mod random_forest;


use node::Node;
use tree::Tree;
use rank_table::RankTable;
use rank_vector::RankVector;
use rank_vector::OrderedDraw;
use raw_vector::RawVector;
use online_madm::OnlineMADM;
use online_madm::slow_vs_fast;
use online_madm::slow_description;
use online_madm::slow_description_test;

fn main() {

    // let args:Vec<String> = env::args().collect();

    let args = Arguments::new(&mut env::args());

    let mut feature_names = None;
    let mut sample_names = None;

    if args.feature_header_file.is_some() {
        feature_names = Some(read_header(args.feature_header_file.clone().unwrap()));
    }

    if args.sample_header_file.is_some() {
        sample_names = Some(read_sample_names(args.sample_header_file.clone().unwrap()));
    }

    println!("Argumnets parsed: {:?}", args);

    println!("Reading data");

    let mut count_array: Vec<Vec<f64>> = read_counts(args.count_array_file.clone());

    let report_address = &args.report_address;

    println!("##############################################################################################################");
    println!("##############################################################################################################");
    println!("##############################################################################################################");



    // let names: Vec<String> = (0..count_array[0].len()).map(|x| x.to_string()).collect();
    // let samples: Vec<String> = (0..count_array.len()).map(|x| x.to_string()).collect();

    // let medium_case = vec![vec![-1.,0.,-2.,10.,-3.,-4.,-20.,15.,20.,25.,100.]];
    //
    // let simple_case = vec![vec![0.,-1.,0.,-2.,10.,-3.,15.,20.]];
    //

    // let mut rng = rand::thread_rng();
    // let input_features = rand::seq::sample_iter(&mut rng, names.clone(), 1000).expect("Couldn't generate input features");

    // let mut tree = Tree::plant_tree(&matrix_flip(&count_array),&names.clone(),&samples.clone(),names.clone(),names.clone(), 20);
    // let mut parallel_tree = Tree::plant_tree(&matrix_flip(&count_array),&names.clone(),&samples.clone(),input_features,names.clone(), 100);
    //
    // parallel_tree.grow_branches();

    let mut rnd_forest = random_forest::Forest::initialize(count_array, args.tree_limit, args.leaf_size_cutoff,args.processor_limit, feature_names, sample_names, report_address);
    rnd_forest.generate(400,800);

}

#[derive(Debug)]
pub struct Arguments {

    count_array_file: String,
    feature_header_file: Option<String>,
    sample_header_file: Option<String>,
    report_address: String,

    processor_limit: usize,
    tree_limit: usize,
    leaf_size_cutoff: usize,
    drop: bool,

}

impl Arguments {
    fn new(args: &mut env::Args) -> Arguments {

        let mut arg_struct = Arguments {
                    count_array_file: "".to_string(),
                    feature_header_file: None,
                    sample_header_file: None,
                    report_address: "./".to_string(),

                    processor_limit: 1,
                    tree_limit: 1,
                    leaf_size_cutoff: 10000,
                    drop: true,
        };


        // let mut current_arg = "";
        // let mut current_arg_vec = Vec::new();
        while let Some(arg) = args.next() {
            match &arg[..] {
                "-c" | "-counts" => {
                    arg_struct.count_array_file = args.next().expect("Error parsing count location!");
                },
                "-p" | "-processors" | "-threads" => {
                    arg_struct.processor_limit = args.next().expect("Error processing processor limit").parse::<usize>().expect("Error parsing processor limit");
                },
                "-o" | "-output" => {
                    arg_struct.report_address = args.next().expect("Error processing output destination")
                },
                "-t" | "-trees" => {
                    arg_struct.tree_limit = args.next().expect("Error processing tree count").parse::<usize>().expect("Error parsing tree count");
                },
                "-l" | "-leaves" => {
                    arg_struct.leaf_size_cutoff = args.next().expect("Error processing leaf limit").parse::<usize>().expect("Error parsing leaf limit");
                },
                "-nd" | "-no_drop" => {
                    arg_struct.drop = false;
                },
                &_ => {
                    println!("Warning, detected unexpected arguments, but so far nothing is wrong");
                }

            }
        }

        arg_struct

    }
}

// fn parse_args(arguments: env::Args) -> Arguments

fn read_counts(location:String) -> Vec<Vec<f64>> {

    let count_array_file = File::open(location).expect("File error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    let mut count_array: Vec<Vec<f64>> = Vec::new();

    for (i,line) in count_array_lines.by_ref().enumerate() {

        let mut gene_vector = Vec::new();

        let gene_line = line.expect("Readline error");

        for (j,gene) in gene_line.split_whitespace().enumerate() {

            if j == 0 && i%200==0{
                print!("\n");
            }

            if i%200==0 && j%200 == 0 {
                print!("{} ", gene.parse::<f64>().unwrap_or(-1.) );
            }

            // if !((gene.0 == 1686) || (gene.0 == 4660)) {
            //     continue
            // }

            match gene.parse::<f64>() {
                Ok(exp_val) => {

                    gene_vector.push(exp_val);

                },
                Err(msg) => {
                    println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                    println!("Cell content: {:?}", gene);
                    gene_vector.push(0.);
                }
            }

        }

        count_array.push(gene_vector);

        if i % 100 == 0 {
            println!("{}", i);
        }


    };

    println!("===========");
    println!("{},{}", count_array.len(),count_array[0].len());

    matrix_flip(&count_array)

}

fn read_header(location: String) -> Vec<String> {

    let mut header_vector = Vec::new();

    let header_file = File::open(location).expect("Header file error!");
    let mut header_file_iterator = io::BufReader::new(&header_file).lines();

    for line in header_file_iterator.by_ref() {
        header_vector.push(line.expect("Error reading header line!").to_string());
    }

    header_vector
}

fn read_sample_names(location: String) -> Vec<String> {

    let mut header_vector = Vec::new();

    let count_array_file = File::open(location).expect("Sample name file error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    for line in count_array_lines.by_ref() {
        header_vector.push(line.expect("Error reading header line!").trim().to_string())
    }

    header_vector
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


/////TESTING CODE///////

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
// let rng = thr_rng.gen_iter::<f64>();
// let temp5: Vec<f64> = rng.take(49).collect();
// let temp6 = matrix_flip(&(vec![temp5.clone()]));

// axis_sum_test.push(vec![1.,2.,3.]);
// axis_sum_test.push(vec![4.,5.,6.]);
// axis_sum_test.push(vec![7.,8.,9.]);

// println!("Source floats: {:?}", matrix_flip(&counts));

// println!("{:?}", count_array);
//
// let mut raw = RawVector::raw_vector(&matrix_flip(&count_array)[0]);
//
// println!("{:?}",raw);
//
// println!("{:?}", raw.iter_full().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Crawlers:");
//
// println!("{:?}", raw.crawl_right(raw.first).cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("{:?}", raw.crawl_left(raw.first).cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Dropping zeroes:");
//
// raw.drop_zeroes();
//
// println!("Crawling dropped list:");
//
// println!("{:?}", raw.crawl_right(raw.first).cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Skipping dropped items:");
//
// println!("{:?}", raw.drop_skip().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Printing non-zero values");
//
// println!("{:?}", raw.drop_skip().cloned().map(|x| x.3).collect::<Vec<f64>>());
//
// println!("Printing non-zero indecies");
//
// println!("{:?}", raw.drop_skip().cloned().map(|x| x.1).collect::<Vec<usize>>());
//
// println!("Printing noned-out drops");
// for i in raw.drop_none() {
//     println!("{:?}",i);
// }
//
// println!("Skipping drops");
// for i in raw.drop_skip() {
//     println!("{:?}",i);
// }
//
// println!("{:?}",raw.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Finding dead center:");
//
// let dead_center = rank_vector::DeadCenter::center(&raw);
//
// println!("{:?}", dead_center);
//
// println!("{:?}", dead_center.median());

// println!("=================================================================");

// println!("Indecies: {:?}", matrix_flip(&count_array)[0]);
//
// println!("Testing Ranked Vector!");
//
// let degenerate_case = vec![0.;10];
//
// let mut ranked: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[0],String::from("test"), Vec::new());
//
// ranked.drop_zeroes();
//
// ranked.initialize();
//
// println!("Dropped values, ranked vector");
//
// println!("{:?}", ranked.vector.drop_skip().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("{:?}", ranked.clone());
//
// ranked.set_boundaries();
//
// println!("{:?}", ranked.clone());
//
// println!("{:?},{:?},{:?},{:?},{:?},{:?}", ranked.left_zone.size,ranked.left_zone.index_set.len(),ranked.median_zone.size,ranked.median_zone.index_set.len(),ranked.right_zone.size,ranked.right_zone.index_set.len());
//
// let ranked_clone = ranked.clone();
//
// {
//     let ordered_draw = OrderedDraw::new(&mut ranked);
//
//     println!("{:?}", ordered_draw.vector.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
//     for sample in ordered_draw {
//         println!("Drawing: {:?}", sample);
//     }
// }
//
// println!("Dumping ranked vector:");
//
// let mut backup_debug_file = File::create("ranked_vec_debug.txt").unwrap();
// backup_debug_file.write_fmt(format_args!("{:?}", ranked));
//
// println!("Dumping ranked clone:");
//
// let mut backup_debug_file = File::create("ranked_vec_clone.txt").unwrap();
// backup_debug_file.write_fmt(format_args!("{:?}", ranked_clone));

// println!("Trying to make a rank table:");
//
// // let mut table = RankTable::new(simple_case,&names,&samples);
//
// println!("{},{}",count_array.len(),count_array[0].len());
//
// let mut table = RankTable::new(matrix_flip(&count_array),&names,&samples);
//
// println!("Finished making a rank table, trying to iterate:");
//
// let mut variance_table = Vec::new();
//
// for (j,i) in table.split(String::from("Test")).0.enumerate() {
//     // variance_table.push(vec![i[0].1/i[0].0,i[1].1/i[1].0,i[2].1/i[2].0,i[3].1/i[3].0]);
//     variance_table.push(vec![i[0].1/i[0].0]);
//     println!("{},{:?}",j,i)
// }
//
// println!("Variance table:");



// let minimal = variance_table.iter().map(|x| x.clone().sum()/(x.len() as f64)).enumerate().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//
// println!("Minimal split is: {:?}", minimal);

// let mut node = Node::root(&vec![matrix_flip(&count_array)[1].clone()],&names,&samples,names[..1].iter().cloned().collect(),names[1..].iter().cloned().collect());

// let mut node = Node::root(&matrix_flip(&count_array),&names,&samples,names[..1].iter().cloned().collect(),names[1..].iter().cloned().collect());

// let mut node = Node::root(&matrix_flip(&count_array),&names.clone(),&samples.clone(),names.clone(),names.clone());

// let mut node = Node::root(&simple_case,&names,&samples,names[..1].iter().cloned().collect(),names[1..].iter().cloned().collect());


//
// println!("{:?}",node.rank_table.sort_by_feature(0));

// node.parallel_derive();
//
// for child in node.children.iter_mut() {
//     child.derive_children();
// }

// tree.test_splits();
// parallel_tree.test_parallel_splits();



// let mut forest = Forest::grow_forest(count_array, 1, 4, true);
// forest.test();

// println!("Inner axist test! {:?}", inner_axis_mean(&axis_sum_test));
// println!("Matrix flip test! {:?}", matrix_flip(&axis_sum_test));

// slow_description_test();
// slow_vs_fast();


// let mut ranked1: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[0],String::from("test"), Vec::new());
// let mut ranked2: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[1],String::from("test"), Vec::new());
// let mut ranked3: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[2],String::from("test"), Vec::new());
// let mut ranked4: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[3],String::from("test"), Vec::new());
//
// ranked1.drop_zeroes();
// ranked2.drop_zeroes();
// ranked3.drop_zeroes();
// ranked4.drop_zeroes();
//
//
// ranked1.initialize();
// ranked2.initialize();
// ranked3.initialize();
// ranked4.initialize();
//
// ranked1.set_boundaries();
// ranked2.set_boundaries();
// ranked3.set_boundaries();
// ranked4.set_boundaries();
//
// {
//     let ordered_draw = OrderedDraw::new(&mut ranked1);
//
//     println!("{:?}", ordered_draw.vector.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
//     for sample in ordered_draw {
//         println!("Drawing: {:?}", sample);
//     }
// }
