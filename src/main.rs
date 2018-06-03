// #![feature(test)]

#[macro_use]
extern crate serde_derive;


use std::env;
use std::io;
use std::f64;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::stdin;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;

extern crate serde;
extern crate serde_json;
extern crate num_cpus;
extern crate rand;
extern crate time;
extern crate smallvec;
// extern crate test;


mod raw_vector;
mod rank_vector;
mod rank_table;
mod node;
mod tree;
mod feature_thread_pool;
mod predict_thread_pool;
mod tree_thread_pool;
mod random_forest;
mod predictor;
mod shuffler;
mod compact_predictor;
mod boosted_forest;
mod boosted_tree_thread_pool;
mod additive_booster;
mod rv2;
mod rv3;
// mod rv4;

use tree::PredictiveTree;
use random_forest::Forest;
use boosted_forest::BoostedForest;
use additive_booster::AdditiveBooster;

/// Author: Boris Brenerman
/// Created: 2017 Academic Year, Johns Hopkins University, Department of Biology, Taylor Lab

/// This is a forest-based regression/classification software package designed with single-cell RNAseq data in mind.
///
/// Currently implemented features are to generate Decision Trees that segment large 2-dimensional matrices, and prediction of samples based on these decision trees
///
/// Features to be implemented include interaction analysis, python-based node clustering and trajectory analysis using minimum spanning trees of clusters, feature correlation analysis, and finally subsampling-based gradient boosting for sequential tree generation.

/// The general structure of the program is as follows:
///
/// The outer-most class is the Random Forest
///


/// Random Forests:
///
/// Random Forest contains:
///     - The matrix to be analyzed
///     - Decision Trees
///
///     - Important methods:
///         - Method that generates decision trees and calls on them to grow branches
///         - Method that generates predicted values for a matrix of samples
///
///



/// Trees:
///
/// Trees contain:
///     - Root Node
///     - Feature Thread Pool Sender Channel
///     - Drop Mode
///
/// Each tree contains a subsampling of both rows and columns of the original matrix. The subsampled rows and columns are contained in a root node, which is the only node the tree has direct access to.
///

/// Feature Thread Pool:
///
/// Feature Thread Pool contains:
///     - Worker Threads
///     - Reciever Channel for jobs
///
///     - Important methods:
///         - A wrapper method to compute a set of medians and MADs for each job passed to the pool. Core method logic is in Rank Vector
///
/// Feature Thread Pools are containers of Worker threads. Each pool contains a multiple in, single out channel locked with a Mutex. Each Worker contained in the pool continuously requests jobs from the channel. If the Mutex is unlocked and has a job, a Worker thread receives it.
///
///     Jobs:
///         Jobs in the pool channel consist of a channel to pass back the solution to the underlying problem and a freshly spawned Rank Vector (see below). The job consists of calling a method on the RV that consumes it and produces the medians and Median Absolute Deviations (MAD) from the Median of the vector if a set of samples is removed from it in a given order. This allows us to determine what the Median Absolute Deviation from the Median would be given the split of that feature by some draw order. The draw orders given to each job are usually denoting that the underlying matrix was sorted by another feature.
///
/// Worker threads are simple anonymous threads kept in a vector in the pool, requesting jobs on loop from the channel.

fn main() {

    // manual_testing::test_command_predict_full();


    let mut arg_iter = env::args();

    let command_literal = arg_iter.next();

    let command = Command::parse(&arg_iter.next().unwrap());

    let mut parameters = Parameters::read(&mut arg_iter);

    parameters.command = command;

    match parameters.command {
        Command::Construct => construct(parameters),
        Command::Predict => predict(parameters),
        Command::Combined => combined(parameters),
        Command::Gradient => gradient(parameters)
    }

}

pub fn construct(args: Parameters) {

    let arc_params = Arc::new(args);

    // println!("Argumnets parsed: {:?}", arc_params);

    println!("Reading data");

    let counts = arc_params.counts.as_ref().unwrap();

    let report_address = &arc_params.report_address;

    println!("##############################################################################################################");
    println!("##############################################################################################################");
    println!("##############################################################################################################");


    let mut rnd_forest = random_forest::Forest::initialize(counts, arc_params.clone(), report_address);

    rnd_forest.generate(arc_params.clone(),false);

}

pub fn predict(args: Parameters) {

    let arc_params = Arc::new(args.clone());
    let tree_backups: TreeBackups;

    if args.backup_vec.as_ref().unwrap_or(&vec![]).len() > 0 {
        tree_backups = TreeBackups::Vector(args.backup_vec.unwrap());
    }
    else {
        tree_backups = TreeBackups::File(args.backups.expect("Backup trees not provided"));
    }

    let counts = args.counts.as_ref().expect("Problem opening the matrix file (eg counts)");


    let dimensions = (counts.get(0).unwrap_or(&vec![]).len(),counts.len());

    let features: Vec<String>;
    let feature_map: HashMap<String,usize>;

    features = args.feature_names.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());
    feature_map = features.iter().cloned().enumerate().map(|x| (x.1,x.0)).collect();

    let forest = Forest::compact_reconstitute(tree_backups, Some(features), None ,None, "./").expect("Forest reconstitution failed");

    let predictions = forest.compact_predict(&counts,&feature_map,arc_params, &args.report_address);


}

pub fn combined(mut args:Parameters) {

    let arc_params = Arc::new(args);

    // println!("Argumnets parsed: {:?}", arc_params);

    let counts = arc_params.counts.as_ref().unwrap();

    let report_address = &arc_params.report_address;

    println!("##############################################################################################################");
    println!("##############################################################################################################");
    println!("##############################################################################################################");


    let mut rnd_forest = random_forest::Forest::initialize(counts, arc_params.clone(), &report_address);

    rnd_forest.generate(arc_params.clone(),true);

    let predictions = rnd_forest.compact_predict(&counts, &rnd_forest.feature_map().unwrap(), arc_params.clone(), &report_address);

}


fn gradient(args: Parameters) {


    // println!("Read arguments: {:?}", args);

    let arc_params = Arc::new(args.clone());

    let counts = read_counts(&args.count_array_file);

    let report_address = args.report_address;

    let boost_mode = args.boost_mode.unwrap_or(BoostMode::Subsampling);

    match boost_mode {
        BoostMode::Additive => {
            let mut forest = AdditiveBooster::initialize(&counts, arc_params.clone() , &report_address);

            forest.additive_growth(arc_params.clone());
            forest.compact_predict(&counts, &forest.feature_map(), arc_params , &report_address);
        },
        BoostMode::Subsampling => {
            let mut forest = BoostedForest::initialize(&counts, arc_params.clone() , &report_address);
            forest.grow_forest(arc_params.clone());
            forest.compact_predict(&counts, &forest.feature_map(), arc_params, &report_address);
        }
    }

}

fn read_counts(location:&str) -> Vec<Vec<f64>> {

    let count_array_file = File::open(location).expect("Count file error!");
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

                    if gene != "nan" && gene != "NAN" {
                        println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        println!("Cell content: {:?}", gene);
                    }
                    gene_vector.push(f64::NAN);
                }
            }

        }

        count_array.push(gene_vector);

        if i % 100 == 0 {
            println!("{}", i);
        }


    };

    println!("===========");
    println!("{},{}", count_array.len(),count_array.get(0).unwrap_or(&vec![]).len());

    matrix_flip(&count_array)

}


#[derive(Debug,Clone)]
pub struct Parameters {
    auto: bool,
    command: Command,
    count_array_file: String,
    counts: Option<Vec<Vec<f64>>>,
    feature_header_file: Option<String>,
    feature_names: Option<Vec<String>>,
    sample_header_file: Option<String>,
    sample_names: Option<Vec<String>>,
    report_address: String,

    processor_limit: Option<usize>,
    tree_limit: Option<usize>,
    leaf_size_cutoff: Option<usize>,
    dropout: Option<DropMode>,

    feature_subsample: Option<usize>,
    sample_subsample: Option<usize>,
    input_features: Option<usize>,
    output_features: Option<usize>,

    prediction_mode: Option<PredictionMode>,
    averaging_mode: Option<AveragingMode>,
    norm_mode: Option<NormMode>,
    weighing_mode: Option<WeighingMode>,
    split_mode: Option<SplitMode>,

    backups: Option<String>,
    backup_vec: Option<Vec<String>>,

    epochs: Option<usize>,
    epoch_duration: Option<usize>,
    boost_mode: Option<BoostMode>,

}

impl Parameters {

    pub fn empty() -> Parameters {
        let arg_struct = Parameters {
            auto: false,
            command: Command::Gradient,
            count_array_file: "".to_string(),
            counts: None,
            feature_header_file: None,
            feature_names: None,
            sample_header_file: None,
            sample_names: None,
            report_address: "./".to_string(),

            processor_limit: None,
            tree_limit: None,
            leaf_size_cutoff: None,
            dropout: None,

            feature_subsample: None,
            sample_subsample: None,
            input_features: None,
            output_features: None,

            prediction_mode: None,
            averaging_mode: None,
            norm_mode: None,
            weighing_mode: None,
            split_mode: None,


            backups: None,
            backup_vec: None,

            epochs: None,
            epoch_duration: None,
            boost_mode: None,
        };
        arg_struct
    }

    pub fn read<T: Iterator<Item = String>>(args: &mut T) -> Parameters {

        let mut arg_struct = Parameters::empty();

        let mut supress_warnings = false;
        let mut continuation_flag = false;
        let mut continuation_argument: String = "".to_string();

        while let Some((i,arg)) = args.enumerate().next() {
            if arg.clone().chars().next().unwrap_or('_') == '-' {
                continuation_flag = false;

            }
            match &arg[..] {
                "-sw" | "-suppress_warnings" => {
                    if i!=1 {
                        println!("If the supress warnings flag is not given first it may not function correctly.");
                    }
                    supress_warnings = true;
                },
                "-auto" | "-a"=> {
                    arg_struct.auto = true;
                    arg_struct.auto()
                },
                "-m" | "-mode" | "-pm" | "-prediction_mode" | "-prediction" => {
                    arg_struct.prediction_mode = Some(PredictionMode::read(&args.next().expect("Error reading prediction mode")));
                },
                "-d" | "-drop" | "-dropout_mode" => {
                    arg_struct.dropout = Some(DropMode::read(&args.next().expect("Error reading dropout mode")));
                },
                "-backups" | "-bk" => {
                    arg_struct.backups = Some(args.next().expect("Error parsing tree locations"));
                },
                "-bm" | "-boost_mode" => {
                    arg_struct.boost_mode = Some(BoostMode::read(&args.next().expect("Failed to read boost mode")));
                },
                "-wm" | "-w" | "-weighing_mode" => {
                    arg_struct.weighing_mode = Some(WeighingMode::read(&args.next().expect("Failed to read weighing mode!")));
                },
                "-sm" | "-split_mode" => {
                    arg_struct.split_mode = Some(SplitMode::read(&args.next().expect("Failed to read split mode")));
                },
                "-n" | "-norm" | "-norm_mode" => {
                    arg_struct.norm_mode = Some(NormMode::read(&args.next().expect("Failed to read norm mode")));
                },
                "-t" | "-trees" => {
                    arg_struct.tree_limit = Some(args.next().expect("Error processing tree count").parse::<usize>().expect("Error parsing tree count"));
                },
                "-tg" | "-tree_glob" => {
                    continuation_flag = true;
                    continuation_argument = arg.clone();
                },
                "-c" | "-counts" => {
                    arg_struct.count_array_file = args.next().expect("Error parsing count location!");
                    arg_struct.counts = Some(read_counts(&arg_struct.count_array_file))
                },
                "-p" | "-processors" | "-threads" => {
                    arg_struct.processor_limit = Some(args.next().expect("Error processing processor limit").parse::<usize>().expect("Error parsing processor limit"));
                },
                "-o" | "-output" => {
                    arg_struct.report_address = args.next().expect("Error processing output destination")
                },
                "-f" | "-h" | "-features" | "-header" => {
                    arg_struct.feature_header_file = Some(args.next().expect("Error processing feature file"));
                    arg_struct.feature_names = Some(read_header(arg_struct.feature_header_file.as_ref().unwrap()));
                },
                "-s" | "-samples" => {
                    arg_struct.sample_header_file = Some(args.next().expect("Error processing feature file"));
                    arg_struct.sample_names = Some(read_sample_names(arg_struct.sample_header_file.as_ref().unwrap()));
                }
                "-l" | "-leaves" => {
                    arg_struct.leaf_size_cutoff = Some(args.next().expect("Error processing leaf limit").parse::<usize>().expect("Error parsing leaf limit"));
                },
                "-if" | "-in_features" => {
                    arg_struct.input_features = Some(args.next().expect("Error processing in feature arg").parse::<usize>().expect("Error in feature  arg"));
                },
                "-of" | "-out_features" => {
                    arg_struct.output_features = Some(args.next().expect("Error processing out feature arg").parse::<usize>().expect("Error out feature arg"));
                },
                "-fs" | "-feature_sub" => {
                    arg_struct.feature_subsample = Some(args.next().expect("Error processing feature subsample arg").parse::<usize>().expect("Error feature subsample arg"));
                },
                "-ss" | "-sample_sub" => {
                    arg_struct.sample_subsample = Some(args.next().expect("Error processing sample subsample arg").parse::<usize>().expect("Error sample subsample arg"));
                },
                "-e" | "-epochs" => {
                    arg_struct.epochs = Some(args.next().expect("Error reading number of epochs").parse::<usize>().expect("-e not a number"));
                },
                "-es" | "-ed" | "-epoch_duration" => {
                    arg_struct.epoch_duration = Some(args.next().expect("Error reading epoch duration").parse::<usize>().expect("-ed not a number"));
                },

                &_ => {
                    if continuation_flag {
                        match &continuation_argument[..] {
                            "-tg" | "-tree_glob" => {
                                arg_struct.backup_vec.get_or_insert(vec![]).push(arg);
                            }
                            &_ => {
                                panic!("Continuation flag set but invalid continuation argument, debug prediction arg parse!");
                            }
                        }
                    }
                    else if !supress_warnings {
                        eprintln!("Warning, detected unexpected argument:{}. Ignoring, press enter to continue, or CTRL-C to stop. Were you trying to input multiple arguments? Only some options take multiple arguments. Watch out for globs(*, also known as wild cards), these count as multiple arguments!",arg);
                        stdin().read_line(&mut String::new());
                    }
                }

            }
        }

        arg_struct

    }


    fn auto(&mut self) {

        let counts = self.counts.as_ref().expect("Please specify counts file before the \"-auto\" argument.");

        let features = counts.len();
        let samples = counts.get(0).unwrap_or(&vec![]).len();

        let mut output_features = ((features as f64 / (features as f64).log10()) as usize).min(features);

        let mut input_features: usize;

        if features < 3 {
            input_features = features;
            output_features = features;
        }
        else if features < 100 {
            input_features = ((features as f64 * ((125 - features) as f64) / 125.) as usize).max(1);
        }

        else {
            input_features = ((features as f64 * (((1500 - features as i32) as f64) / 7000.).max(0.1)) as usize).max(1);
        }

        let feature_subsample = output_features;

        let sample_subsample: usize;

        if samples < 10 {
            eprintln!("Warning, you seem to be using suspiciously few samples, are you sure you specified the right file? If so, trees may not be the right solution to your problem.");
            sample_subsample = samples;
        }
        else if samples < 1000 {
            sample_subsample = (samples/3)*2;
        }
        else if samples < 5000 {
            sample_subsample = samples/2;
        }
        else {
            sample_subsample = samples/4;
        }

        let leaf_size_cutoff = ((sample_subsample as f64).sqrt() as usize);

        let trees = 100;

        let processors = num_cpus::get();

        let dropout: DropMode;

        if counts.iter().flat_map(|x| x).any(|x| x.is_nan()) {
            dropout = DropMode::NaNs;
        }
        else if counts.iter().flat_map(|x| x.iter().map(|y| if *y == 0. {1.} else {0.})).sum::<f64>() > ((samples * features) as f64 / 4.) {
            dropout = DropMode::Zeros;
        }
        else {
            dropout = DropMode::No;
        }

        let prediction_mode: PredictionMode;

        if counts.iter().flat_map(|x| x.iter().map(|y| if *y != 0. {1.} else {0.})).sum::<f64>() < ((samples * features) as f64 / 4.) {
            prediction_mode = PredictionMode::Abort;
        }
        else if counts.iter().flat_map(|x| x.iter().map(|y| if *y != 0. {1.} else {0.})).sum::<f64>() < ((samples * features) as f64 / 2.) {
            prediction_mode = PredictionMode::Truncate;
        }
        else {
            prediction_mode = PredictionMode::Branch;
        }

        println!("Automatic parameters:");
        println!("{:?}",feature_subsample);
        println!("{:?}",sample_subsample);
        println!("{:?}",input_features);
        println!("{:?}",output_features);
        println!("{:?}",processors);
        println!("{:?}",trees,);
        println!("{:?}",leaf_size_cutoff);
        println!("{:?}",dropout);
        println!("{:?}",prediction_mode);

        self.auto = true;

        self.feature_subsample.get_or_insert( feature_subsample );
        self.sample_subsample.get_or_insert( sample_subsample );
        self.input_features.get_or_insert( input_features );
        self.output_features.get_or_insert( output_features );


        self.processor_limit.get_or_insert( processors );
        self.tree_limit.get_or_insert( trees );
        self.leaf_size_cutoff.get_or_insert( leaf_size_cutoff );
        self.dropout.get_or_insert( dropout );

        self.prediction_mode.get_or_insert( prediction_mode );
        self.averaging_mode.get_or_insert( AveragingMode::Arithmetic );

    }


}

// Various modes that are included in Parameters, serving as control elements for program internals. Each mode can parse strings that represent alternative options for that mode. Enums were chosen because they compile down to extremely small memory footprint.


#[derive(Clone,Debug)]
pub enum BoostMode {
    Additive,
    Subsampling,
}

impl BoostMode {
    pub fn read(input: &str) -> BoostMode {
        match input {
            "additive" | "a" | "add" => BoostMode::Additive,
            "s" | "subsampling" | "subsample" => BoostMode::Subsampling,
            _ => {
                eprintln!("Not a valid boost mode, choose sub or add (defaulting to add)");
                BoostMode::Additive
            }
        }
    }

}


#[derive(Debug,Clone)]
pub enum Command {
    Combined,
    Construct,
    Predict,
    Gradient,
}

impl Command {

    pub fn parse(command: &str) -> Command {

        match &command[..] {
            "construct" => {
                Command::Construct
            },
            "predict" => {
                Command::Predict
            },
            "construct_predict" | "conpred" | "combined" => {
                Command::Combined
            }
            "gradient" => {
                Command::Gradient
            }
            _ =>{
                println!("Not a valid top-level command, please choose from \"construct\",\"predict\", or \"construct_predict\". Exiting");
                panic!()
            }
        }
    }
}

impl PredictionMode {
    pub fn read(input:&str) -> PredictionMode {
        match input {
            "branch" | "branching" | "b" => PredictionMode::Branch,
            "truncate" | "truncating" | "t" => PredictionMode::Truncate,
            "abort" | "ab" => PredictionMode::Abort,
            "auto" | "a" => PredictionMode::Auto,
            _ => panic!("Not a valid prediction mode, choose branch, truncate, or abort.")
        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum PredictionMode {
    Branch,
    Truncate,
    Abort,
    Auto
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum AveragingMode {
    Arithmetic,
    Stacking
}

impl AveragingMode {
    pub fn read(input:&str) -> AveragingMode {
        match input {
            "a" | "arithmetic" | "average" => AveragingMode::Arithmetic,
            "s" | "stacking" => AveragingMode::Stacking,
            _ => panic!("Not a valid averaging mode, choose arithmetic or stacking.")
        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum WeighingMode {
    AbsoluteGain,
    AbsGainSquared,
    AbsoluteDispersion,
    AbsDispSquared,
}

impl WeighingMode {
    pub fn read(input:&str) -> WeighingMode {
        match input {
            "gain" | "absolute_gain" | "g" => WeighingMode::AbsoluteGain,
            "gain_squared" | "gs" => WeighingMode::AbsGainSquared,
            "dispersion" | "d" => WeighingMode::AbsoluteDispersion,
            "dispersion_squared" | "ds" => WeighingMode::AbsDispSquared,
            _ => panic!("Not a valid weighing mode, please pick from gain, gain_squared, dispersion, dispersion_squared")
        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum SplitMode {
    Cov,
    CovSquared,
    MAD,
    MADSquared,
}

impl SplitMode {
    pub fn read(input: &str) -> SplitMode {
        match input {
            "c" | "cov" => SplitMode::Cov,
            "m" | "mad"  => SplitMode::MAD,
            _ => panic!("Not a valid split mode, choose cov or mad")

        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum NormMode {
    L1,
    L2,
}

impl NormMode {
    pub fn read(input: &str) -> NormMode {
        match input {
            "1" | "L1" | "l1" => NormMode::L1,
            "2" | "L2" | "l2" => NormMode::L2,
            _ => panic!("Not a valid norm, choose l1 or l2")
        }
    }
}

impl DropMode {
    pub fn read(input: &str) -> DropMode {
        match input {
            "zeros" | "zero" | "z" => DropMode::Zeros,
            "nans" | "nan" | "NaN" => DropMode::NaNs,
            "none" | "no" => DropMode::No,
            _ => panic!("Not a valid drop mode, choose zero, nan, or none")
        }
    }

    pub fn cmp(&self) -> f64 {
        match self {
            &DropMode::Zeros => 0.,
            &DropMode::NaNs => f64::NAN,
            &DropMode::No => f64::INFINITY,
        }
    }

    pub fn bool(&self) -> bool {
        match self {
            &DropMode::Zeros => true,
            &DropMode::NaNs => true,
            &DropMode::No => false,
        }
    }
}

#[derive(Debug,Clone,Copy,Serialize,Deserialize)]
pub enum DropMode {
    Zeros,
    NaNs,
    No,
}

#[derive(Clone,Debug)]
pub enum TreeBackups {
    File(String),
    Vector(Vec<String>),
    Trees(Vec<PredictiveTree>)
}


fn read_header(location: &str) -> Vec<String> {

    println!("Reading header: {}", location);

    let mut header_map = HashMap::new();

    let header_file = File::open(location).expect("Header file error!");
    let mut header_file_iterator = io::BufReader::new(&header_file).lines();

    for (i,line) in header_file_iterator.by_ref().enumerate() {
        let feature = line.unwrap_or("error".to_string());
        let mut renamed = feature.clone();
        let mut j = 1;
        while header_map.contains_key(&renamed) {
            renamed = [feature.clone(),j.to_string()].join("");
            eprintln!("WARNING: Two individual features were named the same thing: {}",feature);
            j += 1;
        }
        header_map.insert(renamed,i);
    };

    let mut header_inter: Vec<(String,usize)> = header_map.iter().map(|x| (x.0.clone().clone(),x.1.clone())).collect();
    header_inter.sort_unstable_by_key(|x| x.1);
    let header_vector: Vec<String> = header_inter.into_iter().map(|x| x.0).collect();

    println!("Read {} lines", header_vector.len());

    header_vector
}

fn read_sample_names(location: &str) -> Vec<String> {

    let mut header_vector = Vec::new();

    let sample_name_file = File::open(location).expect("Sample name file error!");
    let mut sample_name_lines = io::BufReader::new(&sample_name_file).lines();

    for line in sample_name_lines.by_ref() {
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



fn matrix_flip(in_mat: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    let mut out = Vec::new();

    for _ in in_mat.get(0).unwrap_or(&vec![]).iter() {
        out.push(vec![in_mat[0][0];in_mat.len()]);
    }

    for (i,iv) in in_mat.iter().enumerate() {
        for (j,jv) in iv.iter().enumerate() {
            out[j][i] = *jv;
        }
    }

    out
}

fn mtx_dim(in_mat: &Vec<Vec<f64>>) -> (usize,usize) {
    (in_mat.len(),in_mat.get(0).unwrap_or(&vec![]).len())
}

fn add_matrix(mat1: &Vec<Vec<f64>>,mat2:&Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(mat1) != mtx_dim(mat2) {
        panic!("Attempted to add matrices of unequal dimensions: {:?},{:?}", mtx_dim(mat1),mtx_dim(mat2));
    }

    let dim = mtx_dim(mat1);

    let mut output = vec![vec![0.;dim.1];dim.0];

    for i in 0..dim.0 {
        for j in 0..dim.1{
            output[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    output

}

fn sub_matrix(mat1: &Vec<Vec<f64>>,mat2:&Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(mat1) != mtx_dim(mat2) {
        panic!("Attempted to subtract matrices of unequal dimensions: {:?},{:?}", mtx_dim(mat1),mtx_dim(mat2));
    }

    let dim = mtx_dim(mat1);

    let mut output = vec![vec![0.;dim.1];dim.0];

    for i in 0..dim.0 {
        for j in 0..dim.1{
            output[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    output
}

fn add_mtx_ip(mut mtx1: Vec<Vec<f64>>, mtx2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(&mtx1) != mtx_dim(&mtx2) {
        panic!("Attempted to add matrices of unequal dimensions: {:?},{:?}", mtx_dim(&mtx1),mtx_dim(&mtx2));
    }

    let dim = mtx_dim(&mtx1);

    for i in 0..dim.0 {
        for j in 0..dim.1{
            mtx1[i][j] += mtx2[i][j];
        }
    }

    mtx1
}

fn sub_mtx_ip(mut mtx1: Vec<Vec<f64>>, mtx2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(&mtx1) != mtx_dim(&mtx2) {
        panic!("Attempted to add matrices of unequal dimensions: {:?},{:?}", mtx_dim(&mtx1),mtx_dim(&mtx2));
    }

    let dim = mtx_dim(&mtx1);

    for i in 0..dim.0 {
        for j in 0..dim.1{
            mtx1[i][j] += mtx2[i][j];
        }
    }

    mtx1
}

fn abs_mtx_ip(mtx: &mut Vec<Vec<f64>>) {
    let dim = mtx_dim(&mtx);
    for i in 0..dim.0 {
        for j in 0..dim.1{
            mtx[i][j] = mtx[i][j].abs();
        }
    }
}

fn square_mtx_ip(mut mtx: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let dim = mtx_dim(&mtx);
    for i in 0..dim.0 {
        for j in 0..dim.1{
            mtx[i][j] = mtx[i][j].powi(2);
        }
    }
    mtx
}


fn multiply_matrix(mat1: &Vec<Vec<f64>>,mat2:&Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(mat1) != mtx_dim(mat2) {
        panic!("Attempted to multiply matrices of unequal dimensions: {:?},{:?}", mtx_dim(mat1),mtx_dim(mat2));
    }

    let dim = mtx_dim(mat1);

    let mut output = vec![vec![0.;dim.1];dim.0];

    for i in 0..dim.0 {
        for j in 0..dim.1{
            output[i][j] = mat1[i][j] * mat2[i][j];
        }
    }

    output
}

fn zero_matrix(x:usize,y:usize) -> Vec<Vec<f64>> {
    vec![vec![0.;y];x]
}

fn float_matrix(x:usize,y:usize,float:f64) -> Vec<Vec<f64>> {
    vec![vec![float;y];x]
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

fn tsv_format<T:Debug>(input:&Vec<Vec<T>>) -> String {

    input.iter().map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")

}

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

fn covariance(vec1:&Vec<f64>,vec2:&Vec<f64>) -> f64 {

    if vec1.len() != vec2.len() {
        panic!("Tried to compute covariance for unequal length vectors: {}, {}",vec1.len(),vec2.len());
    }

    let mean1: f64 = vec1.iter().sum::<f64>() / (vec1.len() as f64);
    let mean2: f64 = vec2.iter().sum::<f64>() / (vec2.len() as f64);

    let covariance = vec1.iter().zip(vec2.iter()).map(|(x,y)| (x - mean1) * (y - mean2)).sum::<f64>() / (vec1.len() as f64 - 1.);

    covariance

}

fn pearsonr(vec1:&Vec<f64>,vec2:&Vec<f64>) -> f64 {

    if vec1.len() != vec2.len() {
        panic!("Tried to compute covariance for unequal length vectors: {}, {}",vec1.len(),vec2.len());
    }

    0.
}

mod manual_testing {

    use super::*;

    pub fn test_command_predict_full() {
        let mut args = vec!["predict", "-m","branching","-b","tree.txt","-tg","tree.0","tree.1","tree.2","-c","counts.txt","-p","3","-o","./elsewhere/","-f","header_backup.txt"].into_iter().map(|x| x.to_string());

        let command = Command::parse(&args.next().unwrap());

        println!("{:?}",command);

        // panic!();

    }

}

#[cfg(test)]
pub mod primary_testing {

    use super::*;

    #[test]
    fn test_command_trivial() {

        match Command::parse("construct") {
            Command::Construct => {},
            _ => panic!("Failed prediction parse")
        };

        match Command::parse("predict") {
            Command::Predict => {},
            _ => panic!("Failed prediction parse")
        };

        match Command::parse("combined") {
            Command::Combined => {},
            _ => panic!("Failed prediction parse")
        };

    }

    #[test]
    #[should_panic]
    fn test_command_wrong() {
        Command::parse("abc");
    }

    // #[test]
    // fn test_parameters_args() {
    //     let mut args_iter = vec!["predict", "-m","branching","-b","tree.txt","-tg","tree.0","tree.1","tree.2","-c","testing/iris.drop","-p","3","-o","./elsewhere/","-f","header_backup.txt"].into_iter().map(|x| x.to_string());
    //
    //     let args = Parameters::read(&mut args_iter);
    //
    //     match args.prediction_mode.unwrap() {
    //         PredictionMode::Branch => {},
    //         _ => panic!("Branch mode not read correctly")
    //     }
    //
    //     assert_eq!(args.backup_vec.unwrap(), vec!["tree.0".to_string(),"tree.1".to_string(),"tree.2".to_string()]);
    //     assert_eq!(args.backups.unwrap(), "tree.txt".to_string());
    //
    //     assert_eq!(args.count_array_file, "counts.txt".to_string());
    //     assert_eq!(args.feature_header_file.unwrap(), "header_backup.txt".to_string());
    //     assert_eq!(args.sample_header_file, None);
    //     assert_eq!(args.report_address, "./elsewhere/".to_string());
    //
    //     assert_eq!(args.processor_limit.unwrap(), 3);
    //
    // }


    #[test]
    fn test_read_counts_trivial() {
        assert_eq!(read_counts("./testing/trivial.txt"),Vec::<Vec<f64>>::with_capacity(0))
    }

    #[test]
    fn test_read_counts_simple() {
        assert_eq!(read_counts("./testing/simple.txt"), vec![vec![10.,5.,-1.,0.,-2.,10.,-3.,20.]])
    }

    #[test]
    fn test_read_header_trivial() {
        assert_eq!(read_header("./testing/trivial.txt"),Vec::<String>::with_capacity(0))
    }

    #[test]
    fn test_read_header_simple() {
        assert_eq!(read_header("./testing/iris.features"),vec!["petal_length","petal_width","sepal_length","sepal_width"])
    }




}
/////TESTING CODE///////

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
// fn inner_axis_sum(in_mat: &Vec<Vec<f64>>) -> Vec<f64> {
//
//     let mut s = vec![0f64;in_mat[0].len()];
//
//     for i in in_mat {
//         for j in i.iter().enumerate() {
//             s[j.0] += *j.1;
//         }
//     }
//     // println!("Inner axis being computed: {:?}", in_mat.iter().map(|x| x[0]).take(10).collect::<Vec<_>>());
//     // println!("{}", in_mat[0].len());
//     // println!("Inner axis sum: {}", s[0]);
//     s
// }
//
// fn inner_axis_mean(in_mat: &Vec<Vec<f64>>) -> Vec<f64> {
//
//     let mut s = vec![0f64;in_mat[0].len()];
//
//     for i in in_mat {
//         for j in i.iter().enumerate() {
//             s[j.0] += *j.1/(in_mat.len() as f64);
//         }
//     }
//
//     s
// }
//
// fn inner_axis_variance_sum(in_mat: &Vec<Vec<f64>>, in_means: Option<Vec<f64>>) -> Vec<f64> {
//
//     let m: Vec<f64>;
//
//     match in_means {
//         Option::Some(input) => m = input,
//         Option::None => m = inner_axis_mean(in_mat)
//     }
//
//     println!("Inner axis mean: {:?}", m);
//
//     let mut vs = vec![0f64;in_mat[0].len()];
//
//     for i in in_mat {
//         for j in i.iter().enumerate() {
//             // println!("Variance sum compute");
//             // println!("{}",*j.1);
//             // println!("{}", m[j.0]);
//             vs[j.0] += (*j.1 - m[j.0]).powi(2);
//             // println!("{}", vs[j.0]);
//         }
//     }
//     // println!("Inner_axis being computed: {:?}", in_mat.iter().map(|x| x[0]).take(10).collect::<Vec<_>>());
//     // println!("{}", in_mat.len());
//     // println!("Inner axis variance sum: {}", vs[0]);
//     vs
// }
//
// fn inner_axis_stats(in_mat: &Vec<Vec<f64>>) -> (Vec<f64>,Vec<f64>) {
//
//     let m = inner_axis_mean(in_mat);
//
//     let mut v = vec![0f64;in_mat[0].len()];
//
//     for i in in_mat {
//         for j in i.iter().enumerate() {
//             v[j.0] += (*j.1 - m[j.0]).powi(2)/(v.len() as f64);
//         }
//     }
//
//     (m,v)
// }
