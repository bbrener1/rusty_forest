use rand::Rng;
use std::iter::repeat;
use std::collections::HashSet;
use std::collections::HashMap;
use std::io::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::io::Read;
use compact_predictor::compact_predict;
use rand::thread_rng;
use tree::Tree;
use rank_table::RankTable;
use tree::PredictiveTree;
use DropMode;
use PredictionMode;
use matrix_flip;
use sub_matrix;
use tsv_format;


pub struct BoostedForest {
    leaf_size: usize,
    processor_limit: usize,
    report_string: String,

    feature_names: Vec<String>,
    sample_names: Vec<String>,

    feature_similarity_matrix: Vec<Vec<f64>>,
    cell_coocurrence_matrix: Vec<Vec<f64>>,

    error_matrix: Vec<Vec<f64>>,

    counts: Vec<Vec<f64>>,
    trees: Vec<Tree>,
    predictive_trees: Vec<PredictiveTree>,

    dimensions: (usize,usize),
    epoch_duration: usize,
    epochs: usize,

    prototype_rank_table: Option<RankTable>,
    prototype_tree: Option<Tree>,

    dropout: DropMode,
    prediction_mode: PredictionMode,
}

impl BoostedForest {

    pub fn initialize(counts:&Vec<Vec<f64>>,epoch_duration:usize,leaf_size:usize,epochs:usize,processor_limit:usize, feature_option: Option<Vec<String>>, sample_option: Option<Vec<String>>, dropout: DropMode, prediction_mode: PredictionMode, report_address:&str) -> BoostedForest {

        let dimensions: (usize,usize) = (counts.len(),counts.first().unwrap_or(&vec![]).len());

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let feature_similarity_matrix = vec![vec![0.;dimensions.1];dimensions.0];
        let cell_coocurrence_matrix = vec![vec![0.;dimensions.1];dimensions.0];

        let report_string = format!("{}.0",report_address).to_string();

        let prototype_table = RankTable::new(&counts,&feature_names,&sample_names,dropout);

        BoostedForest {
            trees: Vec::new(),
            leaf_size: leaf_size,
            processor_limit: processor_limit,
            report_string: report_string,
            feature_names: feature_names,
            sample_names: sample_names,

            feature_similarity_matrix: feature_similarity_matrix,
            cell_coocurrence_matrix: cell_coocurrence_matrix,
            error_matrix: counts.clone(),

            predictive_trees: Vec::new(),
            dimensions: dimensions,
            epoch_duration: epoch_duration,
            epochs: epochs,
            counts: counts.clone(),
            prototype_rank_table: Some(prototype_table),
            prototype_tree: None,
            dropout:dropout,
            prediction_mode: prediction_mode
        }
    }

    pub fn grow_forest(&mut self) -> Result<(),Error>{

        for i in 0..self.epochs {

            self.grow_epoch(800, 400, 1000, i);

            let epoch_predictions = self.compact_predict(&self.counts, &self.feature_map(), &self.prediction_mode, &self.dropout, &[self.report_string.clone(),format!(".{}.prediction",i)].join(""))?;

            self.error_matrix = sub_matrix(&self.counts, &epoch_predictions)

        }

        Ok(())
    }



    pub fn grow_epoch (&mut self,samples_per_tree:usize,input_features_per_tree:usize,output_features_per_tree:usize,epoch:usize) {

        println!("Initializing an epoch");

        let prototype_tree =  Tree::prototype_tree(&self.counts,&self.counts,&self.sample_names,&self.feature_names,&self.feature_names,self.leaf_size, self.dropout, self.processor_limit, [self.report_string.clone(),format!(".{}.0",epoch)].join(""));

        println!("Epoch prototype done, drawing weights");

        let (mut input_feature_weights, mut output_feature_weights, mut sample_weights) = self.draw_weights(output_features_per_tree * samples_per_tree);

        println!("Weights drawn");

        for i in 1..self.epoch_duration {

            println!("Tree {}", i);

            let samples = weighted_sampling(samples_per_tree, &self.sample_names, &sample_weights, false).0;

            let (input_features,output_features) = self.inputs_and_outputs(input_features_per_tree, output_features_per_tree, input_feature_weights.clone(), output_feature_weights.clone());

            let mut new_tree = prototype_tree.derive_specified(&samples.iter().collect(),&input_features.iter().collect(),&output_features.iter().collect(),i);

            println!("{:?}", new_tree.report_address);
            new_tree.grow_branches();
            new_tree.serialize_compact();
            self.predictive_trees.push(new_tree.strip_consume());

        }


    }


    pub fn compact_predict(&self,counts:&Vec<Vec<f64>>,feature_map: &HashMap<String,usize>,prediction_mode:&PredictionMode, drop_mode: &DropMode,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {

        println!("Predicting:");

        let predictions = compact_predict(&self.predictive_trees,&matrix_flip(counts),feature_map,prediction_mode,drop_mode, self.processor_limit);

        let mut prediction_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction"].join("")).unwrap();
        prediction_dump.write(&tsv_format(&predictions).as_bytes())?;
        prediction_dump.write(b"\n")?;

        let mut truth_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction_truth"].join("")).unwrap();
        truth_dump.write(&format!("{:?}",&matrix_flip(&self.counts)).as_bytes())?;
        truth_dump.write(b"\n")?;

        let mut prediction_header = OpenOptions::new().create(true).append(true).open([report_address,".prediction_header"].join("")).unwrap();
        prediction_header.write(&format!("{:?}",feature_map).as_bytes())?;
        prediction_header.write(b"\n")?;

        Ok(predictions)
    }

    pub fn feature_map(&self) -> HashMap<String,usize> {
        self.feature_names.clone().into_iter().enumerate().map(|x| (x.1,x.0)).collect()
    }

    pub fn inputs_and_outputs(&self, input_draws: usize, output_draws: usize, mut input_feature_weights: Vec<f64>, mut output_feature_weights: Vec<f64>) -> (Vec<String>,Vec<String>) {

        /// Input features and output features are exclusive of each other, so you have to pick input features first.
        /// Otherwise output features that are poorly predicted will frequently get picked, and will never serve as input features
        ///

        let mut input_features = Vec::with_capacity(input_draws);

        for i in 0..input_draws {
            let feature_index = weighted_sampling(1, &self.features(), &input_feature_weights, false).1[0];
            input_feature_weights =
                input_feature_weights
                .iter()
                .zip(
                    self.feature_similarity_matrix[feature_index]
                    .iter()
                )
                .map(|(x,y)| x * (1. - (y/2.)))
                .collect();
            input_feature_weights[feature_index] = 0.;
            output_feature_weights[feature_index] = 0.;

            input_features.push(self.features()[feature_index].clone())
        }

        let output_features = weighted_sampling(output_draws, &self.features(), &output_feature_weights, false).0.iter().cloned().collect();

        (input_features,output_features)
    }

    pub fn sample_cells(&self,draws:usize) -> Vec<(usize,usize,f64)> {

        let flattened_cells: Vec<(usize,usize,f64)> = self.error_matrix.iter().enumerate().flat_map(|y| repeat(y.0).zip(y.1.iter().enumerate())).map(|(a,(b,c))| (a,b,*c)).collect();

        let picks = weighted_sampling(draws, &flattened_cells, &flattened_cells.iter().map(|x| x.2).collect(), false).0;

        picks
    }


    pub fn draw_weights(&self,cells:usize) -> (Vec<f64>,Vec<f64>,Vec<f64>) {

        println!("Drawing weights");

        let error_cells = self.sample_cells(cells);

        println!("Error cells drawn");

        let mut output_feature_weights = vec![0.;self.features().len()];

        for (j,cell) in error_cells.iter().enumerate() {
            output_feature_weights[cell.0] += 1.;
        };

        println!("Output feature weights drawn");

        let input_feature_weights = {
            self.counts
                .iter()
                .map(|x| x
                    .iter()
                    .map(|y| {
                        match self.dropout {
                            DropMode::Zeros => if *y == 0. {0.} else {1.},
                            DropMode::NaNs => if y.is_nan() {0.} else {1.},
                            DropMode::No => 1.,
                        }

                    })
                    .sum()
                )
                .collect::<Vec<f64>>()
            };

        println!("Input feature weights drawn");

        let sample_weights = matrix_flip(&self.error_matrix).iter().map(|x| x.iter().sum()).collect();

        println!("Sample weights drawn");

        (input_feature_weights,output_feature_weights,sample_weights)

    }

    pub fn features(&self) -> &Vec<String> {
        &self.feature_names
    }

    pub fn samples(&self) -> &Vec<String> {
        &self.sample_names
    }

}



pub fn weighted_sampling<T: Clone>(draws: usize, samples: &Vec<T>, weights: &Vec<f64>,replacement:bool) -> (Vec<T>,Vec<usize>) {

    let mut rng = thread_rng();

    let mut exclusion_set: HashSet<usize> = HashSet::new();

    let mut drawn_samples: Vec<T> = Vec::with_capacity(draws);
    let mut drawn_indecies: Vec<usize> = Vec::with_capacity(draws);

    if replacement {

        for i in 0..draws {

            let weighted_choice = rng.gen_range::<f64>(0.,weights.iter().sum());

            let (index,sum) = weights.iter().enumerate().fold( (0,0.), |mut acc,x| {if acc.1 <= weighted_choice {acc.0 = x.0}; (acc.0,acc.1 + x.1) });

            drawn_samples.push(samples[index].clone());
            drawn_indecies.push(index);
        }

    }

    else {

        let mut local_samples: Vec<T> = samples.iter().cloned().collect();
        let mut local_weights: Vec<&f64> = weights.iter().collect();

        for i in 0..draws {

            let weighted_choice = rng.gen_range::<f64>(0.,local_weights.iter().cloned().sum());

            let (index,sum) = local_weights.iter().enumerate().fold( (0,0.), |mut acc,x| {if acc.1 <= weighted_choice {acc.0 = x.0}; (acc.0, acc.1 + *x.1) });

            drawn_samples.push(local_samples[index].clone());
            drawn_indecies.push(index);

            local_samples.remove(index);
            local_weights.remove(index);
        }

    }


    (drawn_samples,drawn_indecies)

}

// pub fn weighted_sampling<'a,T>(draws: usize, samples: &'a Vec<T>, weights: &Vec<f64>,replacement:bool) -> (Vec<&'a T>,Vec<usize>) {
//
//     let mut rng = thread_rng();
//
//     let mut exclusion_set: HashSet<usize> = HashSet::new();
//
//     let mut drawn_samples = Vec::with_capacity(draws);
//     let mut drawn_indecies = Vec::with_capacity(draws);
//
//     if replacement {
//
//         for i in 0..draws {
//
//             let weighted_choice = rng.gen_range::<f64>(0.,weights.iter().sum());
//
//             let (index,sum) = weights.iter().enumerate().fold( (0,0.), |mut acc,x| {if acc.1 <= weighted_choice {acc.0 = x.0}; (acc.0,acc.1 + x.1) });
//
//             drawn_samples.push(&samples[index]);
//             drawn_indecies.push(index);
//         }
//
//     }
//
//     else {
//
//         let mut local_samples: Vec<&T> = samples.iter().collect();
//         let mut local_weights: Vec<&f64> = weights.iter().collect();
//
//         for i in 0..draws {
//
//             let weighted_choice = rng.gen_range::<f64>(0.,local_weights.iter().cloned().sum());
//
//             let (index,sum) = local_weights.iter().enumerate().fold( (0,0.), |mut acc,x| {if acc.1 <= weighted_choice {acc.0 = x.0}; (acc.0, acc.1 + *x.1) });
//
//             drawn_samples.push(local_samples[index]);
//             drawn_indecies.push(index);
//
//             local_samples.remove(index);
//             local_weights.remove(index);
//         }
//
//     }
//
//
//     (drawn_samples,drawn_indecies)
//
// }
