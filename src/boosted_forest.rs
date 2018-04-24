use rand::Rng;
use std::iter::repeat;
use std::collections::HashSet;
use std::collections::HashMap;
use std::io::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::cmp::Ordering;
use std::sync::Arc;
use compact_predictor::compact_predict;
use rand::thread_rng;
use tree::Tree;
use rank_table::RankTable;
use tree::PredictiveTree;
use boosted_tree_thread_pool::BoostedTreeThreadPool;
use boosted_tree_thread_pool::BoostedMessage;
use std::sync::mpsc;
use DropMode;
use PredictionMode;
use matrix_flip;
use sub_matrix;
use mtx_dim;
use tsv_format;
use Parameters;

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

    pub fn initialize(counts:&Vec<Vec<f64>>, parameters: Arc<Parameters>, report_address:&str) -> BoostedForest {

        let dimensions: (usize,usize) = (counts.len(),counts.first().unwrap_or(&vec![]).len());

        let feature_names = parameters.feature_names.clone().unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = parameters.sample_names.clone().unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let feature_similarity_matrix = vec![vec![0.;dimensions.0];dimensions.0];
        let cell_coocurrence_matrix = vec![vec![0.;dimensions.1];dimensions.1];

        let report_string = format!("{}.0",report_address).to_string();

        let prototype_table = RankTable::new(&counts,&feature_names,&sample_names,parameters.clone());

        BoostedForest {
            trees: Vec::new(),
            leaf_size: parameters.leaf_size_cutoff.unwrap_or(1),
            processor_limit: parameters.processor_limit.unwrap_or(1),
            report_string: report_string,
            feature_names: feature_names,
            sample_names: sample_names,

            feature_similarity_matrix: feature_similarity_matrix,
            cell_coocurrence_matrix: cell_coocurrence_matrix,
            error_matrix: counts.clone(),

            predictive_trees: Vec::new(),
            dimensions: dimensions,
            epoch_duration: parameters.epoch_duration.unwrap_or(1),
            epochs: parameters.epochs.unwrap_or(1),
            counts: counts.clone(),
            prototype_rank_table: Some(prototype_table),
            prototype_tree: None,
            dropout:parameters.dropout.unwrap_or(DropMode::Zeros),
            prediction_mode: parameters.prediction_mode.unwrap_or(PredictionMode::Truncate)
        }
    }

    pub fn grow_forest(&mut self, parameters: Arc<Parameters>) -> Result<(),Error>{

        for i in 0..self.epochs {

            self.grow_epoch(parameters.clone(), i);

            let epoch_predictions = self.compact_predict(&self.counts, &self.feature_map(), parameters.clone(), &[self.report_string.clone(),format!(".{}",i)].join(""))?;

            self.error_matrix = sub_matrix(&self.counts, &matrix_flip(&epoch_predictions));

            println!("Error matrix dimensions:{:?}",mtx_dim(&self.error_matrix));

        }

        Ok(())
    }

    pub fn grow_epoch (&mut self, parameters: Arc<Parameters> ,epoch:usize) {

        println!("Initializing an epoch");

        println!("Drawing weights");

        let output_features_per_tree = parameters.output_features.unwrap_or(1);
        let input_features_per_tree = parameters.input_features.unwrap_or(1);
        let samples_per_tree = parameters.sample_subsample.unwrap_or(1);

        let (input_feature_weights, output_feature_weights, sample_weights) = self.draw_weights(output_features_per_tree * samples_per_tree);

        println!("Weights drawn");

        let mut tree_receivers = Vec::with_capacity(self.epoch_duration);

        let mut prototype_tree =  Tree::prototype_tree(&self.counts,&self.counts,&self.sample_names,&self.feature_names,&self.feature_names, Some(output_feature_weights.clone()), parameters, [self.report_string.clone(),format!(".{}.0",epoch)].join(""));

        let mut thread_pool = BoostedTreeThreadPool::new(&prototype_tree,self.processor_limit);

        for i in 1..self.epoch_duration {

            println!("Tree {}", i);

            let samples = weighted_sampling(samples_per_tree, &self.sample_names, &sample_weights, false).0;

            println!("Samples drawn");

            let (input_features,output_features) = self.inputs_and_outputs(input_features_per_tree, output_features_per_tree, input_feature_weights.clone(), output_feature_weights.clone());

            println!("Features drawn");

            let (tx,rx) = mpsc::channel();

            thread_pool.send(BoostedMessage::Selections(i,input_features,output_features,samples,tx));

            tree_receivers.push(rx);

        }

        for receiver in tree_receivers {
            let tree = receiver.recv().expect("Failed to unwrap boosted tree");
            tree.serialize_compact();
            self.predictive_trees.push(tree);
        }

        BoostedTreeThreadPool::terminate(&mut thread_pool);
        prototype_tree.terminate_pool();

    }


    pub fn compact_predict(&self,counts:&Vec<Vec<f64>>,feature_map: &HashMap<String,usize>,parameters: Arc<Parameters>,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {

        println!("Predicting:");

        let predictions = compact_predict(&self.predictive_trees,&matrix_flip(counts),feature_map,parameters);

        let mut prediction_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction"].join("")).unwrap();
        prediction_dump.write(&tsv_format(&predictions).as_bytes())?;
        prediction_dump.write(b"\n")?;

        let mut truth_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction_truth"].join("")).unwrap();
        truth_dump.write(&tsv_format(&matrix_flip(&self.counts)).as_bytes())?;
        truth_dump.write(b"\n")?;

        let mut header_vec = vec!["";feature_map.len()];
        for (f,i) in feature_map { header_vec[*i] = f; };
        let header = tsv_format(&vec![header_vec]);

        let mut prediction_header = OpenOptions::new().create(true).append(true).open([report_address,".prediction_header"].join("")).unwrap();
        prediction_header.write(&header.as_bytes())?;
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

        println!("Drawing features");
        println!("{},{}",input_draws,output_draws);

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

        println!("Inputs drawn");

        let output_features = weighted_sampling(output_draws, &self.features(), &output_feature_weights, false).0.iter().cloned().collect();

        println!("Outputs drawn");

        (input_features,output_features)
    }

    pub fn sample_cells(&self,draws:usize) -> Vec<(usize,usize,f64)> {

        println!("Sampling cells");

        let flattened_cells: Vec<(usize,usize,f64)> = self.error_matrix.iter().enumerate().flat_map(|y| repeat(y.0).zip(y.1.iter().enumerate())).map(|(a,(b,c))| (a,b,*c)).collect();

        println!("Flattened the cells");
        println!("{}", flattened_cells.len());

        let picks = weighted_sampling(draws, &flattened_cells, &flattened_cells.iter().map(|x| x.2.abs()).collect(), false).0;

        println!("Done drawing cells");

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

        let sample_weights = matrix_flip(&self.error_matrix).iter().map(|x| x.iter().map(|y| y.abs()).sum()).collect();

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

    println!("Weighted sampling, draws, weights: {},{}", draws, weights.len());

    let mut exclusion_set: HashSet<usize> = HashSet::new();

    let mut drawn_samples: Vec<T> = Vec::with_capacity(draws);
    let mut drawn_indecies: Vec<usize> = Vec::with_capacity(draws);

    let mut weight_sum: f64 = weights.iter().sum();

    // println!("Initiated sampling");

    if replacement {

        let mut weighted_choices: Vec<f64> = (0..draws).map(|_| rng.gen_range::<f64>(0.,weight_sum)).collect();
        weighted_choices.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

        let mut descending_weight = weight_sum;

        for element in weights.iter().rev() {
            descending_weight -= *element;
            while let Some(choice) = weighted_choices.pop() {
                if choice > descending_weight {

                    if weighted_choices.len()%1000 == 0 {
                        if weighted_choices.len() > 0 {
                            // println!("{}",weighted_choices.len());
                        }
                    }

                    drawn_indecies.push(weighted_choices.len());
                }
                else {
                    weighted_choices.push(choice);
                    break
                }
            }
        }

    }

    else {

        let mut local_weights: Vec<(usize,f64)> = weights.iter().cloned().enumerate().collect();
        // println!("weight debug: {}", local_weights.len());
        // println!("weights: {:?}", local_weights.iter().take(10).collect::<Vec<&(usize,f64)>>());
        let mut maximum_weight = local_weights.iter().max_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).map(|x| x.clone()).unwrap_or((0,0.));

        // println!("draws:{}",draws);

        for i in 0..draws {

            // if i%1000 == 0 {
            //     if i > 0 {
            //         println!("{}",i);
            //     }
            // }

            let mut accumulator = 0.;

            let mut random_index = rng.gen_range::<usize>(0,local_weights.len());
            let mut current_draw = local_weights[random_index];

            while accumulator <= maximum_weight.1 {
                // println!("acc:{}",accumulator);
                accumulator += current_draw.1;
                random_index = rng.gen_range::<usize>(0,local_weights.len());
                current_draw = local_weights[random_index];
            }

            local_weights.swap_remove(random_index);

            if maximum_weight.0 == current_draw.0 {
                maximum_weight = local_weights.iter().max_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater)).map(|x| x.clone()).unwrap_or((0,0.));
            }

            weight_sum -= current_draw.1;

            if weight_sum == 0. {
                panic!("No weighted samples remaining");
            }

            drawn_indecies.push(current_draw.0);
            drawn_samples.push(samples[current_draw.0].clone());


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
