use rand::Rng;
use std::iter::repeat;
use std::collections::HashSet;
use std::collections::HashMap;
use std::io::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::cmp::Ordering;
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
use add_matrix;
use sub_matrix;
use multiply_matrix;
use zero_matrix;
use mtx_dim;
use tsv_format;
use boosted_forest::weighted_sampling;


pub struct AdditiveBooster {
    leaf_size: usize,
    processor_limit: usize,
    report_string: String,

    feature_names: Vec<String>,
    sample_names: Vec<String>,

    feature_similarity_matrix: Vec<Vec<f64>>,
    cell_coocurrence_matrix: Vec<Vec<f64>>,

    error_matrix: Vec<Vec<f64>>,

    counts: Vec<Vec<f64>>,
    predictive_trees: Vec<Vec<PredictiveTree>>,

    dimensions: (usize,usize),
    epoch_duration: usize,
    epochs: usize,

    prototype_rank_table: Option<RankTable>,
    prototype_tree: Option<Tree>,

    dropout: DropMode,
    prediction_mode: PredictionMode,
}

impl AdditiveBooster {

    pub fn initialize(counts:&Vec<Vec<f64>>,epoch_duration:usize,leaf_size:usize,epochs:usize,processor_limit:usize, feature_option: Option<Vec<String>>, sample_option: Option<Vec<String>>, dropout: DropMode, prediction_mode: PredictionMode, report_address:&str) -> AdditiveBooster {

        let dimensions: (usize,usize) = (counts.len(),counts.first().unwrap_or(&vec![]).len());

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let feature_similarity_matrix = vec![vec![0.;dimensions.0];dimensions.0];
        let cell_coocurrence_matrix = vec![vec![0.;dimensions.1];dimensions.1];

        let report_string = format!("{}.0",report_address).to_string();

        let prototype_table = RankTable::new(&counts,&feature_names,&sample_names,dropout);

        AdditiveBooster {
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


    pub fn additive_growth(&mut self) -> Result<(),Error> {

        for i in 0..self.epochs {

            let epoch_trees = self.add_epoch(800, 400, 1000, i);

            self.predictive_trees.push(epoch_trees);

            let epoch_predictions = self.compact_predict(&self.counts, &self.feature_map(), &self.prediction_mode, &self.dropout, &[self.report_string.clone(),format!(".{}",i)].join(""))?;

            self.error_matrix = sub_matrix(&self.counts, &matrix_flip(&epoch_predictions));

            println!("Error matrix dimensions:{:?}",mtx_dim(&self.error_matrix));

        };

        Ok(())
    }

    pub fn add_epoch(&mut self,samples_per_tree:usize,input_features_per_tree:usize,output_features_per_tree:usize,epoch:usize) -> Vec<PredictiveTree> {

        println!("Initializing an epoch");

        let mut p_trees = Vec::with_capacity(self.epoch_duration);

        let mut prototype_tree =  Tree::prototype_tree(&self.counts,&matrix_flip(&self.error_matrix),&self.sample_names,&self.feature_names,&self.feature_names,self.leaf_size, self.dropout, 1, [self.report_string.clone(),format!(".{}.0",epoch)].join(""));

        println!("Epoch prototype done, drawing weights");

        let (input_feature_weights, output_feature_weights, sample_weights) = self.draw_weights(output_features_per_tree * samples_per_tree);

        println!("Weights drawn");

        let mut thread_pool = BoostedTreeThreadPool::new(&prototype_tree,self.processor_limit);

        let mut tree_receivers = Vec::with_capacity(self.epoch_duration);

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
            p_trees.push(tree);
        }

        BoostedTreeThreadPool::terminate(&mut thread_pool);
        prototype_tree.terminate_pool();

        p_trees
    }



    pub fn compact_predict(&self,counts:&Vec<Vec<f64>>,feature_map: &HashMap<String,usize>,prediction_mode:&PredictionMode, drop_mode: &DropMode,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {

        println!("Predicting:");

        let mut aggregate_predictions = Vec::with_capacity(self.predictive_trees.len());

        for p_tree_epoch in &self.predictive_trees {

            let predictions = compact_predict(p_tree_epoch,&matrix_flip(counts),feature_map,prediction_mode,drop_mode,self.processor_limit);

            aggregate_predictions = add_matrix(&aggregate_predictions,&predictions);

        }

        let mut prediction_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction"].join("")).unwrap();
        prediction_dump.write(&tsv_format(&aggregate_predictions).as_bytes())?;
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

        Ok(aggregate_predictions)


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