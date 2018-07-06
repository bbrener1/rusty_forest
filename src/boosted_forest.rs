use rand::{Rng,ThreadRng};
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
use node::StrippedNode;
use std::sync::mpsc;
use DropMode;
use PredictionMode;
use BoostMode;
use matrix_flip;
use sub_mtx;
use mean;
use std_dev;
use mtx_dim;
use add_mtx_ip;
use tsv_format;
use Parameters;
use pearsonr;


pub struct BoostedForest {
    leaf_size: usize,
    processor_limit: usize,
    report_string: String,

    feature_names: Vec<String>,
    sample_names: Vec<String>,

    feature_similarity_matrix: Vec<Vec<f64>>,
    similarity_observation_matrix: Vec<Vec<usize>>,
    cell_coocurrence_matrix: Vec<Vec<f64>>,

    error_matrix: Vec<Vec<f64>>,

    counts: Vec<Vec<f64>>,
    trees: Vec<Tree>,
    predictive_trees: Vec<PredictiveTree>,

    dimensions: (usize,usize),
    epoch_duration: usize,
    epochs: usize,
    current_epoch:usize,

    prototype_rank_table: Option<RankTable>,
    prototype_tree: Option<Tree>,

    dropout: DropMode,
    prediction_mode: PredictionMode,
    boost_mode: BoostMode,
}

impl BoostedForest {

    pub fn initialize(counts:&Vec<Vec<f64>>, parameters: Arc<Parameters>, report_address:&str) -> BoostedForest {

        let dimensions: (usize,usize) = (counts.len(),counts.first().unwrap_or(&vec![]).len());

        let feature_names = parameters.feature_names.clone().unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = parameters.sample_names.clone().unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let feature_similarity_matrix = vec![vec![0.;dimensions.0];dimensions.0];
        let similarity_observation_matrix = vec![vec![0;dimensions.0];dimensions.0];
        let cell_coocurrence_matrix = vec![vec![0.;dimensions.1];dimensions.1];

        let error_matrix: Vec<Vec<f64>>;

        match parameters.boost_mode.unwrap_or(BoostMode::Subsampling) {
            BoostMode::Additive => error_matrix = counts.clone(),
            BoostMode::Subsampling => error_matrix = vec![vec![1.;dimensions.1];dimensions.0],
        }

        let report_string = format!("{}",report_address).to_string();

        let prototype_table = RankTable::new(&counts,&feature_names,&sample_names,parameters.clone());

        BoostedForest {
            trees: Vec::new(),
            leaf_size: parameters.leaf_size_cutoff.unwrap_or(1),
            processor_limit: parameters.processor_limit.unwrap_or(1),
            report_string: report_string,
            feature_names: feature_names,
            sample_names: sample_names,

            feature_similarity_matrix: feature_similarity_matrix,
            similarity_observation_matrix: similarity_observation_matrix,
            cell_coocurrence_matrix: cell_coocurrence_matrix,
            error_matrix: error_matrix,

            predictive_trees: Vec::new(),
            dimensions: dimensions,
            epoch_duration: parameters.epoch_duration.unwrap_or(1),
            epochs: parameters.epochs.unwrap_or(1),
            current_epoch: 0,
            counts: counts.clone(),
            prototype_rank_table: Some(prototype_table),
            prototype_tree: None,
            boost_mode: parameters.boost_mode.unwrap_or(BoostMode::Subsampling),
            dropout: parameters.dropout.unwrap_or(DropMode::Zeros),
            prediction_mode: parameters.prediction_mode.unwrap_or(PredictionMode::Truncate)
        }
    }

    pub fn grow_forest(&mut self, parameters: Arc<Parameters>) -> Result<(),Error>{

        for i in 0..self.epochs {

            let report_string = &[self.report_string.clone(),format!(".{}",i)].join("");

            self.grow_epoch(parameters.clone(), report_string, i);

            let epoch_predictions = self.compact_predict(&self.counts, &self.feature_map(), parameters.clone(), report_string)?;

            self.predict_epoch(i,&self.counts, &self.feature_map(), parameters.clone(), report_string)?;

            self.error_matrix = sub_mtx(&self.counts, &matrix_flip(&epoch_predictions));

            self.update_similarity(report_string)?;

            // println!("Error matrix dimensions:{:?}",mtx_dim(&self.error_matrix));

        }

        Ok(())
    }

    pub fn grow_epoch (&mut self, parameters: Arc<Parameters>, report_string: &str ,epoch:usize) {

        println!("Initializing epoch {}", epoch);

        println!("Drawing weights");

        let output_features_per_tree = parameters.output_features.unwrap_or(1);
        let input_features_per_tree = parameters.input_features.unwrap_or(1);
        let samples_per_tree = parameters.sample_subsample.unwrap_or(1);

        let (input_feature_weights, output_feature_weights, sample_weights) = self.draw_weights(report_string);

        println!("Weights drawn");

        let mut tree_receivers = Vec::with_capacity(self.epoch_duration);

        let mut prototype_tree = match self.boost_mode {
            BoostMode::Additive => Tree::prototype_tree(&self.counts,&self.error_matrix,&self.sample_names,&self.feature_names,&self.feature_names, Some(output_feature_weights.clone()), parameters, [self.report_string.clone(),format!(".{}.0",epoch)].join("")),
            BoostMode::Subsampling => Tree::prototype_tree(&self.counts,&self.counts,&self.sample_names,&self.feature_names,&self.feature_names, Some(output_feature_weights.clone()), parameters, [self.report_string.clone(),format!(".{}.0",epoch)].join("")),
        };

        let mut thread_pool = BoostedTreeThreadPool::new(&prototype_tree,self.processor_limit);

        for i in 0..self.epoch_duration {

            println!("Tree {}", i);

            let samples = weighted_sampling(samples_per_tree, &self.sample_names, &sample_weights, false).0;

            println!("Samples drawn");

            let (input_features,output_features) = self.inputs_and_outputs(input_features_per_tree, output_features_per_tree, input_feature_weights.clone(), output_feature_weights.clone());

            println!("Features drawn");

            let output_scoring_weights = self.error_matrix.iter().map(|x| x.iter().sum::<f64>()).collect::<Vec<f64>>();

            let (tx,rx) = mpsc::channel();

            thread_pool.send(BoostedMessage::Selections(i,input_features,output_features,samples,output_scoring_weights,tx));

            tree_receivers.push(rx);

        }

        for receiver in tree_receivers {
            let tree = receiver.recv().expect("Failed to unwrap boosted tree");
            tree.serialize_compact();
            self.predictive_trees.push(tree);
        }

        self.current_epoch += 1;

        BoostedTreeThreadPool::terminate(&mut thread_pool);
        prototype_tree.terminate_pool();

    }

    pub fn predict_epoch(&self,epoch:usize,counts:&Vec<Vec<f64>>,feature_map: &HashMap<String,usize>,parameters: Arc<Parameters>,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {

        let trees = (epoch*self.epoch_duration,((epoch+1)*self.epoch_duration)-(epoch+1));

        println!("Predicting epoch: {}, trees: {}-{}", epoch,trees.0,trees.1);

        let predictions = compact_predict(&self.predictive_trees[trees.0..trees.1],&matrix_flip(counts),feature_map,parameters);

        let mut prediction_dump = OpenOptions::new().create(true).append(true).open([report_address,&format!(".{}",epoch),".e_prediction"].join("")).unwrap();
        prediction_dump.write(&tsv_format(&predictions).as_bytes())?;
        prediction_dump.write(b"\n")?;

        let mut truth_dump = OpenOptions::new().create(true).append(true).open([report_address,&format!(".{}",epoch),".e_prediction_truth"].join("")).unwrap();
        truth_dump.write(&tsv_format(&matrix_flip(&self.counts)).as_bytes())?;
        truth_dump.write(b"\n")?;

        let mut header_vec = vec!["";feature_map.len()];
        for (f,i) in feature_map { header_vec[*i] = f; };
        let header = tsv_format(&vec![header_vec]);

        let mut prediction_header = OpenOptions::new().create(true).append(true).open([report_address,&format!(".{}",epoch),".e_prediction_header"].join("")).unwrap();
        prediction_header.write(&header.as_bytes())?;
        prediction_header.write(b"\n")?;

        Ok(predictions)
    }

    pub fn compact_predict(&self,counts:&Vec<Vec<f64>>,feature_map: &HashMap<String,usize>,parameters: Arc<Parameters>,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {

        println!("Predicting:");

        let predictions = match self.boost_mode {
            BoostMode::Additive => {
                let mut epoch_sums = vec![vec![0.;self.dimensions.0];self.dimensions.1];
                for epoch in 0..self.current_epoch {
                    epoch_sums = add_mtx_ip(epoch_sums, &self.predict_epoch(epoch, counts, feature_map, parameters.clone(), report_address).expect("Failure predicting epoch"));
                }
                epoch_sums
            },
            BoostMode::Subsampling => {
                compact_predict(&self.predictive_trees,&matrix_flip(counts),feature_map,parameters.clone())
            }

        };

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
        // println!("{},{}",input_draws,output_draws);

        let mut input_features = Vec::with_capacity(input_draws);

        for _ in 0..input_draws {
            // println!("weights: {:?},{:?}", input_feature_weights,output_feature_weights);

            let feature_index = weighted_sampling(1, &self.features(), &input_feature_weights, false).1[0];

            // println!("fi:{}", feature_index);

            // input_feature_weights =
            //     input_feature_weights
            //     .iter()
            //     .zip(
            //         self.feature_similarity_matrix[feature_index]
            //         .iter()
            //     )
            //     .map(|(x,y)| x * (1. - (y/5.)))
            //     .collect();
            input_feature_weights[feature_index] = 0.;
            output_feature_weights[feature_index] = 0.;

            input_features.push(self.features()[feature_index].clone())
        }

        println!("Inputs drawn");

        let mut output_features = Vec::with_capacity(output_draws);

        for _ in 0..output_draws {

            // println!("weights: {:?},{:?}", input_feature_weights,output_feature_weights);

            let feature_index = weighted_sampling(1, &self.features(), &output_feature_weights, false).1[0];

            // println!("fi:{}", feature_index);

            output_feature_weights =
                output_feature_weights
                .iter()
                .zip(
                    self.feature_similarity_matrix[feature_index]
                    .iter()
                )
                .map(|(x,y)| x * (1. + y/2.))
                .collect();
            output_feature_weights[feature_index] = 0.;

            output_features.push(self.features()[feature_index].clone())
        }

        // let output_features = weighted_sampling(output_draws, &self.features(), &output_feature_weights, false).0.iter().cloned().collect();

        println!("Outputs drawn");

        (input_features,output_features)
    }

    pub fn sample_cells(&self,draws:usize) -> Vec<(usize,usize,f64)> {

        println!("Sampling cells");

        let flattened_cells: Vec<(usize,usize,f64)> = self.error_matrix.iter().enumerate().flat_map(|y| repeat(y.0).zip(y.1.iter().enumerate())).map(|(a,(b,c))| (a,b,*c)).collect();

        println!("Flattened the cells");
        println!("{}", flattened_cells.len());

        println!("Drawing {}", draws);

        let picks = weighted_sampling(draws, &flattened_cells, &flattened_cells.iter().map(|x| x.2.abs()).collect(), false).0;

        println!("Done drawing cells");

        picks
    }


    pub fn draw_weights(&self,report_address: &str) -> (Vec<f64>,Vec<f64>,Vec<f64>) {

        // println!("Drawing weights");
        //
        // let error_cells = self.sample_cells(cells);
        //
        // println!("Error cells drawn");

        // let mut output_feature_weights = vec![1.;self.features().len()];
        //
        // for (j,cell) in error_cells.iter().enumerate() {
        //     output_feature_weights[cell.0] += 1.;
        // };

        // let mut output_feature_weights: Vec<f64> = self.error_matrix.iter().map(|x| x.iter().map(|y| y.abs()).sum()).collect();

        let mut output_feature_weights = vec![1.;self.dimensions.0];

        println!("Output feature weights drawn: {}", output_feature_weights.len());

        let mut input_feature_weights = vec![1.;self.dimensions.0];

        // let input_feature_weights: Vec<f64> = {
        //     self.counts
        //         .iter()
        //         .map(|x| x
        //             .iter()
        //             .map(|y| {
        //                 match self.dropout {
        //                     DropMode::Zeros => if *y == 0. {0.} else {1.},
        //                     DropMode::NaNs => if y.is_nan() {0.} else {1.},
        //                     DropMode::No => 1.,
        //                 }
        //
        //             })
        //             .sum()
        //         )
        //         .collect::<Vec<f64>>()
        //     };

        println!("Input feature weights drawn: {}", input_feature_weights.len());

        let mut sample_weights = vec![1.; self.dimensions.1];

        // let sample_weights: Vec<f64> = matrix_flip(&self.error_matrix).iter().map(|x| x.iter().map(|y| y.abs()).sum::<f64>()).collect();

        println!("Sample weights drawn: {}", sample_weights.len());

        let mut i_weight_dump = OpenOptions::new().create(true).append(true).open([report_address,".i_weights"].join("")).unwrap();
        i_weight_dump.write(&tsv_format(&vec![input_feature_weights.clone()]).as_bytes()).expect("weight_reporting_error");
        i_weight_dump.write(b"\n").expect("weight reporting error");

        let mut o_weight_dump = OpenOptions::new().create(true).append(true).open([report_address,".o_weights"].join("")).unwrap();
        o_weight_dump.write(&tsv_format(&vec![output_feature_weights.clone()]).as_bytes()).expect("weight reporting error");
        o_weight_dump.write(b"\n").expect("weight reporting error");

        let mut s_weight_dump = OpenOptions::new().create(true).append(true).open([report_address,".s_weights"].join("")).unwrap();
        s_weight_dump.write(&tsv_format(&vec![sample_weights.clone()]).as_bytes()).expect("weight reporting error");
        s_weight_dump.write(b"\n").expect("weight reporting error");

        (input_feature_weights,output_feature_weights,sample_weights)

    }

    pub fn features(&self) -> &Vec<String> {
        &self.feature_names
    }

    pub fn samples(&self) -> &Vec<String> {
        &self.sample_names
    }

    fn update_similarity(&mut self,report_address:&str) -> Result<(),Error> {

        // let nodes: Vec<&StrippedNode> = self.predictive_trees.iter().flat_map(|x| x.crawl_nodes()).collect();

        let feature_map = self.feature_map();

        // let local_gains: Vec<Vec<(&String,f64)>> = nodes.iter().map(|node| node.features().iter().zip(node.local_gains.as_ref().unwrap_or(&vec![]).iter().cloned()).collect()).collect();

        let mut heirarchal_absolute_gains = Vec::with_capacity(self.predictive_trees.len());

        for tree in self.predictive_trees.iter() {
            let tree_nodes = tree.crawl_to_leaves();
            let tree_gains = tree_nodes.iter().map(|node| node.absolute_gains().as_ref().map(|node| node.clone()).unwrap_or(vec![])).collect();
            heirarchal_absolute_gains.push((tree.root.features(),tree_gains));
        }


        // let absolute_gains: Vec<Vec<(&String,f64)>> = nodes.iter().map(|node| node.features().iter().zip(node.absolute_gains.as_ref().unwrap_or(&vec![]).iter().cloned()).collect()).collect();

        // println!("{:?}",local_gains);
        // println!("Local gains gathered");
        println!("Absolute gains gathered");


        let observed_correlations = observe_correlations(heirarchal_absolute_gains, feature_map);

        println!("Updating similarities");

        for (f1,f2,correlation) in observed_correlations {
            let previous_correlation = self.feature_similarity_matrix[f1][f2];
            self.feature_similarity_matrix[f1][f2] = ((previous_correlation * (self.similarity_observation_matrix[f1][f2] as f64)) + correlation) / ((self.similarity_observation_matrix[f1][f2] + 1) as f64);
            self.similarity_observation_matrix[f1][f2] += 1;
        }

        // self.feature_similarity_matrix = incomplete_correlation_matrix(absolute_gains, feature_map);

        let mut similarity_dump = OpenOptions::new().create(true).append(true).open([report_address,".similarity"].join("")).unwrap();
        similarity_dump.write(&tsv_format(&self.feature_similarity_matrix).as_bytes())?;
        similarity_dump.write(b"\n")?;
        // println!("similarity: {:?}", self.feature_similarity_matrix);

        Ok(())
    }

}

pub fn weighted_choice(weights: &Vec<(usize,f64)>, weight_sum: f64, rng: &mut ThreadRng) -> usize {

    let choice = rng.gen::<f64>() * weight_sum;

    let mut descending_weight = weight_sum;

    for (i,(_,element)) in weights.iter().enumerate() {
        descending_weight -= *element;
        // println!("descending:{}",descending_weight);
        if choice > descending_weight {
            // println!("choice:{}",choice);

            return i
        }
    }

    0
}

pub fn weighted_sampling<T: Clone>(draws: usize, samples: &Vec<T>, weights: &Vec<f64>,replacement:bool) -> (Vec<T>,Vec<usize>) {

    let mut rng = thread_rng();

    // println!("Weighted sampling, draws, weights: {},{}", draws, weights.len());

    let mut exclusion_set: HashSet<usize> = HashSet::new();

    let mut drawn_samples: Vec<T> = Vec::with_capacity(draws);
    let mut drawn_indecies: Vec<usize> = Vec::with_capacity(draws);

    let mut weight_sum: f64 = weights.iter().sum();

    // println!("Initiated sampling");
    // println!("weights:{:?}", weights);

    if replacement {

        let mut weighted_choices: Vec<f64> = (0..draws).map(|_| rng.gen_range::<f64>(0.,weight_sum)).collect();
        weighted_choices.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

        // println!("sum: {}", weight_sum);
        // println!("choices: {:?}", weighted_choices);

        let mut descending_weight = weight_sum;

        for (i,element) in weights.iter().enumerate() {
            descending_weight -= *element;
            // println!("descending:{}",descending_weight);
            while let Some(choice) = weighted_choices.pop() {
                if choice > descending_weight {
                    // println!("choice:{}",choice);

                    if weighted_choices.len()%1000 == 0 {
                        if weighted_choices.len() > 0 {
                            // println!("{}",weighted_choices.len());
                        }
                    }

                    drawn_indecies.push(i);
                    drawn_samples.push(samples[i].clone());
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
        maximum_weight = (maximum_weight.0,maximum_weight.1 * 2.);

        for i in 0..draws {

            // if i%1000 == 0 {
            //     if i > 0 {
            //         println!("{}",i);
            //     }
            // }
            if weight_sum == 0. {
                panic!("No weighted samples remaining");
            }

            let mut accumulator = 0.;

            let mut random_index = rng.gen_range::<usize>(0,local_weights.len());
            let mut current_draw = local_weights[random_index];

            // println!("max:{:?}", maximum_weight);

            while accumulator < maximum_weight.1 {
                random_index = rng.gen_range::<usize>(0,local_weights.len());
                current_draw = local_weights[random_index];
                accumulator += current_draw.1;
                // println!("acc:{}",accumulator);
                // println!("curr: {:?}", current_draw);
            }

            local_weights.swap_remove(random_index);

            // println!("loc: {:?}", local_weights);

            if maximum_weight.0 == current_draw.0 {
                maximum_weight = local_weights.iter().max_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater)).map(|x| x.clone()).unwrap_or((0,0.));
            }

            weight_sum -= current_draw.1;

            drawn_indecies.push(current_draw.0);
            drawn_samples.push(samples[current_draw.0].clone());


        }

    }

    // println!("drawn: {:?}",drawn_indecies);

    (drawn_samples,drawn_indecies)

}

pub fn observe_correlations(local_gains: Vec<(&Vec<String>,Vec<Vec<f64>>)>, map: HashMap<String,usize>) -> Vec<(usize,usize,f64)> {

    println!("Observing correlations");

    let mut correlations = Vec::with_capacity(map.len());

    for (features, tree_nodes) in local_gains {
        let mut feature_sums = vec![0.;features.len()];
        let mut feature_square_sums = vec![0.;features.len()];
        let mut multiplied_sums = vec![vec![0.;features.len()];features.len()];

        let len = tree_nodes.len() as f64;

        for node in tree_nodes {
            for (i,feature_gain) in node.iter().enumerate() {
                feature_sums[i] += feature_gain;
                feature_square_sums[i] += feature_gain.powi(2);
                for (j,second_feature_gain) in node.iter().enumerate() {
                    multiplied_sums[i][j] += feature_gain * second_feature_gain;
                }
            }
        }

        for (i,f1) in features.iter().enumerate() {
            for (j,f2) in features.iter().enumerate() {

                let product_expectation = multiplied_sums[i][j]/len;

                let f1e = feature_sums[i]/len;
                let f2e = feature_sums[j]/len;

                let f1std = (feature_square_sums[i]/len - f1e.powi(2)).sqrt();
                let f2std = (feature_square_sums[j]/len - f2e.powi(2)).sqrt();

                let mut correlation = (product_expectation - (f1e * f2e)) / (f1std * f2std);

                // println!("{}",correlation);

                if correlation.is_nan() {
                    correlation = 0.;
                }

                correlations.push((map[f1],map[f2],correlation));
            }
        }

        if correlations.len()%1000 == 0 {
            println!("Observations: {}", correlations.len());
        }

    }

    correlations

}

// pub fn incomplete_correlation_matrix(values:Vec<Vec<(&String,f64)>>,map:HashMap<String,usize>) -> Vec<Vec<f64>> {
//
//     println!("Computing incomplete matrix");
//
//     // println!("{:?}",values);
//     // println!("{:?}",map);
//
//
//     let mut mtx: Vec<Vec<Option<f64>>> = vec![vec![None; map.len()];values.len()];
//
//     for (i, top_vector) in values.iter().enumerate() {
//         for (feature,value) in top_vector {
//             mtx[i][map[*feature]] = Some(*value);
//         }
//     }
//
//     // println!("{:?}",mtx);
//
//     let mtx_t = matrix_flip(&mtx);
//
//     // println!("{:?}",mtx_t);
//
//     let features: Vec<&String> = map.keys().collect();
//
//     let mut correlations = vec![vec![1.;features.len()];features.len()];
//
//     for f1 in &features {
//         for f2 in &features {
//
//             let f1i = map[*f1];
//             let f2i = map[*f2];
//
//             let mut multiplied_sum = 0.;
//             let mut f1s = 0.;
//             let mut f2s = 0.;
//             let mut f1ss = 0.;
//             let mut f2ss = 0.;
//             let mut len = 0.;
//
//             for (f1vo,f2vo) in mtx_t[f1i].iter().zip(mtx_t[f2i].iter()) {
//                 if let (Some(f1v),Some(f2v)) = (f1vo,f2vo) {
//
//                     // println!("values:");
//                     //
//                     // println!("{:?}", (f1v,f2v));
//
//                     multiplied_sum += f1v*f2v;
//
//                     // println!("{:?}", f1v * f2v);
//
//
//                     f1s += f1v;
//                     f2s += f2v;
//
//                     f1ss += f1v.powi(2);
//                     f2ss += f2v.powi(2);
//
//                     // println!("{:?}", f1ss * f2ss);
//
//                     len += 1.;
//                 }
//             }
//
//             // println!("{:?}", (multiplied_sum,f1s,f2s,f1ss,f2ss,len));
//
//             let product_expectation = multiplied_sum/len;
//             let f1e = f1s/len;
//             let f2e = f2s/len;
//
//             // println!("prod exp:{:?}", product_expectation);
//             // println!("exp: {:?},{:?}", f1e,f2e);
//
//             let f1std = (f1ss/len - f1e.powi(2)).sqrt();
//             let f2std = (f2ss/len - f2e.powi(2)).sqrt();
//
//             // println!("{:?}",(f1std,f2std));
//
//             let mut correlation = (product_expectation - (f1e * f2e)) / (f1std * f2std);
//
//             // println!("{}",correlation);
//
//             if correlation.is_nan() {
//                 correlation = 0.;
//             }
//
//             correlations[f1i][f2i] = correlation;
//
//             // println!("{:?}",pearsonr(&f1v,&f2v))
//         }
//
//
//     }
//
//     correlations
//
// }


#[cfg(test)]
mod raw_vector_tests {

    use super::*;

    #[test]
    fn test_weighted_sampling_with_replacement() {

        let samples = &vec!["a","b","c","d","e","f","g","h","i","j"];

        let weights =  &vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.];

        let mut draws: Vec<&str> = Vec::with_capacity(999);

        for _ in 0..333 {
            draws.extend(weighted_sampling(3, samples, weights, true).0.iter());
        }

        println!("{:?}",draws);

        panic!();

    }

    #[test]
    fn test_weighted_sampling_without_replacement() {

        let samples = &vec!["a","b","c","d","e","f","g","h","i","j"];

        let weights =  &vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.];

        let mut draws: Vec<&str> = Vec::with_capacity(999);

        for _ in 0..333 {
            draws.extend(weighted_sampling(3, samples, weights, false).0.iter());
        }

        println!("{:?}",draws);

        panic!();

    }

    #[test]
    fn test_weighted_sampling_without_replacement_subtle() {

        let samples = &vec!["a","b","c"];

        let weights =  &vec![40.,50.,25.];

        let mut draws: Vec<&str> = Vec::with_capacity(1000);

        for _ in 0..1000 {
            draws.extend(weighted_sampling(1, samples, weights, false).0.iter());
        }

        println!("{:?}",draws);

        panic!();

    }

    // #[test]
    // fn test_incomplete_similarity_matrix() {
    //
    //     let mtx = vec![
    //         vec![("a",1.),("b",2.),("d",5.)],
    //         vec![("b",2.),("c",3.),("d",5.)],
    //         vec![("a",2.),("b",4.),("c",5.),("d",5.)]
    //     ];
    //
    //     println!("mtx:");
    //     println!("{:?}",mtx);
    //
    //     let map: HashMap<String,usize> = vec![("a",0),("b",1),("c",2),("d",3)].iter().map(|(f,i)| (f.to_string(),*i)).collect();
    //
    //     println!("{:?}" , incomplete_correlation_matrix(mtx, map));
    //
    //     panic!()
    //
    // }

}


// let mut local_weights: Vec<(usize,f64)> = weights.iter().cloned().enumerate().collect();
// // println!("weight debug: {}", local_weights.len());
// // println!("weights: {:?}", local_weights.iter().take(10).collect::<Vec<&(usize,f64)>>());
// let mut maximum_weight = local_weights.iter().max_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).map(|x| x.clone()).unwrap_or((0,0.));
//
// // println!("draws:{}",draws);
//
// for i in 0..draws {
//
//     // if i%1000 == 0 {
//     //     if i > 0 {
//     //         println!("{}",i);
//     //     }
//     // }
//     if weight_sum == 0. {
//         panic!("No weighted samples remaining");
//     }
//
//     let mut accumulator = 0.;
//
//     let mut random_index = rng.gen_range::<usize>(0,local_weights.len());
//     let mut current_draw = local_weights[random_index];
//
//     // println!("max:{:?}", maximum_weight);
//
//     while accumulator < maximum_weight.1 {
//         random_index = rng.gen_range::<usize>(0,local_weights.len());
//         current_draw = local_weights[random_index];
//         accumulator += current_draw.1;
//         // println!("acc:{}",accumulator);
//         // println!("curr: {:?}", current_draw);
//     }
//
//     local_weights.swap_remove(random_index);
//
//     // println!("loc: {:?}", local_weights);
//
//     if maximum_weight.0 == current_draw.0 {
//         maximum_weight = local_weights.iter().max_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater)).map(|x| x.clone()).unwrap_or((0,0.));
//     }
//
//     weight_sum -= current_draw.1;
//
//     drawn_indecies.push(current_draw.0);
//     drawn_samples.push(samples[current_draw.0].clone());
//
//
// }
