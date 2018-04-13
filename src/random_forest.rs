use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::io::Error;
use std::io::BufRead;
use std::io;
use std::collections::HashMap;
use std::sync::mpsc;

use tree::Tree;
use tree::PredictiveTree;

extern crate rand;
use rand::seq;

use PredictionMode;
use DropMode;
use TreeBackups;
use feature_thread_pool::FeatureThreadPool;
use tree_thread_pool::TreeThreadPool;
use std::sync::mpsc::sync_channel;
use predictor::predict;
use compact_predictor::compact_predict;
use matrix_flip;
use tsv_format;

impl Forest {
    pub fn initialize(counts:&Vec<Vec<f64>>,trees:usize,leaf_size:usize,processor_limit:usize, feature_option: Option<Vec<String>>, sample_option: Option<Vec<String>>, dropout: DropMode, report_address:&str) -> Forest {

        let dimensions = (counts.len(),counts.first().unwrap_or(&vec![]).len());

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let report_string = format!("{}.0",report_address).to_string();

        let prototype_tree = Tree::prototype_tree(&counts,&counts,&sample_names,&feature_names,&feature_names,leaf_size, dropout ,1,report_string);

        println!("Name dimensions");
        println!("{}",feature_names.len());
        println!("{}",sample_names.len());

        prototype_tree.serialize_compact();

        Forest {
            feature_names: feature_names,
            sample_names: sample_names,
            trees: Vec::new(),
            predictive_trees: Vec::new(),
            size: trees,
            counts: counts.clone(),
            prototype_tree: Some(prototype_tree),
            processor_limit: processor_limit,
            dropout:dropout
        }
    }

    pub fn generate(&mut self, features_per_tree:usize, samples_per_tree:usize,input_features:usize,output_features:usize, remember: bool) {

        if let Some(ref prototype) = self.prototype_tree {

            let mut tree_receivers = Vec::with_capacity(self.size);

            let mut tree_pool = TreeThreadPool::new(self.prototype_tree.as_ref().unwrap(),features_per_tree,samples_per_tree,input_features,output_features, self.processor_limit);

            for tree in 1..self.size+1 {

                let (tx,rx) = mpsc::channel();

                tree_pool.send((tree,tx));

                tree_receivers.push(rx);

            }

            for receiver in tree_receivers {
                println!("Unwrapping tree");
                let new_tree = receiver.recv().unwrap();
                new_tree.serialize_compact();
                if remember {
                    self.predictive_trees.push(new_tree);
                }

            }

            TreeThreadPool::terminate(&mut tree_pool);

        }
        else {
            panic!("Attempted to generate a forest without a prototype tree. Are you trying to do predictions after reloading from compact backups?")
        }

        self.prototype_tree = None;

    }

    pub fn compact_reconstitute(tree_locations: TreeBackups, feature_option: Option<Vec<String>>,sample_option:Option<Vec<String>>,processor_option: Option<usize>, report_address:&str) -> Result<Forest,Error> {

        let mut predictive_trees: Vec<PredictiveTree>;

        let feature_pool = FeatureThreadPool::new(processor_option.unwrap_or(1));


        match tree_locations {
            TreeBackups::File(location) => {
                let tree_file = File::open(location)?;
                let mut tree_locations: Vec<String> = io::BufReader::new(&tree_file).lines().map(|x| x.expect("Tree location error!")).collect();
                predictive_trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    predictive_trees.push(PredictiveTree::reload(&loc,1,"".to_string())?);
                }
            }
            TreeBackups::Vector(tree_locations) => {
                predictive_trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    predictive_trees.push(PredictiveTree::reload(&loc,1,"".to_string())?);
                }
            }
            TreeBackups::Trees(backup_trees) => {
                predictive_trees = backup_trees.iter().map(|tree| tree.strip()).collect();
            }
        }


        let prototype_tree = predictive_trees.remove(0);

        let dimensions = (0,0);

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let report_string = format!("{}.reconstituted.0",report_address).to_string();

        Ok (Forest {
            feature_names: feature_names,
            sample_names: sample_names,
            size: predictive_trees.len(),
            dropout: prototype_tree.root.dropout(),
            prototype_tree: None,
            processor_limit: processor_option.unwrap_or(1),
            trees: Vec::new(),
            predictive_trees: predictive_trees,
            counts: Vec::new(),
        })

    }

    pub fn reconstitute(tree_locations: TreeBackups, feature_option: Option<Vec<String>>,sample_option:Option<Vec<String>>,processor_option: Option<usize>, report_address:&str) -> Result<Forest,Error> {

        let mut trees: Vec<Tree>;

        let feature_pool = FeatureThreadPool::new(processor_option.unwrap_or(1));

        match tree_locations {
            TreeBackups::File(location) => {
                let tree_file = File::open(location)?;
                let mut tree_locations: Vec<String> = io::BufReader::new(&tree_file).lines().map(|x| x.expect("Tree location error!")).collect();
                trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    trees.push(Tree::reload(&loc,feature_pool.clone(),1,"".to_string())?);
                }
            }
            TreeBackups::Vector(tree_locations) => {
                trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    trees.push(Tree::reload(&loc,feature_pool.clone(),1,"".to_string())?);
                }
            }
            TreeBackups::Trees(backup_trees) => {
                trees = backup_trees;
            }
        }

        let prototype_tree = trees.remove(0);

        let dimensions = prototype_tree.dimensions();

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let report_string = format!("{}.reconstituted.0",report_address).to_string();


        Ok (Forest {
            feature_names: feature_names,
            sample_names: sample_names,
            size: trees.len(),
            dropout: prototype_tree.dropout(),
            prototype_tree: Some(prototype_tree),
            processor_limit: processor_option.unwrap_or(1),
            trees: trees,
            predictive_trees: Vec::new(),
            counts: Vec::new(),
        })
    }

    pub fn predict(&self,counts:&Vec<Vec<f64>>,feature_map: &HashMap<String,usize>,prediction_mode:&PredictionMode, drop_mode: &DropMode,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {

        println!("Predicting:");

        let predictions = predict(&self.trees,&matrix_flip(counts),feature_map,prediction_mode,drop_mode);

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
        prediction_header.write(header.as_bytes())?;
        prediction_header.write(b"\n")?;

        Ok(predictions)
    }

    pub fn compact_predict(&self,counts:&Vec<Vec<f64>>,feature_map: &HashMap<String,usize>,prediction_mode:&PredictionMode, drop_mode: &DropMode,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {

        println!("Predicting:");

        let predictions = compact_predict(&self.predictive_trees,&matrix_flip(counts),feature_map,prediction_mode,drop_mode);

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

    pub fn trees(&self) -> &Vec<Tree> {
        &self.trees
    }

    pub fn predictive_trees(&self) -> &Vec<PredictiveTree> {
        &self.predictive_trees
    }

    pub fn dimensions(&self) -> (usize,usize) {
        self.prototype_tree.as_ref().unwrap().dimensions()
    }

    pub fn feature_map(&self) -> HashMap<String,usize> {
        self.feature_names.clone().into_iter().enumerate().map(|x| (x.1,x.0)).collect()
    }

}

pub enum SampleMode {
    Map(Vec<HashMap<String,f64>>),
    VectorHeader(Vec<Vec<f64>>,HashMap<String,usize>),
}

pub struct Forest {
    feature_names: Vec<String>,
    sample_names: Vec<String>,
    trees: Vec<Tree>,
    predictive_trees: Vec<PredictiveTree>,
    size: usize,
    counts: Vec<Vec<f64>>,
    prototype_tree: Option<Tree>,
    processor_limit: usize,
    dropout: DropMode
}

fn split_shuffle<T>(source_vector: Vec<T>, pieces: usize) -> Vec<Vec<T>> {

    let piece_length = source_vector.len()/pieces;
    let mut len = source_vector.len();

    let mut rng = rand::thread_rng();

    let mut shuffled_source = seq::sample_iter(&mut rng, source_vector.into_iter(), len).unwrap_or(vec![]);

    if shuffled_source.len() < 1 {
        panic!("Failed to shuffle features correctly!")
    }

    let mut vector_pieces: Vec<Vec<T>> = Vec::with_capacity(pieces);

    for _ in 0..pieces {

        len -= piece_length;

        vector_pieces.push(shuffled_source.split_off(len))
    }

    vector_pieces
}

#[cfg(test)]
mod random_forest_tests {

    use super::*;
    use super::super::{read_counts,read_header};
    use std::fs::remove_file;

    #[test]
    fn test_forest_initialization_trivial() {
        Forest::initialize(&vec![], 0, 1, 1, None, None, DropMode::No, "./testing/test_trees");
    }

    #[test]
    fn test_forest_initialization_simple() {
        let counts = vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        Forest::initialize(&counts, 1, 1, 1, Some(vec!["one".to_string()]), None, DropMode::Zeros, "./testing/test_trees");
    }

    #[test]
    fn test_forest_initialization_iris() {
        let counts = read_counts("./testing/iris.drop");
        let features = read_header("./testing/iris.features");
        Forest::initialize(&counts, 1, 10, 1, Some(features), None, DropMode::Zeros,"./testing/err");
    }

    #[test]
    fn test_forest_initialization_iris_nan() {
        let counts = read_counts("./testing/iris.nan");
        let features = read_header("./testing/iris.features");
        Forest::initialize(&counts, 1, 10, 1, Some(features), None, DropMode::NaNs,"./testing/err");
    }


    #[test]
    fn test_forest_reconstitution_simple() {
        let new_forest = Forest::compact_reconstitute(TreeBackups::Vector(vec!["./testing/precomputed_trees/simple.0.compact".to_string(), "./testing/precomputed_trees/simple.1.compact".to_string()]), None, None, Some(1),"./testing/").expect("Reconstitution test");

        println!("Reconstitution successful");

        let reconstituted_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        let correct_features: Vec<String> = vec!["0","0","0","0","0","0"].iter().map(|x| x.to_string()).collect();
        assert_eq!(reconstituted_features,correct_features);


        let correct_splits: Vec<f64> = vec![-1.,-2.,20.,10.,10.,5.];
        let reconstituted_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        assert_eq!(reconstituted_splits,correct_splits);
    }


    #[test]
    fn test_forest_reconstitution() {
        let new_forest = Forest::compact_reconstitute(TreeBackups::Vector(vec!["./testing/precomputed_trees/iris.0.compact".to_string(),"./testing/precomputed_trees/iris.1.compact".to_string()]), None, None, Some(1), "./testing/").expect("Reconstitution test");

        println!("Reconstitution successful");

        let reconstituted_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        let correct_features: Vec<String> = vec!["sepal_length","petal_length","sepal_width","sepal_width","sepal_length","sepal_width","sepal_width","sepal_width","sepal_width","sepal_width"].iter().map(|x| x.to_string()).collect();
        assert_eq!(reconstituted_features,correct_features);


        let correct_splits: Vec<f64> = vec![1.5,5.7,1.2,1.1,4.9,1.8,1.4,2.2,1.8,1.];
        let reconstituted_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    }

    #[test]
    fn test_forest_generation() {

        let counts = read_counts("./testing/iris.drop");
        let features = read_header("./testing/iris.features");

        let mut new_forest = Forest::initialize(&counts, 1, 10, 1, Some(features), None, DropMode::Zeros, "./testing/tmp_test");
        new_forest.generate(4, 150, 4, 4, true);


        let computed_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        let correct_features: Vec<String> = vec!["sepal_length","petal_length","sepal_width","sepal_width","sepal_length","sepal_width","sepal_width","sepal_width","sepal_width","sepal_width"].iter().map(|x| x.to_string()).collect();
        assert_eq!(computed_features,correct_features);


        let computed_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        let correct_splits: Vec<f64> = vec![1.5,5.7,1.2,1.1,4.9,1.8,1.4,2.2,1.8,1.];
        assert_eq!(computed_splits,correct_splits);


        remove_file("./testing/tmp_test.0");
        remove_file("./testing/tmp_test.0.summary");
        remove_file("./testing/tmp_test.0.dump");
        remove_file("./testing/tmp_test.1");
        remove_file("./testing/tmp_test.1.summary");
        remove_file("./testing/tmp_test.1.dump");
    }
}
