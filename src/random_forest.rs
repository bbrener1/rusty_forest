use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::io::Error;
use std::io::BufRead;
use std::io;
use std::collections::HashMap;

use tree::Tree;

extern crate rand;
use rand::seq;

use PredictionMode;
use DropMode;
use TreeBackups;
use feature_thread_pool::FeatureThreadPool;
use predictor::predict;
use matrix_flip;

impl Forest {
    pub fn initialize(counts:&Vec<Vec<f64>>,trees:usize,leaf_size:usize,processor_limit:usize, feature_option: Option<Vec<String>>, sample_option: Option<Vec<String>>, dropout: DropMode, report_address:&str) -> Forest {

        let dimensions = (counts.len(),counts.first().unwrap_or(&Vec::with_capacity(0)).len());

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let report_string = format!("{}.0",report_address).to_string();

        let prototype_tree = Tree::plant_tree(&counts,&feature_names,&sample_names,feature_names.clone(),sample_names.clone(),leaf_size, dropout ,processor_limit,report_string);

        Forest {
            feature_names: feature_names,
            sample_names: sample_names,
            trees: Vec::new(),
            size: trees,
            counts: counts.clone(),
            prototype_tree: prototype_tree,
            dropout:dropout
        }
    }

    pub fn generate(&mut self, features_per_tree:usize, samples_per_tree:usize,input_features:usize,output_features:usize, remember: bool) {
        self.prototype_tree.clone().serialize();
        for tree in 1..self.size+1 {
            let mut new_tree = self.prototype_tree.derive_from_prototype(features_per_tree,samples_per_tree,input_features,output_features,tree);
            println!("{:?}", new_tree.report_address);
            new_tree.grow_branches();
            new_tree.clone().serialize();
            if remember {
                self.trees.push(new_tree);
            }
        }
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
            prototype_tree: prototype_tree,
            trees: trees,
            counts: Vec::new(),
        })
    }

    pub fn predict(&self,feature_map: &HashMap<String,usize>,prediction_mode:&PredictionMode, drop_mode: &DropMode,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {

        let predictions = predict(&self.trees,&matrix_flip(&self.counts),feature_map,prediction_mode,drop_mode);

        let mut prediction_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction"].join("")).unwrap();
        prediction_dump.write(&format!("{:?}",predictions).as_bytes())?;
        prediction_dump.write(b"\n")?;

        let mut truth_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction_truth"].join("")).unwrap();
        truth_dump.write(&format!("{:?}",&matrix_flip(&self.counts)).as_bytes())?;
        truth_dump.write(b"\n")?;

        let mut prediction_header = OpenOptions::new().create(true).append(true).open([report_address,".prediction_header"].join("")).unwrap();
        prediction_header.write(&format!("{:?}",feature_map).as_bytes())?;
        prediction_header.write(b"\n")?;

        Ok(predictions)
    }

    pub fn trees(&self) -> &Vec<Tree> {
        &self.trees
    }

    pub fn dimensions(&self) -> (usize,usize) {
        self.prototype_tree.dimensions()
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
    size: usize,
    counts: Vec<Vec<f64>>,
    prototype_tree: Tree,
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
        Forest::initialize(&vec![], 0, 1, 1, None, None, "./testing/test_trees");
    }

    #[test]
    fn test_forest_initialization_simple() {
        let counts = vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        Forest::initialize(&counts, 1, 1, 1, Some(vec!["one".to_string()]), None, "./testing/test_trees");
    }

    #[test]
    fn test_forest_initialization_iris() {
        let counts = read_counts("./testing/iris.drop");
        let features = read_header("./testing/iris.features");
        Forest::initialize(&counts, 1, 10, 1, Some(features), None, "./testing/err");
    }

    #[test]
    fn test_forest_reconstitution_simple() {
        let new_forest = Forest::reconstitute(TreeBackups::Vector(vec!["./testing/precomputed_trees/simple.0".to_string(),"./testing/precomputed_trees/simple.1".to_string()]), None, None, Some(1), "./testing/");


        let reconstituted_features: Vec<String> = new_forest.trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        let correct_features: Vec<String> = vec!["0","0","0","0","0","0"].iter().map(|x| x.to_string()).collect();
        assert_eq!(reconstituted_features,correct_features);


        let correct_splits: Vec<f64> = vec![-1.,-2.,20.,10.,10.,5.];
        let reconstituted_splits: Vec<f64> = new_forest.trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        assert_eq!(reconstituted_splits,correct_splits);
    }


    #[test]
    fn test_forest_reconstitution() {
        let new_forest = Forest::reconstitute(TreeBackups::Vector(vec!["./testing/precomputed_trees/iris.0".to_string(),"./testing/precomputed_trees/iris.1".to_string()]), None, None, Some(1), "./testing/");


        let reconstituted_features: Vec<String> = new_forest.trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        let correct_features: Vec<String> = vec!["sepal_length","petal_length","sepal_width","sepal_width","sepal_length","sepal_width","sepal_width","sepal_width","sepal_width","sepal_width"].iter().map(|x| x.to_string()).collect();
        assert_eq!(reconstituted_features,correct_features);


        let correct_splits: Vec<f64> = vec![1.5,5.7,1.2,1.1,4.9,1.8,1.4,2.2,1.8,1.];
        let reconstituted_splits: Vec<f64> = new_forest.trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    }

    #[test]
    fn test_forest_generation() {
        let counts = read_counts("./testing/iris.drop");
        let features = read_header("./testing/iris.features");

        let mut new_forest = Forest::initialize(&counts, 1, 10, 1, Some(features), None, "./testing/tmp_test");
        new_forest.generate(4, 150, 4, 4, true);


        let computed_features: Vec<String> = new_forest.trees[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
        let correct_features: Vec<String> = vec!["sepal_length","petal_length","sepal_width","sepal_width","sepal_length","sepal_width","sepal_width","sepal_width","sepal_width","sepal_width"].iter().map(|x| x.to_string()).collect();
        assert_eq!(computed_features,correct_features);


        let computed_splits: Vec<f64> = new_forest.trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
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
