use std::fs::File;
use std::io;
use std::io::BufRead;
use std::collections::HashMap;

use tree::Tree;
use node::Node;

extern crate rand;
use rand::Rng;
use rand::seq;

use PredictionMode;
use TreeBackups;
use feature_thread_pool::FeatureThreadPool;

impl Forest {
    pub fn initialize(counts:&Vec<Vec<f64>>,trees:usize,leaf_size:usize,processor_limit:usize, feature_option: Option<Vec<String>>, sample_option: Option<Vec<String>>, report_address:&str) -> Forest {

        let dimensions = (counts.len(),counts.first().unwrap_or(&Vec::with_capacity(0)).len());

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let report_string = format!("{}.0",report_address).to_string();

        let prototype_tree = Tree::plant_tree(&counts,&feature_names,&sample_names,feature_names.clone(),sample_names.clone(),leaf_size,processor_limit,report_string);

        Forest {
            feature_names: feature_names,
            sample_names: sample_names,
            trees: Vec::new(),
            size: trees,
            counts: Vec::new(),
            prototype_tree: prototype_tree,
            dropout: true
        }
    }

    pub fn generate(&mut self, features_per_tree:usize, samples_per_tree:usize,input_features:usize,output_features:usize) {
        self.prototype_tree.clone().serialize();
        for tree in 1..self.size+1 {
            let mut new_tree = self.prototype_tree.derive_from_prototype(features_per_tree,samples_per_tree,input_features,output_features,tree);
            println!("{:?}", new_tree.report_address);
            new_tree.grow_branches();
            new_tree.serialize();
            // self.trees.push(new_tree);
        }
    }

    pub fn reconstitute(tree_locations: TreeBackups, feature_option: Option<Vec<String>>,sample_option:Option<Vec<String>>,processor_option: Option<usize>,report_address:&str) -> Forest {

        let mut trees: Vec<Tree> = Vec::with_capacity(0);

        let feature_pool = FeatureThreadPool::new(processor_option.unwrap_or(1));

        match tree_locations {
            TreeBackups::File(location) => {
                let tree_file = File::open(location).expect("Count file error!");
                let mut tree_locations: Vec<String> = io::BufReader::new(&tree_file).lines().map(|x| x.expect("Tree location error!")).collect();
                trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    trees.push(Tree::reload(&loc,feature_pool.clone(),1,"".to_string()));
                }
            }
            TreeBackups::Vector(tree_locations) => {
                trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    trees.push(Tree::reload(&loc,feature_pool.clone(),1,"".to_string()));
                }
            }
            TreeBackups::Trees(backup_trees) => {
                trees = backup_trees;
            }
        }

        let prototype_tree = trees.remove(0);

        let dimensions = (prototype_tree.root().features().len(),prototype_tree.root().samples().len());

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let report_string = format!("{}.0",report_address).to_string();


        Forest {
            feature_names: feature_names,
            sample_names: sample_names,
            size: trees.len(),
            prototype_tree: trees.remove(0),
            trees: trees,
            counts: Vec::new(),
            dropout: true
        }
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
    dropout: bool
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
