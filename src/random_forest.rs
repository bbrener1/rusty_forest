
use tree::Tree;

extern crate rand;
use rand::Rng;
use rand::seq;


impl Forest {
    pub fn initialize(counts:Vec<Vec<f64>>,trees:usize,leaf_size:usize,processor_limit:usize, feature_option: Option<Vec<String>>, sample_option: Option<Vec<String>>, report_address:&str) -> Forest {

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

    pub fn generate(&mut self, features_per_tree:usize, samples_per_tree:usize) {
        for tree in 0..self.size {
            let mut new_tree = self.prototype_tree.derive_from_prototype(features_per_tree,samples_per_tree,features_per_tree,features_per_tree,tree);
            println!("{:?}", new_tree.report_address);
            new_tree.grow_branches();
            self.trees.push(new_tree);
        }
    }

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
