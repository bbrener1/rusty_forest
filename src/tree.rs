use std::sync::Arc;
use std::fs::File;
use std::io::Write;
use std::io::Read;

use std::sync::mpsc;
use std::fs::OpenOptions;
use std::iter::repeat;
use serde_json;
use random_forest::SampleMode;
use PredictionMode;


extern crate rand;
use rand::Rng;
use rand::seq;


use rank_table::RankTable;
use rank_table::RankTableSplitter;
use node::Node;
use node::NodeWrapper;
use thread_pool::ThreadPool;
use feature_thread_pool::FeatureThreadPool;
use rank_vector::RankVector;

impl<'a> Tree {

    pub fn plant_tree(counts:&Vec<Vec<f64>>,feature_names:&[String],sample_names:&[String],input_features: Vec<String>,output_features:Vec<String>,size_limit:usize,processor_limit:usize,report_address: String) -> Tree {
        // let pool = ThreadPool::new(processor_limit);
        let feature_pool = FeatureThreadPool::new(processor_limit);
        // let mut root = Node::root(counts,feature_names,sample_names,input_features,output_features,pool.clone());
        let root = Node::feature_root(counts,feature_names,sample_names,input_features,output_features,feature_pool.clone());
        let dropout = true;
        let weights = None;

        Tree{
            // pool: pool,
            feature_pool: feature_pool,
            root: root,
            dropout: dropout,
            weights: weights,
            size_limit: size_limit,
            report_address: report_address
        }
    }

    pub fn serialize(self) {

        self.report_summary();
        self.dump_data();

        println!("Serializing to:");
        println!("{}",self.report_address);

        let mut tree_dump = OpenOptions::new().create(true).append(true).open(self.report_address).unwrap();
        tree_dump.write(self.root.wrap_consume().to_string().as_bytes());
        tree_dump.write(b"\n");
    }

    pub fn reload(location: &str,feature_pool: mpsc::Sender<(((RankVector,Arc<Vec<usize>>),mpsc::Sender<Vec<(f64,f64)>>))>, size_limit: usize , report_address: String) -> Tree {

        println!("Reloading!");

        let mut json_file = File::open(location).expect("Deserialization error!");
        let mut json_string = String::new();
        json_file.read_to_string(&mut json_string);

        println!("{}",json_string);

        let root_wrapper: NodeWrapper = serde_json::from_str(&json_string).unwrap();

        println!("Deserialized root wrapper");

        let root = root_wrapper.unwrap(feature_pool.clone());

        println!("Finished recursive unwrapping and obtained a Node tree");

        Tree {
            feature_pool: feature_pool,
            root: root,
            dropout: true,
            weights: None,
            size_limit: size_limit,
            report_address: report_address
        }

    }


    pub fn grow_branches(&mut self) {
        grow_branches(&mut self.root, self.size_limit,&self.report_address,0);
        crawl_absolute_gains(&mut self.root,None,None);
    }

    pub fn derive_from_prototype(&mut self, features:usize,samples:usize,input_features:usize,output_features:usize,iteration: usize) -> Tree {

        let new_root = self.root.derive_from_prototype(features,samples,input_features,output_features,"RT");

        let mut address: Vec<String> = self.report_address.split('.').map(|x| x.to_string()).collect();
        *address.last_mut().unwrap() = iteration.to_string();
        let mut address_string: String = address.iter().zip(repeat(".")).fold(String::new(),|mut acc,x| {acc.push_str(x.0); acc.push_str(x.1); acc});
        address_string.pop();

        // println!("Derived from prototype, rank table size: {:?}", new_root.rank_table.dimensions);

        Tree{
            // pool: self.pool.clone(),
            feature_pool: self.feature_pool.clone(),
            root: new_root,
            dropout: self.dropout,
            weights: self.weights.clone(),
            size_limit: self.size_limit,
            report_address: address_string,
        }
    }

    pub fn nodes(&self) -> Vec<&Node> {
        let mut nodes = vec![&self.root];
        let mut finished = false;

        while !finished {
            finished = true;
            let mut new_nodes = Vec::new();
            for node in nodes {
                if node.children.len() > 0 {
                    new_nodes.append(&mut node.children.iter().collect());
                    finished = false;
                }
                else {
                    new_nodes.push(node);
                }
            }
            nodes = new_nodes;
        }
        println!("Finished crawling nodes!");
        nodes
    }

    pub fn root(&self) -> &Node {
        &self.root
    }

    pub fn dimensions(&self) -> (usize,usize) {
        self.root.dimensions()
    }

    // pub fn predict_leaves(self,meta_vector:& Vec<Vec<f64>>, header: &HashMap<String,usize> ,prediction_mode:& PredictionMode) -> Vec<&Node> {
    //
    //     node_predict_leaves(&self.root, vector, header, prediction_mode);
    //
    // }

    pub fn mut_crawl_to_leaves(&'a self, target: &'a mut Node) -> Vec<&'a mut Node> {
        let mut output = Vec::new();
        if target.children.len() < 1 {
            return vec![target]
        }
        else {
            for child in target.children.iter_mut() {
                output.extend(self.mut_crawl_to_leaves(child));
            }
        };
        output
    }

    pub fn crawl_to_leaves(&self) -> Vec<& Node> {
        crawl_to_leaves(&self.root)
    }

    pub fn crawl_nodes(&self) -> Vec<& Node> {
        crawl_nodes(&self.root)
    }

    pub fn report_summary(&self) {
        let mut tree_dump = OpenOptions::new().create(true).append(true).open([&self.report_address,".summary"].join("")).unwrap();
        for node in self.crawl_nodes() {
            tree_dump.write(node.summary().as_bytes());
        }
    }

    pub fn dump_data(&self) {
        report_node_structure(&self.root, &[&self.report_address,".dump"].join(""))
    }

}

#[derive(Clone)]
pub struct Tree {
    // pool: mpsc::Sender<((usize, (RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>,
    feature_pool: mpsc::Sender<(((RankVector,Arc<Vec<usize>>),mpsc::Sender<Vec<(f64,f64)>>))>,
    pub root: Node,
    dropout: bool,
    weights: Option<Vec<f64>>,
    size_limit: usize,
    pub report_address: String,
}

pub fn report_node_structure(target:&Node,name:&str) {
    let mut tree_dump = OpenOptions::new().create(true).append(true).open(&name).unwrap();
    for node in crawl_nodes(target) {
        tree_dump.write(node.data_dump().as_bytes());
    }
}


pub fn crawl_to_leaves<'a>(target: &'a Node) -> Vec<&'a Node> {
    let mut output = Vec::new();
    if target.children.len() < 1 {
        return vec![target]
    }
    else {
        for child in &target.children {
            output.extend(crawl_to_leaves(child));
        }
    };
    output
}

pub fn crawl_nodes<'a>(target: &'a Node) -> Vec<&'a Node> {
    let mut output = Vec::new();
    if target.children.len() < 1 {
        return vec![target]
    }
    else {
        for child in &target.children {
            output.extend(crawl_nodes(child));
        }
    };
    output.push(target);
    output
}


pub fn grow_branches(target:&mut Node, size_limit:usize,report_address:&str,level:usize) {
    if target.internal_report().len() > size_limit {
        target.feature_parallel_derive();
        for child in target.children.iter_mut() {
            grow_branches(child, size_limit,report_address, level+1);
        }
    }
    // report_node_structure(target,report_address);
}



pub fn crawl_absolute_gains<'a>(target:&'a mut Node,in_dispersions:Option<&'a Vec<f64>>,in_medians:Option<&'a Vec<f64>>) {

    let root_dispersions = in_dispersions;
    let root_medians = in_medians;

    if root_dispersions.is_none() {
        let root_dispersions = Some(&target.dispersions);
        let root_medians = Some(&target.medians);
    }
    else {
        let mut absolute_gains = Vec::with_capacity(root_dispersions.unwrap().len());

        for ((nd,nm),(od,om)) in target.dispersions.iter().zip(target.medians.iter()).zip(root_dispersions.unwrap().iter().zip(root_medians.unwrap().iter())) {
            absolute_gains.push((od/om)/(nd/nm));
        }

        target.absolute_gains = Some(absolute_gains);

    }


    for child in target.children.iter_mut() {
        crawl_absolute_gains(child, root_dispersions,root_medians);
    }

}

// #[cfg(test)]
// mod tree_tests {
//     fn test_reconstitution()
// }

// pub fn test_splits(&mut self) {
//     self.root.derive_children();
//     for child in self.root.children.iter_mut() {
//         child.derive_children();
//         for second_children in child.children.iter_mut() {
//             if second_children.internal_report().len() > 20 {
//                 second_children.derive_children();
//             }
//         }
//     }
// }
//
// pub fn test_parallel_splits(&mut self) {
//     self.root.feature_parallel_derive();
//     for child in self.root.children.iter_mut() {
//         child.feature_parallel_derive();
//     }
// }


//
// pub fn node_predict_leaves<'a>(node: &'a Node, vector: &Vec<f64>, header: &HashMap<String,usize>, prediction_mode: &PredictionMode) -> Vec<&'a Node> {
//
//     let mut leaves: Vec<&Node> = Vec::new();
//
//     if let (&Some(ref feature),&Some(ref split)) = (&node.feature,&node.split) {
//         if header.contains_key(feature) {
//             if vector[header[feature]] > split.clone() {
//                 leaves.append(&mut node_predict_leaves(&node.children[1], vector, header, prediction_mode));
//             }
//             else {
//                 leaves.append(&mut node_predict_leaves(&node.children[0], vector, header, prediction_mode));
//             }
//         }
//         else {
//             match prediction_mode {
//                 &PredictionMode::Branch => {
//                     leaves.append(&mut node_predict_leaves(&node.children[0], vector, header, prediction_mode));
//                     leaves.append(&mut node_predict_leaves(&node.children[1], vector, header, prediction_mode));
//                 },
//                 &PredictionMode::Truncate => {
//                     leaves.push(&node)
//                 },
//                 &PredictionMode::Abort => {},
//                 &PredictionMode::Auto => {
//                     leaves.append(&mut node_predict_leaves(&node, vector, header, &PredictionMode::Branch));
//                 }
//             }
//         }
//     }
//     else {
//         leaves.push(&node);
//     }
//
//     return leaves
//
// }

// pub fn sum_leaves(leaves: Vec<&Node>) -> (Vec<f64>,Vec<f64>,Vec<usize>) {
//
//     median
//
//     for
// }
//
// pub fn interval_stack(intervals: Vec<(&f64,&f64,&f64)>) -> Vec<(f64,f64,f64)> {
//     let mut aggregate_intervals: Vec<f64> = intervals.iter().fold(Vec::with_capacity(intervals.len()*2), |mut acc,x| {acc.push(*x.0); acc.push(*x.1); acc});
//     aggregate_intervals.sort_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));
//     let mut aggregate_scores = vec![0.;aggregate_intervals.len()-1];
//     for (s_start,s_end,score) in intervals {
//         for (i,(w_start,w_end)) in aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).enumerate() {
//             if (*w_start >= *s_start) && (*w_end <= *s_end) {
//                 aggregate_scores[i] += score;
//             }
//             else {
//                 aggregate_scores[i] -= score;
//             }
//         }
//     }
//     let scored = aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).zip(aggregate_scores.into_iter()).map(|((begin,end),score)| (*begin,*end,score)).collect();
//     scored
// }


// pub fn grow_recursively(&mut self, target: ) {
//     if target.upgrade().unwrap().internal_report().len() < self.size_limit {
//         target.parallel_derive();
//         for child in target.children.iter_mut() {
//             self.grow_recursively(child);
//         }
//     }
// }
//
//
// pub fn crawl_leaves<'a>(&'a mut self, target: &'a mut Node<U,T>) -> Vec<&'a mut Node<U,T>> {
//     let mut output = Vec::new();
//     if target.children.len() < 1 {
//         return vec![target]
//     }
//     else {
//         for child in target.children.iter_mut() {
//             output.extend(self.crawl_to_leaves(child));
//             output.push(&mut target);
//         }
//     };
//     output
// }
//
// pub fn weigh_leaves(&mut self) {
//     let root_dispersions = self.root.dispersions;
//     for leaf in self.crawl_leaves(&mut self.root) {
//
//         let leaf_weights = Vec::with_capacity(root_dispersions.len());
//
//         for (rv,lv) in leaf.dispersions.iter().zip(root_dispersions.iter()) {
//             if *lv != 0. && *rv != 0. {
//                 leaf_weights.push(rv)
//             }
//         }
//     }
// }


// impl<'a, U:Clone + std::cmp::Eq + std::hash::Hash + Debug,T:Clone + std::cmp::Eq + std::hash::Hash + Debug> LeafCrawler<'a, U, T> {
//
//     pub fn new(target:&'a mut Node<U,T>) -> LeafCrawler<'a,U,T> {
//         LeafCrawler{root: target}
//     }
//
//     pub fn crawl_leaves(&'a self, target: &'a mut Node<U,T>) -> Vec<&'a mut Node<U,T>> {
//         let mut output = Vec::new();
//         if target.children.len() < 1 {
//             return vec![target]
//         }
//         else {
//             for child in target.children.iter_mut() {
//                 output.extend(self.crawl_leaves(child));
//                 // output.push(&'a mut target);
//             }
//         };
//         output
//     }
//
// }
//
// pub struct LeafCrawler<'a, U:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug,T:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug> {
//     root: &'a mut Node<U,T>,
// }
