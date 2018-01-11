use std;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::Weak;
use std::fs::File;
use std::io::Write;
use std::marker::PhantomData;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::cmp::max;
use std::collections::HashSet;
use std::collections::HashMap;
use std::fmt::Debug;
use std::thread;
use std::sync::mpsc;
use std::fs::OpenOptions;
use std::iter::repeat;

extern crate rand;
use rand::Rng;
use rand::seq;


use rank_table::RankTable;
use rank_table::RankTableSplitter;
use node::Node;
use thread_pool::ThreadPool;


impl Tree {

    pub fn plant_tree(counts:&Vec<Vec<f64>>,feature_names:&[String],sample_names:&[String],input_features: Vec<String>,output_features:Vec<String>,size_limit:usize,processor_limit:usize,report_address: String) -> Tree {
        let pool = ThreadPool::new(processor_limit);
        let mut root = Node::root(counts,feature_names,sample_names,input_features,output_features,pool.clone());
        let dropout = true;
        let weights = None;

        Tree{
            pool: pool,
            root: root,
            dropout: dropout,
            weights: weights,
            size_limit: size_limit,
            report_address: report_address
        }
    }

    pub fn test_splits(&mut self) {
        self.root.derive_children();
        for child in self.root.children.iter_mut() {
            child.derive_children();
            for second_children in child.children.iter_mut() {
                if second_children.internal_report().len() > 20 {
                    second_children.derive_children();
                }
            }
        }
    }

    pub fn test_parallel_splits(&mut self) {
        self.root.parallel_derive();
        for child in self.root.children.iter_mut() {
            child.parallel_derive();
        }
    }

    pub fn grow_branches(&mut self) {
        grow_branches(&mut self.root, self.size_limit,&self.report_address);
    }

    pub fn derive_from_prototype(&mut self, features:usize,samples:usize) -> Tree {

        let mut rng = rand::thread_rng();

        let input_features = rand::seq::sample_iter(&mut rng, self.root.input_features.iter().cloned(), features).expect("Couldn't generate input features");
        let output_features = self.root.output_features.clone();

        let samples = rand::seq::sample_iter(&mut rng, (0..self.root.rank_table.dimensions.0), samples).expect("Couldn't generate a subsample");

        let mut new_root = self.root.derive(&samples);

        new_root.input_features = input_features;
        new_root.output_features = output_features;

        let mut address: Vec<&str> = self.report_address.split('.').collect();
        let iteration = address.last().unwrap_or(&"0").parse::<usize>().unwrap_or(0);
        *address.last_mut().unwrap() = &(iteration+1).to_string();
        let address_string: String = address.iter().zip(repeat(".")).fold(String::new(),|mut acc,x| {acc.push_str(x.0); acc.push_str(x.1); acc});

        Tree{
            pool: self.pool.clone(),
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

    // pub fn grow_recursively(&mut self, target: ) {
    //     if target.upgrade().unwrap().internal_report().len() < self.size_limit {
    //         target.parallel_derive();
    //         for child in target.children.iter_mut() {
    //             self.grow_recursively(child);
    //         }
    //     }
    // }
    //
    // pub fn crawl_to_leaves<'a>(&'a mut self, target: &'a mut Node<U,T>) -> Vec<&'a mut Node<U,T>> {
    //     let mut output = Vec::new();
    //     if target.children.len() < 1 {
    //         return vec![target]
    //     }
    //     else {
    //         for child in target.children.iter_mut() {
    //             output.extend(self.crawl_to_leaves(child));
    //         }
    //     };
    //     output
    // }
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

}

pub struct Tree {
    pool: mpsc::Sender<((usize, (RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>,
    pub root: Node,
    dropout: bool,
    weights: Option<Vec<f64>>,
    size_limit: usize,
    report_address: String,
}

pub fn report_node_structure(node:&Node,name:&str) {
    let mut tree_dump = OpenOptions::new().create(true).append(true).open(&name).unwrap();
    tree_dump.write(node.data_dump().as_bytes());
}

pub fn grow_branches(target:&mut Node, size_limit:usize,report_address:&str) {
    if target.internal_report().len() > size_limit {
        target.parallel_derive();
        for child in target.children.iter_mut() {
            grow_branches(child, size_limit,report_address)
        }
    }
    report_node_structure(target,report_address);
}

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
