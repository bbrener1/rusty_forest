use std;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::Weak;
use std::marker::PhantomData;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::cmp::max;
use std::collections::HashSet;
use std::collections::HashMap;
use std::fmt::Debug;
use std::thread;

extern crate rand;
use rand::Rng;

use rank_table::RankTable;
use rank_table::RankTableSplitter;
use node::Node;
use thread_pool::ThreadPool;


impl<U:Clone + std::cmp::Eq + std::hash::Hash + Debug,T:Clone + std::cmp::Eq + std::hash::Hash + Debug> Tree<U,T> {

    pub fn plant_tree(counts:&Vec<Vec<f64>>,feature_names:&[U],sample_names:&[T],input_features: Vec<U>,output_features:Vec<U>,size_limit:usize) -> Tree<U,T> {
        let root = Node::root(counts,feature_names,sample_names,input_features,output_features);
        let dropout = true;
        let nodes = Vec::new();
        let weights = None;

        Tree{
            root: root,
            dropout: dropout,
            nodes: nodes,
            weights: weights,
            size_limit: size_limit,
        }
    }

    pub fn grow_branches(&mut self, target: &mut Node<U,T>) {
        if target.internal_report().len() < self.size_limit {
            target.derive_children();
            for child in target.children.iter_mut() {
                self.grow_branches(child);
            }
        }
    }

    pub fn crawl_to_leaves<'a>(&'a mut self, target: &'a mut Node<U,T>) -> Vec<&'a mut Node<U,T>> {
        let mut output = Vec::new();
        if target.children.len() < 1 {
            return vec![target]
        }
        else {
            for child in target.children.iter_mut() {
                output.extend(self.crawl_to_leaves(child));
            }
        };
        output
    }
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

pub struct Tree<U:Clone + std::cmp::Eq + std::hash::Hash + Debug,T:Clone + std::cmp::Eq + std::hash::Hash + Debug> {
    root: Node<U,T>,
    dropout: bool,
    nodes: Vec<Weak<Node<U,T>>>,
    weights: Option<Vec<f64>>,
    size_limit: usize,
}


impl<'a, U:Clone + std::cmp::Eq + std::hash::Hash + Debug,T:Clone + std::cmp::Eq + std::hash::Hash + Debug> LeafCrawler<'a, U, T> {

    pub fn new(target:&'a mut Node<U,T>) -> LeafCrawler<'a,U,T> {
        LeafCrawler{root: target}
    }

    pub fn crawl_leaves(&'a self, target: &'a mut Node<U,T>) -> Vec<&'a mut Node<U,T>> {
        let mut output = Vec::new();
        if target.children.len() < 1 {
            return vec![target]
        }
        else {
            for child in target.children.iter_mut() {
                output.extend(self.crawl_leaves(child));
                // output.push(&'a mut target);
            }
        };
        output
    }

}

pub struct LeafCrawler<'a, U:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug,T:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug> {
    root: &'a mut Node<U,T>,
}
