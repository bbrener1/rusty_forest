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

extern crate rand;
use rand::Rng;

use rank_table::RankTable;
use rank_table::RankTableSplitter;


impl<U:Clone + std::cmp::Eq + std::hash::Hash + Debug,T:Clone + std::cmp::Eq + std::hash::Hash + Debug> Node<U,T> {

    pub fn root(counts:&Vec<Vec<f64>>,feature_names:&[U],sample_names:&[T],input_features: Vec<U>,output_features:Vec<U>) -> Node<U,T> {

        let rank_table = RankTable::new(counts,feature_names,sample_names);

        let feature_weights = vec![1.;feature_names.len()];

        let medians = rank_table.medians();

        let dispersions = rank_table.dispersions();

        let new_node = Node {

            rank_table: rank_table,
            dropout: true,

            parent: None,
            children: Vec::new(),
            self_reference: Cell::new(None),

            feature: None,
            split: None,

            output_features: output_features,
            input_features: input_features,

            medians: medians,
            feature_weights: feature_weights,
            dispersion: Vec::new(),
        };

        let (new_node,self_reference) = generate_weak(new_node);

        new_node.self_reference.set(Some(self_reference));

        new_node

    }

    pub fn indecies(&self) -> Vec<T> {
        self.rank_table.sample_names.clone()
    }

    pub fn split(&mut self, feature: &U) -> (U,usize,T,usize,f64,f64,Vec<usize>) {

        println!("Splitting a node");

        let (forward,reverse,draw_order) = self.rank_table.split(feature);

        self.feature_weights[self.rank_table.feature_index(feature)] = 0.;



        let mut fw_dsp = vec![0.;forward.length];

        for (i,sample) in forward.enumerate() {

            // println!("{:?}",sample);
            fw_dsp[i] = sample
                .iter()
                .enumerate()
                .fold(0.,|acc,x| {
                    let mut div = (x.1).1/(x.1).0;
                    if div.is_nan() {
                        div = 0.;
                    };
                    div.powi(2) * self.feature_weights[x.0] + acc
                })
                .sqrt();

        }

        let mut rv_dsp = vec![0.;reverse.length];

        // println!("Done with forward, printing reverse");

        for (i,sample) in reverse.enumerate() {

            // println!("{:?}",sample);
            rv_dsp[i] = sample
                .iter().enumerate()
                .fold(0.,|acc,x| {
                    let mut div = (x.1).1/(x.1).0;
                    if div.is_nan() {
                        div = 0.;
                    };
                    div.powi(2) + acc
                })
                .sqrt();


        }

        rv_dsp.reverse();

        for combo in fw_dsp.iter().zip(rv_dsp.iter()) {
            println!("{:?},{}", combo, combo.0 + combo.1);
        }

        let (split_index, split_dispersion) = fw_dsp.iter().zip(rv_dsp.iter()).map(|x| x.0 + x.1).enumerate().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or((0,0.));

        let split_sample_value = self.rank_table.feature_fetch(feature, draw_order[split_index]);

        let split_sample_index = draw_order[split_index];

        let split_sample_name = self.rank_table.sample_name(split_sample_index);

        (feature.clone(),split_index,split_sample_name,split_sample_index,split_sample_value,split_dispersion,draw_order)
    }

    // pub fn best_split(&mut self) -> (U,usize,T,usize,f64,f64,Vec<usize>) {
    pub fn best_split(&mut self) -> (U,Vec<usize>,Vec<usize>) {

        if self.input_features.len() < 1 {
            panic!("Tried to split with no input features");
        };

        let first_feature = self.input_features.first().unwrap().clone();

        let mut minimum_dispersion = self.split(&first_feature);

        for feature in self.input_features.clone().iter().enumerate() {
            if feature.0 == 0 {
                continue
            }
            else {
                let current_dispersion = self.split(&feature.1);
                if current_dispersion.5 < minimum_dispersion.5 {
                    minimum_dispersion = current_dispersion;
                }
            }

        }

        self.feature = Some(minimum_dispersion.0);
        self.split = Some(minimum_dispersion.4);

        (minimum_dispersion.0,minimum_dispersion.6[..minimum_dispersion.1].iter().cloned().collect(),minimum_dispersion.6[minimum_dispersion.1..].iter().cloned().collect())

    }

    pub fn derive(&mut self, indecies: &[usize]) -> Node<U,T> {
        let new_rank_table = self.rank_table.derive(indecies);
        let parent = self.self_reference.take().clone();
        self.self_reference.set(parent.clone());

        let child = Node {
            rank_table: new_rank_table,
            dropout: self.dropout,

            parent: parent,
            children: Vec::new(),
            self_reference: Cell::new(None),

            feature: None,
            split: None,

            output_features: self.output_features.clone(),
            input_features: self.input_features.clone(),

            medians: Vec::with_capacity(self.rank_table.dimensions.0),
            feature_weights: Vec::with_capacity(self.rank_table.dimensions.0),
            dispersion: Vec::with_capacity(self.rank_table.dimensions.0),
        };

        let (child, child_reference) = generate_weak(child);
        child.self_reference.set(Some(child_reference));

        child
    }

    pub fn derive_children(&mut self) {
        let (feature, left_indecies,right_indecies) = self.best_split();
        let left_child = self.derive(&left_indecies);
        let right_child = self.derive(&right_indecies);
        println!("{:?}",left_child.indecies());
        println!("{:?}", right_child.indecies());
        self.children.push(left_child);
        self.children.push(right_child);
    }

}

pub struct Node<U:Clone + std::cmp::Eq + std::hash::Hash,T:Clone + std::cmp::Eq + std::hash::Hash> {

    pub rank_table: RankTable<U,T>,
    dropout: bool,

    parent: Option<Weak<Node<U,T>>>,
    children: Vec<Node<U,T>>,
    self_reference: Cell<Option<Weak<Node<U,T>>>>,

    feature: Option<U>,
    split: Option<f64>,

    output_features: Vec<U>,
    input_features: Vec<U>,

    medians: Vec<(usize,f64)>,
    feature_weights: Vec<f64>,
    dispersion: Vec<f64>,
}


pub fn generate_weak<T>(target:T) -> (T,Weak<T>) {
    let arc_t = Arc::new(target);
    let weak_t = Arc::downgrade(&arc_t);
    match Arc::try_unwrap(arc_t) {
        Ok(object) => return(object,weak_t),
        Err(err) => panic!("Tried to unwrap an empty reference, something went wrong with weak reference construction!")
    }
}
