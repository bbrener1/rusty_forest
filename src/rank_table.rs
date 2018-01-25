use std;
use std::marker::PhantomData;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::cmp::max;
use std::collections::HashSet;
use std::collections::HashMap;
use std::fmt::Debug;
use time;

extern crate rand;
use rand::Rng;

use rank_vector::RankVector;
use rank_vector::OrderedDraw;
use rank_vector::ProceduralDraw;

impl RankTable {

    pub fn new<'a> (counts: &Vec<Vec<f64>>,feature_names:&'a [String],sample_names:&'a [String]) -> RankTable {

        let mut meta_vector = Vec::new();

        let mut feature_dictionary: HashMap<String,usize> = HashMap::with_capacity(feature_names.len());

        let mut sample_dictionary: HashMap<String,usize> = sample_names.iter().cloned().enumerate().map(|x| (x.1,x.0)).collect();

        for (i,(name,loc_counts)) in feature_names.iter().cloned().zip(counts.iter()).enumerate() {
            if i%200 == 0 {
                println!("Initializing: {}",i);
            }
            // println!("Starting to iterate");
            feature_dictionary.insert(name.clone(),i);
            // println!("Updated feature dict");
            let mut construct = RankVector::new(loc_counts,name);
            // println!("Made a rank vector");
            construct.drop_zeroes();
            construct.initialize();
            construct.set_boundaries();
            meta_vector.push(construct);
        }

        let draw_order = (0..counts[0].len()).collect::<Vec<usize>>();

        let dim = (meta_vector.len(),meta_vector[0].vector.vector.len());

        println!("Made rank table with {} features, {} samples:", dim.0,dim.1);

        RankTable{meta_vector:meta_vector,feature_names:feature_names.iter().cloned().collect(),sample_names:sample_names.iter().cloned().collect(),draw_order:draw_order,index:0,dimensions:dim, feature_dictionary: feature_dictionary, sample_dictionary: sample_dictionary}

    }

    pub fn medians(&self) -> Vec<f64> {
        self.meta_vector.iter().map(|x| x.median()).collect()
    }

    pub fn dispersions(&self) -> Vec<f64> {
        self.meta_vector.iter().map(|x| x.mad()).collect()
    }

    pub fn trunc_iterate(&mut self) -> RankTableIter {
        let limit = self.dimensions.1;
        RankTableIter::new(self,limit)
    }

    pub fn sort_by_feature(& self, feature: &str) -> Vec<usize> {
        self.meta_vector[self.feature_dictionary[feature]].give_dropped_order()
    }

    pub fn feature_name(&self, feature_index: usize) -> String {
        self.feature_names[feature_index].clone()
    }

    pub fn feature_index(&self, feature_name: &str) -> usize {
        self.feature_dictionary[feature_name]
    }

    pub fn feature_fetch(&self, feature: &str, index: usize) -> f64 {
        self.meta_vector[self.feature_dictionary[feature]].vector[index].3
    }

    pub fn sample_name(&self, index:usize) -> String {
        self.sample_names[index].clone()
    }

    pub fn sample_index(&self, sample_name: &str) -> usize {
        self.sample_dictionary[sample_name]
    }

    pub fn split(&self, feature: &str) -> (RankTableSplitter,RankTableSplitter, Vec<usize>) {
        RankTableSplitter::split(self,feature)
    }

    pub fn between(&self, feature: &str,begin:&str,end:&str) -> usize {
        self.meta_vector[self.feature_dictionary[feature]].between(self.sample_dictionary[begin],self.sample_dictionary[end])
    }

    pub fn full_values(&self) -> Vec<Vec<f64>> {
        let mut values = Vec::new();
        for feature in &self.meta_vector {
            values.push(feature.draw_values());
        }
        values
    }

    pub fn full_ordered_values(&self) -> Vec<Vec<f64>> {
        self.meta_vector.iter().map(|x| x.draw_ordered_values()).collect()
    }

    pub fn samples(&self) -> &[String] {
        &self.sample_names[..]
    }


    pub fn derive(&self, indecies:&[usize]) -> RankTable {

        let mut new_meta_vector: Vec<RankVector> = Vec::with_capacity(indecies.len());

        let new_sample_dictionary: HashMap<String,usize> = indecies.iter().enumerate().map(|x| (self.sample_names[*x.1].clone(),x.0)).collect();

        let mut new_sample_names: Vec<String> = Vec::with_capacity(indecies.len());

        for sample_name in &self.sample_names {
            if new_sample_dictionary.contains_key(sample_name) {
                new_sample_names.push(sample_name.clone());
            }
        }

        for feature in &self.meta_vector {
            new_meta_vector.push(feature.derive(indecies));
        }

        let new_draw_order: Vec<usize> = (0..indecies.len()).collect();

        let dimensions = (self.meta_vector.len(),self.meta_vector[0].vector.vector.len());

        RankTable {

            meta_vector: new_meta_vector,
            feature_names: self.feature_names.clone(),
            sample_names: new_sample_names,
            feature_dictionary: self.feature_dictionary.clone(),
            sample_dictionary: new_sample_dictionary,
            draw_order: new_draw_order,
            index: 0,
            dimensions: dimensions,
        }
    }


    pub fn derive_from_prototype(&self, features:usize,samples:usize) -> RankTable {

        let mut rng = rand::thread_rng();

        let indecies = rand::seq::sample_iter(&mut rng, 0..self.sample_names.len(), samples).expect("Couldn't generate sample subset");

        // println!("Derive debug {},{}", samples, indecies.len());

        let mut new_meta_vector: Vec<RankVector> = Vec::with_capacity(features);

        let mut new_sample_dictionary : HashMap<String,usize> = indecies.iter().enumerate().map(|(count,index)| (self.sample_names[*index].clone(),count)).collect();
        let mut new_sample_names = indecies.iter().map(|index| self.sample_names[*index].clone()).collect();

        let mut new_feature_dictionary = HashMap::with_capacity(features);
        let mut new_feature_names = Vec::with_capacity(features);

        for (i,feature) in rand::seq::sample_iter(&mut rng, self.feature_names.iter().enumerate(), features).expect("Couldn't process feature during subsampling") {
            new_meta_vector.push(self.meta_vector[i].derive(&indecies));
            new_feature_names.push(feature.clone());
            new_feature_dictionary.insert(feature.clone(),new_feature_names.len()-1);
        }

        let new_draw_order: Vec<usize> = (0..indecies.len()).collect();

        let dimensions = (new_meta_vector.len(),new_meta_vector[0].vector.vector.len());

        println!("Feature dict {:?}", new_feature_dictionary.clone());
        println!("New sample dict {:?}", new_sample_dictionary.clone());

        RankTable {

            meta_vector: new_meta_vector,
            feature_names: new_feature_names,
            sample_names: new_sample_names,
            feature_dictionary: new_feature_dictionary,
            sample_dictionary: new_sample_dictionary,
            draw_order: new_draw_order,
            index: 0,
            dimensions: dimensions,
        }
    }

}


#[derive(Debug,Clone)]
pub struct RankTable {
    meta_vector: Vec<RankVector>,
    pub feature_names: Vec<String>,
    pub sample_names: Vec<String>,
    feature_dictionary: HashMap<String,usize>,
    sample_dictionary: HashMap<String,usize>,
    draw_order: Vec<usize>,
    index: usize,
    pub dimensions: (usize,usize),
}

impl<'a> RankTableIter<'a> {
    pub fn new(rank_table: &mut RankTable,limit:usize) -> RankTableIter {

        // println!("Starting new meta-iterator:");

        let mut table = Vec::new();

        for vector in rank_table.meta_vector.iter_mut() {
            table.push(vector.ordered_draw());
        }

        // println!("Finished making iterators, yielding meta-iterator");

        RankTableIter{table:table,index:0,limit:limit-1,current_sample:None}
    }

}

impl<'a> Iterator for RankTableIter<'a> {
    type Item = Vec<(f64,f64)>;

    fn next(&mut self) -> Option<Vec<(f64,f64)>> {

        if self.index > self.limit {
            return None
        }

        let mut output = Vec::new();

        for (i,draw) in self.table.iter_mut().enumerate() {
            let io = draw.next();
            // println!("{},{},{},{:?}",i,self.index,self.limit,io);

            output.push(io.unwrap_or((0.,0.)));
        }

        self.index += 1;



        Some(output)
    }
}

pub struct RankTableIter<'a> {
    table: Vec<OrderedDraw<'a>>,
    index: usize,
    current_sample: Option<String>,
    limit: usize,
}

impl RankTableSplitter {
    pub fn new(rank_table: & RankTable,feature:&str) -> RankTableSplitter {

        let draw_order = rank_table.sort_by_feature(feature);

        let length = draw_order.len();

        // println!("Starting new meta-iterator:");

        let mut table = Vec::new();

        for vector in rank_table.meta_vector.iter() {
            table.push(vector.clone().consumed_draw());
        }

        // println!("Finished making iterators, yielding meta-iterator");

        RankTableSplitter{table:table,index:0,draw_order:draw_order, length: length, current_sample:None}
    }

    pub fn split(rank_table: & RankTable, feature:&str) -> (RankTableSplitter,RankTableSplitter,Vec<usize>) {
        let mut forward_splitter = RankTableSplitter::new(rank_table,feature);
        let draw_order = forward_splitter.draw_order.clone();
        let mut reverse_splitter = forward_splitter.clone();
        reverse_splitter.draw_order.reverse();

        (forward_splitter,reverse_splitter,draw_order)
    }

}



impl Iterator for RankTableSplitter {
    type Item = Vec<(f64,f64)>;

    fn next(&mut self) -> Option<Vec<(f64,f64)>> {

        // let start_time = time::PreciseTime::now();

        if self.index > self.length - 1 {
            return None
        }

        let mut output = Vec::with_capacity(self.table.len());

        let target = self.draw_order[self.index];

        for draw in self.table.iter_mut() {
            let io = draw.next(target);
            // println!("{},{},{},{:?}",i,self.index,self.limit,io);

            output.push(io.unwrap_or((0.,0.)));
        }

        self.index += 1;

        // let end_time = time::PreciseTime::now();

        // println!("Time to serve a single splitter iteration {}", start_time.to(end_time).num_nanoseconds().unwrap_or(-1));

        Some(output)
    }
}

#[derive(Clone)]
pub struct RankTableSplitter {
    table: Vec<ProceduralDraw>,
    draw_order: Vec<usize>,
    index: usize,
    current_sample: Option<String>,
    pub length: usize,
}
