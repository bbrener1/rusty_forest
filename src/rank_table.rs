
use std::collections::HashSet;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::sync::Arc;
use std::mem::swap;
use std::sync::mpsc;
use feature_thread_pool::FeatureMessage;
extern crate rand;


use rank_vector::RankVector;
use DropMode;

impl RankTable {

    pub fn new<'a> (counts: &Vec<Vec<f64>>,feature_names:&'a [String],sample_names:&'a [String],dropout:DropMode) -> RankTable {

        let mut meta_vector = Vec::new();

        let mut feature_dictionary: HashMap<String,usize> = HashMap::with_capacity(feature_names.len());

        let sample_dictionary: HashMap<String,usize> = sample_names.iter().cloned().enumerate().map(|x| (x.1,x.0)).collect();

        for (i,(name,loc_counts)) in feature_names.iter().cloned().zip(counts.iter()).enumerate() {
            if i%200 == 0 {
                println!("Initializing: {}",i);
            }
            // println!("Starting to iterate");
            feature_dictionary.insert(name.clone(),i);
            // println!("Updated feature dict");
            let mut construct = RankVector::new(loc_counts,name,dropout);
            // println!("Made a rank vector");
            construct.drop();
            construct.initialize();
            construct.set_boundaries();
            construct.backup();
            meta_vector.push(construct);
        }

        let draw_order = (0..counts.get(0).unwrap_or(&vec![]).len()).collect::<Vec<usize>>();

        let dim = (meta_vector.len(),meta_vector.get(0).unwrap_or(&RankVector::new(&vec![],"".to_string(),DropMode::No)).vector.vector.len());

        println!("Made rank table with {} features, {} samples:", dim.0,dim.1);

        RankTable {
            meta_vector:meta_vector,
            feature_names:feature_names.iter().cloned().collect(),
            sample_names:sample_names.iter().cloned().collect(),
            draw_order:draw_order,
            index:0,
            dimensions:dim,
            feature_dictionary: feature_dictionary,
            sample_dictionary: sample_dictionary,
            dropout:dropout,
        }

    }

    pub fn medians(&self) -> Vec<f64> {
        self.meta_vector.iter().map(|x| x.median()).collect()
    }

    pub fn dispersions(&self) -> Vec<f64> {
        self.meta_vector.iter().map(|x| x.mad()).collect()
    }

    pub fn covs(&self) -> Vec<f64> {
        self.dispersions().into_iter().zip(self.dispersions().into_iter()).map(|x| x.0/x.1).map(|y| if y.is_nan() {0.} else {y}).collect()
    }

    pub fn sort_by_feature(& self, feature: &str) -> (&Vec<usize>,&HashSet<usize>) {
        //
        // println!("Dropped: {:?}", self.meta_vector[self.feature_dictionary[feature]].give_dropped_order());
        // println!("Full: {:?}", self.meta_vector[self.feature_dictionary[feature]].give_draw_order());
        //
        self.meta_vector[self.feature_dictionary[feature]].give_draw_order()
    }

    pub fn split_indecies_by_feature(&self, feature: &str, split: &f64) -> (Vec<usize>,Vec<usize>){
        self.meta_vector[self.feature_dictionary[feature]].split_indecies(split)
    }

    pub fn feature_name(&self, feature_index: usize) -> Option<&String> {
        self.feature_names.get(feature_index)
    }

    pub fn feature_index(&self, feature_name: &str) -> Option<&usize> {
        self.feature_dictionary.get(feature_name)
    }

    pub fn feature_fetch(&self, feature: &str, index: usize) -> f64 {
        self.meta_vector[self.feature_dictionary[feature]].vector[index].3
    }

    pub fn features(&self) -> &Vec<String> {
        &self.feature_names
    }

    pub fn sample_name(&self, index:usize) -> String {
        self.sample_names[index].clone()
    }

    pub fn sample_index(&self, sample_name: &str) -> usize {
        self.sample_dictionary[sample_name]
    }


    pub fn between(&self, feature: &str,begin:&str,end:&str) -> usize {
        self.meta_vector[self.feature_dictionary[feature]].crawl_between(self.sample_dictionary[begin],self.sample_dictionary[end])
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

        let index_set: HashSet<&usize> = indecies.iter().collect();

        let mut new_sample_dictionary: HashMap<String,usize> = HashMap::with_capacity(indecies.len());

        let mut new_sample_names: Vec<String> = Vec::with_capacity(indecies.len());

        for (i,sample_name) in self.sample_names.iter().enumerate() {
            if index_set.contains(&i) {
                new_sample_names.push(sample_name.clone());
                new_sample_dictionary.insert(sample_name.clone(),new_sample_names.len()-1);
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
            dropout: self.dropout,
        }
    }


    pub fn cloned_features(&self) -> Vec<RankVector> {
        self.meta_vector.clone()
    }

    pub fn drain_features(&mut self) -> Vec<RankVector> {

        let mut out = Vec::new();
        swap(&mut self.meta_vector,&mut out);
        out
    }

    pub fn return_features(&mut self, returned: Vec<RankVector>) {
        self.meta_vector = returned;
    }

    pub fn derive_specified(&self, features:&Vec<&String>,samples:&Vec<&String>) -> RankTable {

        let indecies: Vec<usize> = samples.iter().map(|x| self.sample_index(x)).collect();
        let index_set: HashSet<&usize> = indecies.iter().collect();

        let mut new_meta_vector: Vec<RankVector> = Vec::with_capacity(features.len());

        let new_sample_names:Vec<String> = samples.iter().cloned().cloned().collect();
        let new_sample_dictionary : HashMap<String,usize> = new_sample_names.iter().enumerate().map(|(count,sample)| (sample.clone(),count)).collect();

        let mut new_feature_dictionary = HashMap::with_capacity(features.len());
        let mut new_feature_names = Vec::with_capacity(features.len());

        for (i,feature) in features.iter().cloned().enumerate() {
            new_meta_vector.push(self.meta_vector[self.feature_dictionary[feature]].derive(&indecies));
            new_feature_names.push(feature.clone());
            new_feature_dictionary.insert(feature.clone(),new_feature_names.len()-1);
        }

        let new_draw_order: Vec<usize> = (0..indecies.len()).collect();

        let dimensions = (new_meta_vector.len(),new_meta_vector.get(0).unwrap_or(&RankVector::empty()).vector.vector.len());

        RankTable {

            meta_vector: new_meta_vector,
            feature_names: new_feature_names,
            sample_names: new_sample_names,
            feature_dictionary: new_feature_dictionary,
            sample_dictionary: new_sample_dictionary,
            draw_order: new_draw_order,
            index: 0,
            dimensions: dimensions,
            dropout: self.dropout,

        }

    }

    pub fn derive_random(&self, features:usize,samples:usize) -> RankTable {

        let mut rng = rand::thread_rng();

        let indecies = rand::seq::sample_iter(&mut rng, 0..self.sample_names.len(), samples).expect("Couldn't generate sample subset");

        let index_set: HashSet<&usize> = indecies.iter().collect();

        // println!("Derive debug {},{}", samples, indecies.len());

        let mut new_meta_vector: Vec<RankVector> = Vec::with_capacity(features);

        let new_sample_names: Vec<String> = self.sample_names.iter().enumerate().filter(|x| index_set.contains(&x.0)).map(|x| x.1).cloned().collect();
        let new_sample_dictionary : HashMap<String,usize> = new_sample_names.iter().enumerate().map(|(count,sample)| (sample.clone(),count)).collect();

        let mut new_feature_dictionary = HashMap::with_capacity(features);
        let mut new_feature_names = Vec::with_capacity(features);

        for (i,feature) in rand::seq::sample_iter(&mut rng, self.feature_names.iter().enumerate(), features).expect("Couldn't process feature during subsampling") {
            new_meta_vector.push(self.meta_vector[i].derive(&indecies));
            new_feature_names.push(feature.clone());
            new_feature_dictionary.insert(feature.clone(),new_feature_names.len()-1);
        }

        let new_draw_order: Vec<usize> = (0..indecies.len()).collect();

        let dimensions = (new_meta_vector.len(),new_meta_vector.get(0).unwrap_or(&RankVector::empty()).vector.vector.len());

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
            dropout: self.dropout,
        }
    }



    pub fn parallel_split_order(&mut self,draw_order:&Vec<usize>, drop_set: &HashSet<usize>,feature_weights:&Vec<f64>,  pool:mpsc::Sender<FeatureMessage>) -> Option<(usize,f64)> {

        let forward_draw = Arc::new(draw_order.clone());
        let mut reverse_draw: Arc<Vec<usize>> = Arc::new(draw_order.iter().cloned().rev().collect());

        let drop_arc = Arc::new(drop_set.clone());

        if forward_draw.len() < 2 {
            return None
        }

        let mut forward_covs: Vec<Vec<f64>> = vec![vec![0.;self.dimensions.0];forward_draw.len()];
        let mut reverse_covs: Vec<Vec<f64>> = vec![vec![0.;self.dimensions.0];reverse_draw.len()];

        let mut forward_receivers = Vec::with_capacity(self.dimensions.0);
        let mut reverse_receivers = Vec::with_capacity(self.dimensions.0);

        for feature in self.meta_vector.drain(..) {
            let (tx,rx) = mpsc::channel();
            pool.send(FeatureMessage::Message((feature,forward_draw.clone(),drop_arc.clone()),tx));
            forward_receivers.push(rx);
        }

        for (i,fr) in forward_receivers.iter().enumerate() {
            if let Ok((disp,feature)) = fr.recv() {
                for (j,(m,d)) in disp.into_iter().enumerate() {
                    // forward_covs[j][i] = (d/m).abs();
                    forward_covs[j][i] = d.abs();
                    if forward_covs[j][i].is_nan(){
                        forward_covs[j][i] = 0.;
                    }
                }
                self.meta_vector.push(feature);
            }
            else {
                panic!("Parellelization error!")
            }

        }

        for feature in self.meta_vector.drain(..) {
            let (tx,rx) = mpsc::channel();
            pool.send(FeatureMessage::Message((feature,reverse_draw.clone(),drop_arc.clone()),tx));
            reverse_receivers.push(rx);
        }

        for (i,rr) in reverse_receivers.iter().enumerate() {
            if let Ok((disp,feature)) = rr.recv() {
                for (j,(m,d)) in disp.into_iter().enumerate() {
                    // reverse_covs[reverse_draw.len() - j - 1][i] = (d/m).abs();
                    reverse_covs[reverse_draw.len() - j - 1][i] = d.abs();
                    if reverse_covs[reverse_draw.len() - j - 1][i].is_nan(){
                        reverse_covs[reverse_draw.len() - j - 1][i] = 0.;
                    }
                }
                self.meta_vector.push(feature);
            }
            else {
                panic!("Parellelization error!")
            }

        }

        Some(mad_minimum(forward_covs, reverse_covs, feature_weights, self.sample_names.len()))

    }


}

#[derive(Serialize,Deserialize,Debug,Clone)]
pub struct RankTable {
    meta_vector: Vec<RankVector>,
    pub feature_names: Vec<String>,
    pub sample_names: Vec<String>,
    feature_dictionary: HashMap<String,usize>,
    sample_dictionary: HashMap<String,usize>,
    draw_order: Vec<usize>,
    index: usize,
    pub dimensions: (usize,usize),
    dropout: DropMode,
}

pub fn mad_minimum(forward:Vec<Vec<f64>>,reverse: Vec<Vec<f64>>, feature_weights: &Vec<f64>, total_samples: usize) -> (usize,f64) {

    let mut dispersions: Vec<f64> = Vec::with_capacity(forward.len());

    for i in 0..forward.len() {
        let mut sample_dispersions = Vec::with_capacity(forward[i].len());

        for j in 0..forward[i].len() {
            let feature_dispersion = (forward[i][j] * ((total_samples - i) as f64 / total_samples as f64)) + (reverse[i][j] * ((i + 1) as f64/ total_samples as f64));

            sample_dispersions.push(feature_dispersion.powi(2) * feature_weights[j])

        }

        dispersions.push(sample_dispersions.iter().sum::<f64>() / feature_weights.iter().sum::<f64>());

    }

    let mut truncated: Vec<(usize,f64)> = dispersions.into_iter().enumerate().collect();
    if truncated.len() > 6 {
        truncated = truncated[3..truncated.len()-3].to_vec();
    }
    else if truncated.len() > 3 {
        truncated = truncated[1..truncated.len()-1].to_vec();
    }
    else if truncated.len() > 1 {
        truncated = truncated[1..].to_vec();
    }

    // println!("{:?}", truncated.iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or(&(0,0.)));

    truncated.into_iter().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or((0,0.))

}


#[cfg(test)]
mod rank_table_tests {

    use super::*;
    use feature_thread_pool::FeatureThreadPool;

    #[test]
    fn rank_table_general_test() {
        let table = RankTable::new(&vec![vec![1.,2.,3.],vec![4.,5.,6.],vec![7.,8.,9.]], &vec!["one".to_string(),"two".to_string(),"three".to_string()], &vec!["0".to_string(),"1".to_string(),"2".to_string()],DropMode::Zeros);
        assert_eq!(table.medians(),vec![2.,5.,8.]);
        assert_eq!(table.dispersions(),vec![1.,1.,1.]);
        assert_eq!(*table.feature_index("one").unwrap(),0);
    }

    #[test]
    fn rank_table_trivial_test() {
        let table = RankTable::new(&Vec::new(), &Vec::new(), &Vec::new(),DropMode::No);
        let empty: Vec<f64> = Vec::new();
        assert_eq!(table.medians(),empty);
        assert_eq!(table.dispersions(),empty);
    }

    #[test]
    pub fn rank_table_simple_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &vec!["one".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..],DropMode::Zeros);
        let draw_order = table.sort_by_feature("one");
        let mad_order = table.meta_vector[*table.feature_index("one").unwrap()].clone().ordered_mad(&draw_order);
        assert_eq!(mad_order, vec![(7.5,8.),(10.,5.),(12.5,5.),(15.,5.),(17.5,2.5),(20.,0.),(0.,0.)]);
    }

    #[test]
    pub fn split() {
        let mut table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &vec!["one".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..],DropMode::Zeros);
        let pool = FeatureThreadPool::new(1);
        let draw_order = table.sort_by_feature("one");
        println!("{:?}", table.parallel_split_order(draw_order, &vec![1.], pool));
    }

    #[test]
    pub fn rank_table_derive_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &vec!["one".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..],DropMode::Zeros);
        let kid1 = table.derive(&vec![0,2,4,6]);
        let kid2 = table.derive(&vec![1,3,5,7]);
    }

    #[test]
    pub fn rank_table_derive_empty_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],vec![0.,1.,0.,1.,0.,1.,0.,1.]], &vec!["one".to_string(),"two".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..],DropMode::Zeros);
        let kid1 = table.derive(&vec![0,2,4,6]);
        let kid2 = table.derive(&vec![1,3,5,7]);
    }


}
