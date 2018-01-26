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
use std::sync::mpsc;

extern crate rand;
use rand::Rng;

use thread_pool::ThreadPool;
use rank_table::RankTable;
use rank_table::RankTableSplitter;


impl Node {

    pub fn root<'a>(counts:&Vec<Vec<f64>>,feature_names:&'a[String],sample_names:&'a[String],input_features: Vec<String>,output_features:Vec<String>,pool:mpsc::Sender<((usize, (RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>) -> Node
    {

        let rank_table = RankTable::new(counts,&feature_names,&sample_names);

        let feature_weights = vec![1.;feature_names.len()];

        let medians = rank_table.medians();

        let dispersions = rank_table.dispersions();

        let new_node = Node {
            pool: pool,

            rank_table: rank_table,
            dropout: true,

            id: "RT".to_string(),
            parent_id: "RT".to_string(),
            children: Vec::new(),

            feature: None,
            split: None,

            output_features: output_features,
            input_features: input_features,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
        };

        new_node

    }

    pub fn between(&self, feature:&str, begin:&str, end:&str) -> usize {
        self.rank_table.between(feature,begin,end)
    }

    pub fn samples(&self) -> Vec<String> {
        self.rank_table.sample_names.clone()
    }

    pub fn split(&self, feature: &str) -> (String,usize,String,usize,f64,f64,Vec<usize>) {

        println!("Splitting a node");

        let feature_index = self.rank_table.feature_index(feature);

        let (forward,reverse,draw_order) = self.rank_table.split(feature);

        let mut fw_dsp = vec![0.;forward.length as usize];

        for (i,sample) in forward.enumerate() {

            println!("{:?}",sample);
            fw_dsp[i] = sample
                .iter()
                .enumerate()
                .fold(0.,|acc,x| {
                    let mut div = (x.1).1/(x.1).0;
                    if div.is_nan() {
                        div = 0.;
                    };
                    div.powi(2) * self.feature_weights[x.0] * ((x.0 != feature_index) as i32 as f64) + acc
                })
                .sqrt();

        }

        let mut rv_dsp = vec![0.;reverse.length as usize];

        // println!("Done with forward, printing reverse");

        for (i,sample) in reverse.enumerate() {

            // println!("{:?}",sample);
            rv_dsp[i] = sample
                .iter()
                .enumerate()
                .fold(0.,|acc,x| {
                    let mut div = (x.1).1/(x.1).0;
                    if div.is_nan() {
                        div = 0.;
                    };
                    div.powi(2) * self.feature_weights[x.0] * ((x.0 != feature_index) as i32 as f64) + acc
                })
                .sqrt();


        }

        rv_dsp.reverse();

        // for combo in fw_dsp.iter().zip(rv_dsp.iter()) {
        //     println!("{:?},{}", combo, combo.0 + combo.1);
        // }

        let (mut split_index, mut split_dispersion) = (0,std::f64::INFINITY);

        for (i,(fw,rv)) in fw_dsp.iter().zip(rv_dsp).enumerate() {
            if fw_dsp.len() > 6 && i > 2 && i < fw_dsp.len() - 3 {
                if fw+rv < split_dispersion {
                    split_index = i;
                    split_dispersion = fw+rv;
                }
            }
            else if fw_dsp.len() > 3 && fw_dsp.len() < 6 && i > 1 && i < fw_dsp.len() -1 {
                if fw+rv < split_dispersion {
                    split_index = i;
                    split_dispersion = fw+rv;
                }
            }
            else if fw_dsp.len() < 3 {
                if fw+rv < split_dispersion {
                    split_index = i;
                    split_dispersion = fw+rv;
                }
            }
        }

        let split_sample_value = self.rank_table.feature_fetch(feature, draw_order[split_index]);

        let split_sample_index = draw_order[split_index];

        let split_sample_name = self.rank_table.sample_name(split_sample_index);

        let output = (String::from(feature),split_index,split_sample_name,split_sample_index,split_sample_value,split_dispersion,draw_order);

        println!("Split output: {:?}",output.clone());

        output

    }




    pub fn parallel_best_split(& mut self) -> (String,f64,f64,Vec<usize>,Vec<usize>) {

        // pool: mpsc::Sender<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>),mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>

        if self.input_features.len() < 1 {
            panic!("Tried to split with no input features");
        };

        let mut feature_receivers: Vec<mpsc::Receiver<(usize,usize,f64,Vec<usize>)>> = Vec::with_capacity(self.input_features.len());

        for feature in &self.input_features {

            let feature_index = self.rank_table.feature_index(feature);
            let splitters = self.rank_table.split(feature);
            let mut feature_weights = self.feature_weights.clone();
            feature_weights[feature_index] = 0.;

            let (tx,rx) = mpsc::channel();

            self.pool.send(((feature_index,splitters,feature_weights),tx)).unwrap();

            feature_receivers.push(rx);

        }

        let mut feature_dispersions: Vec<(usize,usize,f64,Vec<usize>)> = Vec::with_capacity(self.input_features.len());

        for receiver in feature_receivers {
            feature_dispersions.push(receiver.recv().unwrap());
        }

        let mut minimum_dispersion = (0,feature_dispersions[0].clone());

        for (i,feature) in feature_dispersions.iter().enumerate() {
            if i == 0 {
                continue
            }
            else {
                if feature.2 < (minimum_dispersion.1).2 {
                    minimum_dispersion = (i,feature.clone());
                }
            }
        }

        let (feature_index,(split_index, split_sample_index, split_dispersion, split_order)) = minimum_dispersion;

        let best_feature = self.input_features[feature_index].clone();

        let split_value = self.rank_table.feature_fetch(&best_feature,split_sample_index);

        self.feature = Some(best_feature.clone());
        self.split = Some(split_value.clone());

        println!("Best split: {:?}", (best_feature.clone(),split_index, split_value,split_dispersion));

        (best_feature,split_dispersion,split_value,split_order[..split_index].iter().cloned().collect(),split_order[split_index..].iter().cloned().collect())

    }

    // pub fn best_split(&mut self) -> (U,usize,T,usize,f64,f64,Vec<usize>) {
    pub fn best_split(&mut self) -> (String,f64,f64,Vec<usize>,Vec<usize>) {

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

        self.feature = Some(minimum_dispersion.0.clone());
        self.split = Some(minimum_dispersion.4);

        println!("Best split: {:?}", minimum_dispersion.clone());

        (minimum_dispersion.0,minimum_dispersion.5,minimum_dispersion.4,minimum_dispersion.6[..minimum_dispersion.1].iter().cloned().collect(),minimum_dispersion.6[minimum_dispersion.1..].iter().cloned().collect())

    }

    pub fn derive(&self, indecies: &[usize],new_id:&str) -> Node {
            let new_rank_table = self.rank_table.derive(indecies);

            let medians = new_rank_table.medians();
            let dispersions = new_rank_table.dispersions();
            let feature_weights = vec![1.;new_rank_table.dimensions.0];


            let child = Node {
                pool: self.pool.clone(),

                rank_table: new_rank_table,
                dropout: self.dropout,

                parent_id: self.id.clone(),
                id: new_id.to_string(),
                children: Vec::new(),

                feature: None,
                split: None,

                output_features: self.output_features.clone(),
                input_features: self.input_features.clone(),

                medians: medians,
                feature_weights: feature_weights,
                dispersions: dispersions,
            };


            child
        }


    pub fn derive_from_prototype(&self,features:usize, samples: usize, input_features: usize, output_features: usize, new_id:&str, ) -> Node {

        let mut rng = rand::thread_rng();

        let new_rank_table = self.rank_table.derive_from_prototype(features, samples);

        let new_input_features = rand::seq::sample_iter(&mut rng, new_rank_table.feature_names.iter().cloned(), input_features).expect("Couldn't generate input features");
        let new_output_features = rand::seq::sample_iter(&mut rng, new_rank_table.feature_names.iter().cloned(), output_features).expect("Couldn't generate output features");

        let medians = new_rank_table.medians();
        let dispersions = new_rank_table.dispersions();
        let feature_weights = vec![1.;new_rank_table.dimensions.0];

        let child = Node {
            pool: self.pool.clone(),

            rank_table: new_rank_table,
            dropout: self.dropout,

            parent_id: self.id.clone(),
            id: new_id.to_string(),
            children: Vec::new(),

            feature: None,
            split: None,

            output_features: new_output_features,
            input_features: new_input_features,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
        };


        child
    }

    pub fn parallel_derive(&mut self) {
        let (feature, dispersion,split_value, left_indecies ,right_indecies) = self.parallel_best_split();

        let mut left_child_id = self.id.clone();
        let mut right_child_id = self.id.clone();
        left_child_id.push_str(&format!(":F{}S{}L",feature,split_value));
        right_child_id.push_str(&format!(":F{}S{}R",feature,split_value));

        let left_child = self.derive(&left_indecies,&left_child_id);
        let right_child = self.derive(&right_indecies,&right_child_id);
        println!("{:?}",left_child.samples());
        println!("{:?}", right_child.samples());

        self.report(false);
        left_child.report(false);
        right_child.report(false);

        self.children.push(left_child);
        self.children.push(right_child);
    }

    pub fn derive_children(&mut self) {
            let (feature,dispersion,split_value, left_indecies,right_indecies) = self.best_split();

            let mut left_child_id = self.id.clone();
            let mut right_child_id = self.id.clone();
            left_child_id.push_str(&format!(":F{}S{}L",feature,split_value));
            right_child_id.push_str(&format!(":F{}S{}R",feature,split_value));

            let left_child = self.derive(&left_indecies, &left_child_id);
            let right_child = self.derive(&right_indecies, &right_child_id);
            println!("{:?}",left_child.samples());
            println!("{:?}", right_child.samples());

            self.report(true);
            left_child.report(true);
            right_child.report(true);

            self.children.push(left_child);
            self.children.push(right_child);
    }

    pub fn report(&self,verbose:bool) {
        println!("Node reporting:");
        println!("Feature:{:?}",self.feature);
        println!("Split:{:?}", self.split);
        println!("Output features:{}",self.output_features.len());
        if verbose {
            println!("{:?}",self.output_features);
            println!("{:?}",self.medians);
            println!("{:?}",self.dispersions);
            println!("{:?}",self.feature_weights);
        }
        println!("Samples: {}", self.rank_table.sample_names.len());
        if verbose {
            println!("{:?}", self.rank_table.samples());
            println!("Counts: {:?}", self.rank_table.full_ordered_values());
            println!("Ordered counts: {:?}", self.rank_table.full_values());
        }

    }

    pub fn data_dump(&self) -> String {
        let mut report_string = String::new();
        report_string.push_str(&format!("!ID:{}\n",self.id));
        report_string.push_str(&format!("Children:"));
        for child in &self.children {
            report_string.push_str(&format!("!C:{}",child.id));
        }
        report_string.push_str("\n");
        report_string.push_str(&format!("ParentID:{}\n",self.parent_id));
        report_string.push_str(&format!("Feature:{:?}\n", self.feature));
        report_string.push_str(&format!("Split:{:?}\n",self.split));
        report_string.push_str(&format!("Output features:{:?}\n",self.output_features.len()));
        report_string.push_str(&format!("{:?}\n",self.output_features));
        report_string.push_str(&format!("Medians:{:?}\n",self.medians));
        report_string.push_str(&format!("Dispersions:{:?}\n",self.dispersions));
        report_string.push_str(&format!("Feature weights:{:?}\n",self.feature_weights));
        report_string.push_str(&format!("Samples:{:?}\n",self.internal_report().len()));
        report_string.push_str(&format!("{:?}\n",self.internal_report()));
        report_string.push_str(&format!("Full:{:?}\n",self.rank_table.full_ordered_values()));
        report_string
    }



    pub fn internal_report(&self) -> &[String] {
        self.rank_table.samples()
    }

    pub fn set_weights(&mut self, weights:Vec<f64>) {
        self.feature_weights = weights;
    }

}

#[derive(Clone)]
pub struct Node {

    pool: mpsc::Sender<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>),mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>,

    pub rank_table: RankTable,
    dropout: bool,

    pub parent_id: String,
    pub id: String,
    pub children: Vec<Node>,

    feature: Option<String>,
    split: Option<f64>,

    pub output_features: Vec<String>,
    pub input_features: Vec<String>,

    medians: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub dispersions: Vec<f64>,
}


// pub fn generate_weak<T>(target:T) -> (T,Weak<T>) {
//     let arc_t = Arc::new(target);
//     let weak_t = Arc::downgrade(&arc_t);
//     match Arc::try_unwrap(arc_t) {
//         Ok(object) => return(object,weak_t),
//         Err(err) => panic!("Tried to unwrap an empty reference, something went wrong with weak reference construction!")
//     }
// }
