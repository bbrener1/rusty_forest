
use std::sync::Arc;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::sync::mpsc;
use serde_json;


extern crate rand;
use rank_table::RankTable;
use rank_vector::RankVector;
use DropMode;

impl Node {


    pub fn feature_root<'a>(input_counts:&Vec<Vec<f64>>,output_counts:&Vec<Vec<f64>>,input_feature_names:&'a[String],output_feature_names:&'a[String],sample_names:&'a[String],dropout:DropMode,feature_pool: mpsc::Sender<(((RankVector,Arc<Vec<usize>>),mpsc::Sender<(Vec<(f64,f64)>,RankVector)>))>) -> Node {

        let input_table = RankTable::new(input_counts,&input_feature_names,&sample_names,dropout);

        let output_table = RankTable::new(output_counts,&output_feature_names,&sample_names,dropout);

        let feature_weights = vec![1.;output_feature_names.len()];

        let medians = output_table.medians();

        let dispersions = output_table.dispersions();

        let local_gains = vec![0.;dispersions.len()];

        let new_node = Node {
            feature_pool: feature_pool,

            input_table: input_table,
            output_table: output_table,
            dropout: dropout,

            id: "RT".to_string(),
            parent_id: "RT".to_string(),
            children: Vec::new(),

            feature: None,
            split: None,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
            local_gains: Some(local_gains),
            absolute_gains: None
        };

        new_node
    }

    pub fn feature_parallel_best_split(&mut self) -> (String,f64,f64,Vec<usize>,Vec<usize>) {

        if self.input_features().len() < 1 {
            panic!("Tried to split with no input features");
        };

        let mut minima = Vec::new();

        let draw_orders: Vec<Vec<usize>> = self.input_features().iter().map(|x| self.input_table.sort_by_feature(x)).collect();

        for draw_order in draw_orders {
            if let Some(minimum) = self.output_table.parallel_split_order(draw_order,&self.feature_weights,self.feature_pool.clone()) {
                // println!("Minimum: {:?}",minimum);
                minima.push(minimum);
            }
        };

        let (best_input_feature_index,(split_index,split_dispersion)) = minima.into_iter().enumerate().min_by(|a,b| (a.1).1.partial_cmp(&(b.1).1).unwrap_or(Ordering::Greater)).unwrap();

        let best_feature = self.input_features()[best_input_feature_index].clone();

        let split_order = self.input_table.sort_by_feature(&best_feature);

        let split_sample_index = split_order[split_index];

        let split_value = self.input_table.feature_fetch(&best_feature,split_sample_index);

        self.feature = Some(best_feature.clone());
        self.split = Some(split_value.clone());

        println!("Best split: {:?}", (best_feature.clone(),split_index, split_value,split_dispersion));

        (best_feature.clone(),split_dispersion,split_value,split_order[..split_index].iter().cloned().collect(),split_order[split_index..].iter().cloned().collect())

    }



    pub fn feature_parallel_derive(&mut self) {

        let (feature,_dispersion,split_value, left_indecies,right_indecies) = self.feature_parallel_best_split();

        let mut left_child_id = self.id.clone();
        let mut right_child_id = self.id.clone();
        left_child_id.push_str(&format!("!F{}:S{}L",feature,split_value));
        right_child_id.push_str(&format!("!F{}:S{}R",feature,split_value));

        println!("{:?}",&left_indecies);
        println!("{:?}",&right_indecies);

        let left_child = self.derive(&left_indecies, &left_child_id);
        let right_child = self.derive(&right_indecies, &right_child_id);
        // println!("{:?}",left_child.samples());
        // println!("{:?}", right_child.samples());

        self.report(false);
        left_child.report(false);
        right_child.report(false);

        self.children.push(left_child);
        self.children.push(right_child);

        println!("Derived children!");

    }

    pub fn derive(&self, indecies: &[usize],new_id:&str) -> Node {

            let new_input_table = self.input_table.derive(indecies);
            let new_output_table = self.output_table.derive(indecies);

            let medians = new_output_table.medians();
            let dispersions = new_output_table.dispersions();
            let feature_weights = vec![1.;new_output_table.dimensions.0];

            let mut local_gains = Vec::with_capacity(dispersions.len());

            for ((nd,nm),(od,om)) in dispersions.iter().zip(medians.iter()).zip(self.dispersions.iter().zip(self.medians.iter())) {
                let mut old_cov = od/om;
                if !old_cov.is_normal() {
                    old_cov = 0.;
                }
                let mut new_cov = nd/nm;
                if !new_cov.is_normal() {
                    new_cov = 0.;
                }
                local_gains.push(old_cov-new_cov)
                // local_gains.push((od/om)/(nd/nm));
            }

            let child = Node {
                // pool: self.pool.clone(),
                feature_pool: self.feature_pool.clone(),

                input_table: new_input_table,
                output_table: new_output_table,
                dropout: self.dropout,

                parent_id: self.id.clone(),
                id: new_id.to_string(),
                children: Vec::new(),

                feature: None,
                split: None,

                medians: medians,
                feature_weights: feature_weights,
                dispersions: dispersions,
                local_gains: Some(local_gains),
                absolute_gains: None
            };


            child
        }

    pub fn derive_specified(&self, samples: &Vec<&String>, input_features: &Vec<&String>, output_features: &Vec<&String>, new_id: &str) -> Node {

        let mut new_input_table = self.input_table.derive_specified(&input_features,samples);
        let mut new_output_table = self.output_table.derive_specified(&output_features,samples);

        let medians = new_output_table.medians();
        let dispersions = new_output_table.dispersions();
        let feature_weights = vec![1.;new_output_table.dimensions.0];

        let child = Node {
            // pool: self.pool.clone(),
            feature_pool: self.feature_pool.clone(),

            input_table: new_input_table,
            output_table: new_output_table,
            dropout: self.dropout,

            parent_id: self.id.clone(),
            id: new_id.to_string(),
            children: Vec::new(),

            feature: None,
            split: None,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
            local_gains: None,
            absolute_gains: None
        };


        child
    }

    pub fn derive_random(&self, samples: usize, input_features: usize, output_features: usize, new_id:&str, ) -> Node {

        let mut rng = rand::thread_rng();

        let shuffle = rand::seq::sample_indices(&mut rng, input_features.max(output_features), input_features.max(output_features));

        let new_input_features = &shuffle[..input_features].iter().map(|x| &self.input_features()[*x]).collect();

        let new_output_features = &shuffle[(shuffle.len() - output_features)..].iter().map(|x| &self.input_features()[*x]).collect();

        // println!("Deriving a node:");
        // println!("Input features: {:?}", new_input_features);
        // println!("Output features: {:?}", new_output_features);

        let new_samples = rand::seq::sample_iter(&mut rng, self.samples().iter(), samples).expect("Couldn't generate sample subsample");

        self.derive_specified(&new_samples,new_input_features,new_output_features,new_id)
    }

    pub fn derive_known_split(&self,feature:&str,split:&f64) -> (Node,Node){
        let (left_indecies,right_indecies) = self.input_table.split_indecies_by_feature(feature,split);

        let mut left_child_id = self.id.clone();
        let mut right_child_id = self.id.clone();
        left_child_id.push_str(&format!("!F{}:S{}L",feature,split));
        right_child_id.push_str(&format!("!F{}:S{}R",feature,split));

        let left_child = self.derive(&left_indecies, &left_child_id);
        let right_child = self.derive(&right_indecies, &right_child_id);

        (left_child,right_child)

    }


    pub fn report(&self,verbose:bool) {
        println!("Node reporting:");
        println!("Feature:{:?}",self.feature);
        println!("Split:{:?}", self.split);
        println!("Output features:{}",self.output_features().len());
        if verbose {
            println!("{:?}",self.output_features());
            println!("{:?}",self.medians);
            println!("{:?}",self.dispersions);
            println!("{:?}",self.feature_weights);
        }
        println!("Samples: {}", self.samples().len());
        if verbose {
            println!("{:?}", self.samples());
            println!("Counts: {:?}", self.output_table.full_ordered_values());
            println!("Ordered counts: {:?}", self.output_table.full_values());
        }

    }


    pub fn summary(&self) -> String {
        let mut report_string = "".to_string();
        if self.children.len() > 1 {
            report_string.push_str(&format!("!ID:{}\n",self.id));
            report_string.push_str(&format!("F:{}\n",self.feature.clone().unwrap_or("".to_string())));
            report_string.push_str(&format!("S:{}\n",self.split.unwrap_or(0.)));
        }

        report_string
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
        report_string.push_str(&format!("Output features:{:?}\n",self.output_features().len()));
        report_string.push_str(&format!("{:?}\n",self.output_features()));
        report_string.push_str(&format!("Medians:{:?}\n",self.medians));
        report_string.push_str(&format!("Dispersions:{:?}\n",self.dispersions));
        report_string.push_str(&format!("Local gains:{:?}\n",self.local_gains));
        report_string.push_str(&format!("Absolute gains:{:?}\n",self.absolute_gains));
        report_string.push_str(&format!("Feature weights:{:?}\n",self.feature_weights));
        report_string.push_str(&format!("Samples:{:?}\n",self.samples().len()));
        report_string.push_str(&format!("{:?}\n",self.samples()));
        report_string.push_str(&format!("Full:{:?}\n",self.output_table.full_ordered_values()));
        report_string
    }


    pub fn set_weights(&mut self, weights:Vec<f64>) {
        self.feature_weights = weights;
    }

    pub fn wrap_consume(self) -> NodeWrapper {

        // let mut children: Vec<String> = Vec::with_capacity(self.children.len());
        let mut children: Vec<NodeWrapper> = Vec::with_capacity(self.children.len());

        for child in self.children {
            // children.push(child.wrap_consume().to_string())
            children.push(child.wrap_consume())
        }

        NodeWrapper {
            input_table: self.input_table,
            output_table: self.output_table,
            dropout: self.dropout,

            parent_id: self.parent_id,
            id: self.id,
            children: children,

            feature: self.feature,
            split: self.split,

            medians: self.medians,
            feature_weights: self.feature_weights,
            dispersions: self.dispersions,
            local_gains: self.local_gains,
            absolute_gains: self.absolute_gains

        }

    }

    pub fn strip_consume(self) -> StrippedNode {

        let features = self.output_features().clone();
        let samples = self.samples().clone();

        let mut stripped_children = Vec::new();

        for child in self.children.into_iter() {
            stripped_children.push(child.strip_consume())
        }


        StrippedNode {
            dropout: self.dropout,

            children: stripped_children,

            feature: self.feature,
            split: self.split,

            features: features,
            samples: samples,

            medians: self.medians,
            dispersions: self.dispersions,

            local_gains: self.local_gains,
            absolute_gains: self.absolute_gains,
        }
    }

    pub fn strip_clone(&self) -> StrippedNode {

        let mut stripped_children = Vec::new();

        for child in &self.children {
            stripped_children.push(child.strip_clone())
        }

        StrippedNode {
            dropout: self.dropout,

            children: stripped_children,

            feature: self.feature.clone(),
            split: self.split.clone(),

            features: self.output_features().clone(),
            samples: self.samples().clone(),

            medians: self.medians.clone(),
            dispersions: self.dispersions.clone(),

            local_gains: self.local_gains.clone(),
            absolute_gains: self.absolute_gains.clone(),
        }

    }

    pub fn output_rank_table(&self) -> &RankTable {
        &self.output_table
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn samples(&self) -> &Vec<String> {
        &self.output_table.sample_names
    }

    pub fn input_features(&self) -> &Vec<String> {
        &self.input_table.features()
    }

    pub fn output_features(&self) -> &Vec<String> {
        &self.output_table.features()
    }

    pub fn feature(&self) -> &Option<String> {
        &self.feature
    }

    pub fn split(&self) -> &Option<f64> {
        &self.split
    }

    pub fn medians(&self) -> &Vec<f64> {
        &self.medians
    }

    pub fn dispersions(&self) -> &Vec<f64> {
        &self.dispersions
    }

    pub fn mads(&self) -> &Vec<f64> {
        &self.dispersions
    }

    pub fn dimensions(&self) -> (usize,usize) {
        self.output_table.dimensions
    }

    pub fn dropout(&self) -> DropMode {
        self.dropout
    }

    pub fn absolute_gains(&self) -> &Option<Vec<f64>> {
        &self.absolute_gains
    }

    pub fn local_gains(&self) -> &Option<Vec<f64>> {
        &self.local_gains
    }

    pub fn wrap_clone(&self) -> NodeWrapper {
        self.clone().wrap_consume()
    }

    pub fn crawl_children(&self) -> Vec<&Node> {
        let mut output = Vec::new();
        for child in &self.children {
            output.extend(child.crawl_children());
        }
        output.push(&self);
        output
    }

    pub fn compute_absolute_gains(&mut self, root_medians: &Vec<f64>,root_dispersions: &Vec<f64>) {

        let mut absolute_gains = Vec::with_capacity(root_dispersions.len());

        for ((nd,nm),(od,om)) in self.dispersions.iter().zip(self.medians.iter()).zip(root_dispersions.iter().zip(root_medians.iter())) {
            let mut old_cov = od/om;
            if !old_cov.is_normal() {
                old_cov = 0.;
            }
            let mut new_cov = nd/nm;
            if !new_cov.is_normal() {
                new_cov = 0.;
            }
            absolute_gains.push(old_cov-new_cov)

        }

        self.absolute_gains = Some(absolute_gains);

        for child in self.children.iter_mut() {
            child.compute_absolute_gains(root_medians,root_dispersions);
        }
    }

    pub fn cascading_interaction<'a>(&'a self,mut parents:Vec<(&'a Node,&'a str)>) -> Vec<(&'a str, &'a str, f64, &'a str, &'a str,f64,&'a str, f64)> {

        let mut interactions: Vec<(&'a str, &'a str, f64, &'a str, &'a str,f64,&'a str, f64)> = Vec::new();

        if let (&Some(ref feature),&Some(ref split)) = (&self.feature,&self.split) {

            for &(parent,inequality) in parents.iter(){
                let (left,right) = parent.derive_known_split(&feature,&split);

                let interaction_gain: Vec<f64> =
                self.children[0]
                    .local_gains()
                        .as_ref()
                            .unwrap()
                                .iter()
                    .zip(
                        left
                            .local_gains()
                                .as_ref()
                                    .unwrap()
                                        .iter()
                        )
                    .map(|x| *x.0 - *x.1)
                        .collect();

                for (interaction,int_feature) in interaction_gain.iter().zip(self.output_features()) {
                    interactions.push((feature,"<",*split,parent.feature().as_ref().unwrap(),inequality,parent.split().clone().unwrap(),int_feature,*interaction));
                }

                let interaction_gain: Vec<f64> =
                self.children[1]
                    .local_gains()
                        .as_ref()
                            .unwrap()
                                .iter()
                    .zip(
                        right
                            .local_gains()
                                .as_ref()
                                    .unwrap()
                                        .iter()
                        )
                    .map(|x| *x.0 - *x.1)
                        .collect();

                for (interaction,int_feature) in interaction_gain.iter().zip(self.output_features()) {
                    interactions.push((feature,">",*split,parent.feature().as_ref().unwrap(),inequality,parent.split().clone().unwrap(),int_feature,*interaction));
                }


            }

            let mut next = parents.clone();
            next.push((&self,"<"));
            interactions.extend(self.children[0].cascading_interaction(next));

            let mut next = parents.clone();
            next.push((&self,">"));
            interactions.extend(self.children[1].cascading_interaction(next));

        }

        interactions

    }

    pub fn translate_interactions(&self) -> String {
        let interactions = self.cascading_interaction(vec![]);
        let mut report = String::new();
        for line in interactions {
            report.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", line.0,line.1,line.2,line.3,line.4,line.5,line.6,line.7));
        }
        report
    }

    pub fn root_absolute_gains(&mut self) {
        for child in self.children.iter_mut() {
            child.compute_absolute_gains(&self.medians,&self.dispersions);
        }
    }

    pub fn crawl_leaves(&self) -> Vec<&Node> {
        let mut output = Vec::new();
        if self.children.len() < 1 {
            return vec![&self]
        }
        else {
            for child in &self.children {
                output.extend(child.crawl_leaves());
            }
        };
        output
    }

}

#[derive(Clone)]
pub struct Node {

    // pool: mpsc::Sender<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>),mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>,
    feature_pool: mpsc::Sender<(((RankVector,Arc<Vec<usize>>),mpsc::Sender<(Vec<(f64,f64)>,RankVector)>))>,

    input_table: RankTable,
    output_table: RankTable,
    dropout: DropMode,

    pub parent_id: String,
    pub id: String,
    pub children: Vec<Node>,

    feature: Option<String>,
    split: Option<f64>,

    pub medians: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub dispersions: Vec<f64>,
    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>
}

impl NodeWrapper {
    pub fn to_string(self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    pub fn unwrap(self,feature_pool: mpsc::Sender<(((RankVector,Arc<Vec<usize>>),mpsc::Sender<(Vec<(f64,f64)>,RankVector)>))>) -> Node {
        let mut children: Vec<Node> = Vec::with_capacity(self.children.len());
        for child in self.children {
            // println!("#######################################\n");
            // println!("#######################################\n");
            // println!("#######################################\n");
            // println!("Unwrapping child:");
            // println!("{}", child);
            // children.push(serde_json::from_str::<NodeWrapper>(&child).unwrap().unwrap(feature_pool.clone()));
            // println!("Unwrapped child");
            children.push(child.unwrap(feature_pool.clone()));
        }

        println!("Recursive unwrap finished!");

        Node {

            feature_pool: feature_pool,

            input_table: self.input_table,
            output_table: self.output_table,
            dropout: self.dropout,

            parent_id: self.parent_id,
            id: self.id,
            children: children,

            feature: self.feature,
            split: self.split,

            medians: self.medians,
            feature_weights: self.feature_weights,
            dispersions: self.dispersions,
            local_gains: self.local_gains,
            absolute_gains: self.absolute_gains
        }

    }

}

#[derive(Serialize,Deserialize)]
pub struct NodeWrapper {

    pub input_table: RankTable,
    pub output_table: RankTable,
    pub dropout: DropMode,

    pub parent_id: String,
    pub id: String,
    // pub children: Vec<String>,
    pub children: Vec<NodeWrapper>,

    pub feature: Option<String>,
    pub split: Option<f64>,

    pub medians: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub dispersions: Vec<f64>,
    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>

}


pub fn mad_minimum(forward:Vec<Vec<f64>>,reverse: Vec<Vec<f64>>,feature_weights: &Vec<f64>) -> (usize,f64) {

    let x = forward.len();
    let y = forward.get(0).unwrap_or(&vec![]).len();

    // feature_weights[exclusion] = 0.;

    let mut dispersions: Vec<f64> = Vec::with_capacity(x);

    // println!("Dispersion split:");

    for i in 0..x {
        let mut sample_dispersions = Vec::with_capacity(y);
        for j in 0..y {
            let feature_dispersion = (forward[i][j] * ((x - i) as f64 / x as f64)) + (reverse[i][j] * ((i + 1) as f64/ x as f64));
            sample_dispersions.push((feature_dispersion * feature_weights[j]).powi(2))

            ////
            //// IMPORTANT: Squaring the dispersions before summing allows us to focus on features
            //// that improve a few other features a lot instead of many features a little
            ////

        }
        // print!("F:{:?} :" , forward[i]);
        // println!("{:?}", forward[i].iter().sum::<f64>());
        // print!("R:{:?} :", reverse[i]);
        // println!("{:?}", reverse[i].iter().sum::<f64>());
        // print!("{:?} ", sample_dispersions);
        // println!("{:?}", sample_dispersions.iter().sum::<f64>());
        dispersions.push(sample_dispersions.iter().sum());
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

#[derive(Serialize,Deserialize)]
pub struct StrippedNode {

    dropout: DropMode,

    pub children: Vec<StrippedNode>,

    feature: Option<String>,
    split: Option<f64>,

    features: Vec<String>,
    samples: Vec<String>,
    medians: Vec<f64>,
    dispersions: Vec<f64>,


    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>,
}

impl StrippedNode {

    pub fn to_string(self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    pub fn feature(&self) -> &Option<String> {
        &self.feature
    }

    pub fn features(&self) -> &Vec<String> {
        &self.features
    }

    pub fn samples(&self) -> &Vec<String> {
        &self.samples
    }

    pub fn split(&self) -> &Option<f64> {
        &self.split
    }

    pub fn medians(&self) -> &Vec<f64> {
        &self.medians
    }

    pub fn mads(&self) -> &Vec<f64> {
        &self.dispersions
    }

    pub fn absolute_gains(&self) -> &Option<Vec<f64>> {
        &self.absolute_gains
    }

    pub fn local_gains(&self) -> &Option<Vec<f64>> {
        &self.local_gains
    }

    pub fn dropout(&self) -> DropMode {
        self.dropout
    }

    pub fn crawl_leaves(&self) -> Vec<&StrippedNode> {
        let mut output = Vec::new();
        if self.children.len() < 1 {
            return vec![&self]
        }
        else {
            for child in &self.children {
                output.extend(child.crawl_leaves());
            }
        };
        output
    }

    pub fn crawl_children(&self) -> Vec<&StrippedNode> {
        let mut output = Vec::new();
        for child in &self.children {
            output.extend(child.crawl_children());
        }
        output.push(&self);
        output
    }

}

#[cfg(test)]
mod node_testing {

    use super::*;
    use feature_thread_pool::FeatureThreadPool;

    #[test]
    fn node_test_trivial_trivial() {
        let mut root = Node::feature_root(&vec![], &vec![], &vec![][..], &vec![][..], &vec![][..], DropMode::Zeros, FeatureThreadPool::new(1));
        root.mads();
        root.medians();
    }

    #[test]
    fn node_test_trivial() {
        let mut root = Node::feature_root(&vec![vec![]],&vec![vec![]], &vec!["one".to_string()][..], &vec!["a".to_string()][..], &vec!["1".to_string()][..], DropMode::Zeros, FeatureThreadPool::new(1));
        root.mads();
        root.medians();
    }

    #[test]
    fn node_test_simple() {
        let mut root = Node::feature_root(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &vec!["one".to_string()],&vec!["one".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..], DropMode::Zeros, FeatureThreadPool::new(1));

        root.feature_parallel_derive();

        assert_eq!(root.children[0].samples(),&vec!["1".to_string(),"4".to_string(),"5".to_string()]);
        assert_eq!(root.children[1].samples(),&vec!["0".to_string(),"3".to_string(),"6".to_string(),"7".to_string()]);
    }

}



// pub fn feature_parallel_given_orders(&mut self,orders:Vec<Vec<usize>>) -> (usize,(usize,f64)) {
//
//     let mut minima = Vec::new();
//
//     for draw_order in orders.into_iter(){
//
//         let forward_draw = Arc::new(draw_order);
//         let mut reverse_draw: Arc<Vec<usize>> = Arc::new(forward_draw.iter().cloned().rev().collect());
//
//         if forward_draw.len() < 2 {
//             continue
//         }
//
//         let mut forward_covs: Vec<Vec<f64>> = vec![vec![0.;self.output_table.dimensions.0];forward_draw.len()];
//         let mut reverse_covs: Vec<Vec<f64>> = vec![vec![0.;self.output_table.dimensions.0];reverse_draw.len()];
//
//         let mut forward_receivers = Vec::with_capacity(self.output_table.dimensions.0);
//         let mut reverse_receivers = Vec::with_capacity(self.output_table.dimensions.0);
//
//         let mut features = self.output_table.drain_features();
//
//         for feature in features.drain(..) {
//             let (tx,rx) = mpsc::channel();
//             self.feature_pool.send(((feature,forward_draw.clone()),tx));
//             forward_receivers.push(rx);
//         }
//
//         for (i,fr) in forward_receivers.iter().enumerate() {
//             if let Ok((disp,feature)) = fr.recv() {
//                 for (j,(m,d)) in disp.into_iter().enumerate() {
//                     forward_covs[j][i] = (d/m).abs();
//                     if forward_covs[j][i].is_nan(){
//                         forward_covs[j][i] = 0.;
//                     }
//                 }
//                 features.push(feature);
//             }
//             else {
//                 panic!("Parellelization error!")
//             }
//
//         }
//
//         for feature in features.drain(..) {
//             let (tx,rx) = mpsc::channel();
//             self.feature_pool.send(((feature,reverse_draw.clone()),tx));
//             reverse_receivers.push(rx);
//         }
//
//
//         for (i,rr) in reverse_receivers.iter().enumerate() {
//             if let Ok((disp,feature)) = rr.recv() {
//                 for (j,(m,d)) in disp.into_iter().enumerate() {
//                     reverse_covs[reverse_draw.len() - j - 1][i] = (d/m).abs();
//                     if reverse_covs[reverse_draw.len() - j - 1][i].is_nan(){
//                         reverse_covs[reverse_draw.len() - j - 1][i] = 0.;
//                     }
//                 }
//                 features.push(feature);
//             }
//             else {
//                 panic!("Parellelization error!")
//             }
//
//         }
//
//         self.output_table.return_features(features);
//
//         minima.push(mad_minimum(forward_covs, reverse_covs, &self.feature_weights.clone()));
//
//     }
//
//     minima.into_iter().enumerate().min_by(|a,b| (a.1).1.partial_cmp(&(b.1).1).unwrap_or(Ordering::Greater)).unwrap()
//
// }

//// MONOTHREADED METHODS:

// pub fn root<'a>(counts:&Vec<Vec<f64>>,feature_names:&'a[String],sample_names:&'a[String],input_features: Vec<String>,output_features:Vec<String>,pool:mpsc::Sender<((usize, (RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>) -> Node
// {
//
//     let rank_table = RankTable::new(counts,&feature_names,&sample_names);
//
//     let feature_weights = vec![1.;feature_names.len()];
//
//     let medians = rank_table.medians();
//
//     let dispersions = rank_table.dispersions();
//
//     let new_node = Node {
//         pool: pool,
//
//         rank_table: rank_table,
//         dropout: true,
//
//         id: "RT".to_string(),
//         parent_id: "RT".to_string(),
//         children: Vec::new(),
//
//         feature: None,
//         split: None,
//
//         output_features: output_features,
//         input_features: input_features,
//
//         medians: medians,
//         feature_weights: feature_weights,
//         dispersions: dispersions,
//     };
//
//     new_node
//
// }


// pub fn split(&self, feature: &str) -> (String,usize,String,usize,f64,f64,Vec<usize>) {
//
//     println!("Splitting a node");
//
//     let feature_index = self.rank_table.feature_index(feature);
//
//     let (forward,reverse,draw_order) = self.rank_table.split(feature);
//
//     let mut fw_dsp = vec![0.;forward.length as usize];
//
//     for (i,sample) in forward.enumerate() {
//
//         println!("{:?}",sample);
//         fw_dsp[i] = sample
//             .iter()
//             .enumerate()
//             .fold(0.,|acc,x| {
//                 let mut div = (x.1).1/(x.1).0;
//                 if div.is_nan() {
//                     div = 0.;
//                 };
//                 div.powi(2) * self.feature_weights[x.0] * ((x.0 != feature_index) as i32 as f64) + acc
//             })
//             .sqrt();
//
//     }
//
//     let mut rv_dsp = vec![0.;reverse.length as usize];
//
//     // println!("Done with forward, printing reverse");
//
//     for (i,sample) in reverse.enumerate() {
//
//         // println!("{:?}",sample);
//         rv_dsp[i] = sample
//             .iter()
//             .enumerate()
//             .fold(0.,|acc,x| {
//                 let mut div = (x.1).1/(x.1).0;
//                 if div.is_nan() {
//                     div = 0.;
//                 };
//                 div.powi(2) * self.feature_weights[x.0] * ((x.0 != feature_index) as i32 as f64) + acc
//             })
//             .sqrt();
//
//
//     }
//
//     rv_dsp.reverse();
//
//     // for combo in fw_dsp.iter().zip(rv_dsp.iter()) {
//     //     println!("{:?},{}", combo, combo.0 + combo.1);
//     // }
//
//     let (mut split_index, mut split_dispersion) = (0,std::f64::INFINITY);
//
//     for (i,(fw,rv)) in fw_dsp.iter().zip(rv_dsp).enumerate() {
//         if fw_dsp.len() > 6 && i > 2 && i < fw_dsp.len() - 3 {
//             if fw+rv < split_dispersion {
//                 split_index = i;
//                 split_dispersion = fw+rv;
//             }
//         }
//         else if fw_dsp.len() > 3 && fw_dsp.len() < 6 && i > 1 && i < fw_dsp.len() -1 {
//             if fw+rv < split_dispersion {
//                 split_index = i;
//                 split_dispersion = fw+rv;
//             }
//         }
//         else if fw_dsp.len() < 3 {
//             if fw+rv < split_dispersion {
//                 split_index = i;
//                 split_dispersion = fw+rv;
//             }
//         }
//     }
//
//     let split_sample_value = self.rank_table.feature_fetch(feature, draw_order[split_index]);
//
//     let split_sample_index = draw_order[split_index];
//
//     let split_sample_name = self.rank_table.sample_name(split_sample_index);
//
//     let output = (String::from(feature),split_index,split_sample_name,split_sample_index,split_sample_value,split_dispersion,draw_order);
//
//     println!("Split output: {:?}",output.clone());
//
//     output
//
// }






// // pub fn best_split(&mut self) -> (U,usize,T,usize,f64,f64,Vec<usize>) {
// pub fn best_split(&mut self) -> (String,f64,f64,Vec<usize>,Vec<usize>) {
//
//     if self.input_features.len() < 1 {
//         panic!("Tried to split with no input features");
//     };
//
//     let first_feature = self.input_features.first().unwrap().clone();
//
//     let mut minimum_dispersion = self.split(&first_feature);
//
//     for feature in self.input_features.clone().iter().enumerate() {
//         if feature.0 == 0 {
//             continue
//         }
//         else {
//             let current_dispersion = self.split(&feature.1);
//             if current_dispersion.5 < minimum_dispersion.5 {
//                 minimum_dispersion = current_dispersion;
//             }
//         }
//
//     }
//
//     self.feature = Some(minimum_dispersion.0.clone());
//     self.split = Some(minimum_dispersion.4);
//
//     println!("Best split: {:?}", minimum_dispersion.clone());
//
//     (minimum_dispersion.0,minimum_dispersion.5,minimum_dispersion.4,minimum_dispersion.6[..minimum_dispersion.1].iter().cloned().collect(),minimum_dispersion.6[minimum_dispersion.1..].iter().cloned().collect())
//
// }
//
// pub fn derive(&self, indecies: &[usize],new_id:&str) -> Node {
//
//         let new_rank_table = self.rank_table.derive(indecies);
//
//         let medians = new_rank_table.medians();
//         let dispersions = new_rank_table.dispersions();
//         let feature_weights = vec![1.;new_rank_table.dimensions.0];
//
//         let mut local_gains = Vec::with_capacity(dispersions.len());
//
//         for ((nd,nm),(od,om)) in dispersions.iter().zip(medians.iter()).zip(self.dispersions.iter().zip(self.medians.iter())) {
//             let mut old_cov = od/om;
//             if !old_cov.is_normal() {
//                 old_cov = 0.;
//             }
//             let mut new_cov = nd/nm;
//             if !new_cov.is_normal() {
//                 new_cov = 0.;
//             }
//             local_gains.push(old_cov-new_cov)
//             // local_gains.push((od/om)/(nd/nm));
//         }
//
//         let child = Node {
//             // pool: self.pool.clone(),
//             feature_pool: self.feature_pool.clone(),
//
//             rank_table: new_rank_table,
//             dropout: self.dropout,
//
//             parent_id: self.id.clone(),
//             id: new_id.to_string(),
//             children: Vec::new(),
//
//             feature: None,
//             split: None,
//
//             output_features: self.output_features.clone(),
//             input_features: self.input_features.clone(),
//
//             medians: medians,
//             feature_weights: feature_weights,
//             dispersions: dispersions,
//             local_gains: Some(local_gains),
//             absolute_gains: None
//         };
//
//
//         child
//     }
//
//
// pub fn derive_from_prototype(&self,features:usize, samples: usize, input_features: usize, output_features: usize, new_id:&str, ) -> Node {
//
//     let mut rng = rand::thread_rng();
//
//     let new_rank_table = self.rank_table.derive_from_prototype(features, samples);
//
//     let new_input_features = rand::seq::sample_iter(&mut rng, new_rank_table.feature_names.iter().cloned(), input_features).expect("Couldn't generate input features");
//     let new_output_features = rand::seq::sample_iter(&mut rng, new_rank_table.feature_names.iter().cloned(), output_features).expect("Couldn't generate output features");
//
//     let medians = new_rank_table.medians();
//     let dispersions = new_rank_table.dispersions();
//     let feature_weights = vec![1.;new_rank_table.dimensions.0];
//
//     let child = Node {
//         // pool: self.pool.clone(),
//         feature_pool: self.feature_pool.clone(),
//
//         rank_table: new_rank_table,
//         dropout: self.dropout,
//
//         parent_id: self.id.clone(),
//         id: new_id.to_string(),
//         children: Vec::new(),
//
//         feature: None,
//         split: None,
//
//         output_features: new_output_features,
//         input_features: new_input_features,
//
//         medians: medians,
//         feature_weights: feature_weights,
//         dispersions: dispersions,
//         local_gains: None,
//         absolute_gains: None
//     };
//
//
//     child
// }
//
//
// pub fn derive_children(&mut self) {
//
//         let (feature,_dispersion,split_value, left_indecies,right_indecies) = self.best_split();
//
//         let mut left_child_id = self.id.clone();
//         let mut right_child_id = self.id.clone();
//         left_child_id.push_str(&format!(":F{}S{}L",feature,split_value));
//         right_child_id.push_str(&format!(":F{}S{}R",feature,split_value));
//
//         let left_child = self.derive(&left_indecies, &left_child_id);
//         let right_child = self.derive(&right_indecies, &right_child_id);
//         println!("{:?}",left_child.samples());
//         println!("{:?}", right_child.samples());
//
//         self.report(true);
//         left_child.report(true);
//         right_child.report(true);
//
//         self.children.push(left_child);
//         self.children.push(right_child);
// }

// pub fn generate_weak<T>(target:T) -> (T,Weak<T>) {
//     let arc_t = Arc::new(target);
//     let weak_t = Arc::downgrade(&arc_t);
//     match Arc::try_unwrap(arc_t) {
//         Ok(object) => return(object,weak_t),
//         Err(err) => panic!("Tried to unwrap an empty reference, something went wrong with weak reference construction!")
//     }
// }

// pub fn parallel_best_split(& mut self) -> (String,f64,f64,Vec<usize>,Vec<usize>) {
//
//     // pool: mpsc::Sender<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>),mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>
//
//     if self.input_features.len() < 1 {
//         panic!("Tried to split with no input features");
//     };
//
//     let mut feature_receivers: Vec<mpsc::Receiver<(usize,usize,f64,Vec<usize>)>> = Vec::with_capacity(self.input_features.len());
//
//     for feature in &self.input_features {
//
//         let feature_index = self.rank_table.feature_index(feature);
//         let splitters = self.rank_table.split(feature);
//         let mut feature_weights = self.feature_weights.clone();
//         feature_weights[feature_index] = 0.;
//
//         let (tx,rx) = mpsc::channel();
//
//         self.pool.send(((feature_index,splitters,feature_weights),tx)).unwrap();
//
//         feature_receivers.push(rx);
//
//     }
//
//     let mut feature_dispersions: Vec<(usize,usize,f64,Vec<usize>)> = Vec::with_capacity(self.input_features.len());
//
//     for receiver in feature_receivers {
//         feature_dispersions.push(receiver.recv().unwrap());
//     }
//
//     let mut minimum_dispersion = (0,feature_dispersions[0].clone());
//
//     for (i,feature) in feature_dispersions.iter().enumerate() {
//         if i == 0 {
//             continue
//         }
//         else {
//             if feature.2 < (minimum_dispersion.1).2 {
//                 minimum_dispersion = (i,feature.clone());
//             }
//         }
//     }
//
//     let (feature_index,(split_index, split_sample_index, split_dispersion, split_order)) = minimum_dispersion;
//
//     let best_feature = self.input_features[feature_index].clone();
//
//     let split_value = self.rank_table.feature_fetch(&best_feature,split_sample_index);
//
//     self.feature = Some(best_feature.clone());
//     self.split = Some(split_value.clone());
//
//     println!("Best split: {:?}", (best_feature.clone(),split_index, split_value,split_dispersion));
//
//     (best_feature,split_dispersion,split_value,split_order[..split_index].iter().cloned().collect(),split_order[split_index..].iter().cloned().collect())
//
// }
