use std::cmp::PartialOrd;
use std::cmp::Ordering;
use PredictionMode;
use DropMode;
use std::collections::HashMap;
use shuffler::fragment_nodes;

extern crate rand;

use std::sync::mpsc;
use std::sync::mpsc::Sender;

use node::StrippedNode;
use tree::PredictiveTree;
use predict_thread_pool::PredictThreadPool;
use predict_thread_pool::PredictionMessage;

pub fn compact_predict(trees: &Vec<PredictiveTree>, counts: &Vec<Vec<f64>>, features: &HashMap<String,usize>, prediction_mode: &PredictionMode,drop_mode: &DropMode, processor_limit: usize) -> Vec<Vec<f64>> {
    let mut predictions: Vec<Vec<f64>> = Vec::with_capacity(counts.len());
    let feature_intervals: Vec<Vec<(f64,f64,f64)>> = Vec::with_capacity(features.len());
    // println!("Predicting");
    // println!("{}",counts.len());
    // println!("Individual observations");

    let mut prediction_pool = PredictThreadPool::new(processor_limit);


    for sample in counts {
        let mut leaves = Vec::with_capacity(trees.len());
        println!("Trees: {}",trees.len());
        for tree in trees {
            leaves.push(node_predict_leaves(&tree.root,sample,features,prediction_mode,drop_mode));
        }
        println!("Leaves: {}", leaves.len());

        let sample_prediction: Vec<f64>;

            /// Hard-coded alternative modes of averaging leaves. I'll add an option later.

        match true {
            true => {
                let sample_intervals = intervals(leaves);
                sample_prediction = aggregate_predictions(sample_intervals, features, prediction_pool.clone());
            },
            _ => sample_prediction = average_leaves(leaves, features),
        }
        // println!("Intervals: {:?}", sample_intervals);
        predictions.push(sample_prediction);
        // println!("{}",predictions.len());

    }

    prediction_pool.send(PredictionMessage::Terminate);

    predictions
}

pub fn node_predict_leaves<'a>(node: &'a StrippedNode, vector: &Vec<f64>, header: &HashMap<String,usize>, prediction_mode: &PredictionMode, drop_mode: &DropMode) -> Vec<&'a StrippedNode> {

    // println!("Crawling node: {:?},{:?}", node.feature(),node.split());

    let mut leaves: Vec<&StrippedNode> = Vec::new();

    if let (&Some(ref feature),&Some(ref split)) = (node.feature(),node.split()) {
        if *vector.get(*header.get(feature).unwrap_or(&(vector.len()+1))).unwrap_or(&drop_mode.cmp()) != drop_mode.cmp() {
            // println!("Observing: {}", vector[header[feature]]);
            if vector[header[feature]] > *split {
                // println!("More than split, going right???");
                leaves.append(&mut node_predict_leaves(&node.children[1], vector, header, prediction_mode, drop_mode));
            }
            else {
                // println!("Less than split, going left???");
                leaves.append(&mut node_predict_leaves(&node.children[0], vector, header, prediction_mode, drop_mode));
            }
        }
        else {
            // println!("Branching");
            match prediction_mode {
                &PredictionMode::Branch => {
                    leaves.append(&mut node_predict_leaves(&node.children[0], vector, header, prediction_mode, drop_mode));
                    leaves.append(&mut node_predict_leaves(&node.children[1], vector, header, prediction_mode, drop_mode));
                },
                &PredictionMode::Truncate => {
                    leaves.push(&node)
                },
                &PredictionMode::Abort => {},
                &PredictionMode::Auto => {
                    leaves.append(&mut node_predict_leaves(&node, vector, header, &PredictionMode::Branch, drop_mode));
                }
            }
        }
    }
    else {
        // println!("Found a leaf");
        leaves.push(&node);
    }

    return leaves

}

pub fn average_leaves(nodes: Vec<Vec<&StrippedNode>>,features:&HashMap<String,usize>) -> Vec<f64> {

    let flat_nodes: Vec<&StrippedNode> = nodes.into_iter().flat_map(|x| x).collect();

    let mut predictions: HashMap<&String,Vec<(f64,f64)>> = HashMap::new();

    for node in flat_nodes {
        for (feature,(median,gain)) in node.features().iter().zip(node.medians().iter().zip(node.absolute_gains().as_ref().unwrap_or(&vec![]).iter())) {
            predictions.entry(feature).or_insert(Vec::new()).push((*median,*gain));
        }
    }

    let mut agg_predictions = vec![0.;features.len()];

    for (feature,values) in predictions {
        let sum = values.iter().fold((0.,0.),|acc,x| (acc.0 + (x.0 * x.1.max(0.)), acc.1 + x.1.max(0.)));
        let mut mean = sum.0 / sum.1;
        if mean.is_nan() {
            mean = 0.;
        }
        agg_predictions[features[feature]] = mean;
    }

    agg_predictions

}

pub fn intervals<'a>(nodes: Vec<Vec<&'a StrippedNode>>) -> HashMap<&String,Vec<(f64,f64,f64)>> {

    let flat_nodes: Vec<&StrippedNode> = nodes.into_iter().flat_map(|x| x).collect();

    let mut intervals: HashMap<&String,Vec<(f64,f64,f64)>> = HashMap::new();

    for node in flat_nodes {
        for (feature,(median,mad)) in node.features().iter().zip(node.medians().iter().zip(node.mads().iter())) {
            intervals.entry(feature).or_insert(Vec::new()).push((*median-*mad,*median+*mad,1.));
        }
    }

    // println!("Features with intervals: {}", intervals.len());
    // println!("Intervals: {}", intervals.iter().map(|x| x.1.len()).sum::<usize>());
    // println!("{:?}", intervals);

    intervals
}

pub fn aggregate_predictions(feature_intervals:HashMap<&String,Vec<(f64,f64,f64)>>,features: &HashMap<String,usize>,prediction_pool: Sender<PredictionMessage>) -> Vec<f64> {

    let mut predictions = vec![0.;features.len()];

    let mut receivers = Vec::with_capacity(feature_intervals.len());

    for (feature,intervals) in feature_intervals.into_iter() {

        let (tx,rx) = mpsc::channel();

        prediction_pool.send(PredictionMessage::Message(intervals,tx)).expect("Failed to send feature");

        receivers.push((feature,rx));

    }

    for (feature,reciever) in receivers {

        println!("Receiving feature: {}", feature);
        predictions[features[feature]] = reciever.recv().expect("Predictor pool error");

    }

    predictions
}


pub fn interval_stack(intervals: Vec<(f64,f64,f64)>) -> Vec<(f64,f64,f64)> {
    let mut aggregate_intervals: Vec<f64> = intervals.iter().fold(Vec::with_capacity(intervals.len()*2), |mut acc,x| {acc.push(x.0); acc.push(x.1); acc});
    aggregate_intervals.sort_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));
    let mut aggregate_scores = vec![0.;aggregate_intervals.len()-1];
    for (s_start,s_end,score) in intervals {
        for (i,(w_start,w_end)) in aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).enumerate() {
            if (*w_start >= s_start) && (*w_end <= s_end) {
                aggregate_scores[i] += score;
            }
            // else {
            //     aggregate_scores[i] -= score;
            // }
        }
    }
    let scored: Vec<(f64,f64,f64)> = aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).zip(aggregate_scores.into_iter()).map(|((begin,end),score)| (*begin,*end,score)).collect();
    let filtered: Vec<(f64,f64,f64)> = scored.into_iter().filter(|x| x.0 != x.1 && x.2 != 0.).collect();
    filtered
}

pub fn max_interval(intervals: Vec<(f64,f64,f64)>) -> f64 {
    let max = intervals.into_iter().max_by(|a,b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Greater)).unwrap_or((0.,0.,0.));
    return (max.0 + max.1)/2.
}

#[cfg(test)]
mod predictor_testing {
    use super::*;

    #[test]
    fn interval_stack_test_bookend() {
        let new_intervals = interval_stack(vec![(0.,20.,1.),(10.,20.,2.)]);
        assert_eq!(new_intervals,vec![(0.,10.,1.),(10.,20.,3.)]);
    }

    #[test]
    fn interval_stack_test_no_overlap() {
        let new_intervals = interval_stack(vec![(0.,10.,1.),(20.,30.,1.)]);
        assert_eq!(new_intervals,vec![(0.,10.,1.),(20.,30.,1.)])
    }

    #[test]
    fn interval_stack_total_overlap() {
        let new_intervals = interval_stack(vec![(0.,10.,1.),(0.,10.,1.)]);
        assert_eq!(new_intervals,vec![(0.,10.,2.)]);
    }

    #[test]
    fn interval_stack_internal() {
        let new_intervals = interval_stack(vec![(0.,10.,1.),(4.,6.,1.),(5.,7.,1.)]);
        assert_eq!(new_intervals,vec![(0.,4.,1.),(4.,5.,2.),(5.,6.,3.),(6.,7.,2.),(7.,10.,1.)]);
    }

    fn interval_stack_max() {
        let new_intervals = interval_stack(vec![(0.,10.,1.),(4.,6.,1.),(5.,7.,1.)]);
        assert_eq!(max_interval(new_intervals),5.5);
    }

}
