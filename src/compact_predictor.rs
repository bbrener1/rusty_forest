use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::sync::Arc;
use PredictionMode;
use DropMode;
use Parameters;
use std::collections::HashMap;
use shuffler::fragment_nodes;
use AveragingMode;
use WeighingMode;

extern crate rand;

use std::sync::mpsc;
use std::sync::mpsc::Sender;

use node::StrippedNode;
use tree_v2::PredictiveTree;
use predict_thread_pool::PredictThreadPool;
use predict_thread_pool::PredictionMessage;

pub fn compact_predict(trees: &[PredictiveTree], counts: &Vec<Vec<f64>>, features: &HashMap<String,usize>, parameters: Arc<Parameters>) -> Vec<Vec<f64>> {
    let mut predictions: Vec<Vec<f64>> = Vec::with_capacity(counts.len());
    let feature_intervals: Vec<Vec<(f64,f64,f64)>> = Vec::with_capacity(features.len());
    // println!("Predicting");
    // println!("{}",counts.len());
    // println!("Individual observations");

    let mut prediction_pool = PredictThreadPool::new(parameters.processor_limit.unwrap_or(1));
    let prediction_mode = parameters.prediction_mode.unwrap_or(PredictionMode::Abort);
    let drop_mode = parameters.dropout.unwrap_or(DropMode::Zeros);
    let weighing_mode = parameters.weighing_mode.unwrap_or(WeighingMode::AbsoluteGain);

    for sample in counts {
        let mut leaves = Vec::with_capacity(trees.len());
        println!("Trees: {}",trees.len());
        for tree in trees {
            leaves.push(tree.predict_leaves(sample,features,&prediction_mode,&drop_mode));
        }
        println!("Leaves: {}", leaves.iter().flat_map(|x| x).collect::<Vec<&&StrippedNode>>().len());

        let sample_prediction: Vec<f64>;

            /// Hard-coded alternative modes of averaging leaves. I'll add an option later.

        match parameters.averaging_mode.as_ref().unwrap() {
            &AveragingMode::Stacking => {
                let sample_intervals = intervals(leaves);
                sample_prediction = aggregate_predictions(sample_intervals, features, prediction_pool.clone());
            },
            &AveragingMode::Arithmetic => {

                match weighing_mode {
                    WeighingMode::AbsoluteGain => sample_prediction = average_leaves_gain(leaves, features),
                    WeighingMode::AbsGainSquared => sample_prediction = average_leaves_squared_gain(leaves, features),
                    WeighingMode::AbsoluteDispersion => sample_prediction = average_leaves_cov(leaves, features),
                    WeighingMode::AbsDispSquared => sample_prediction = average_leaves_cov_squared(leaves, features),
                    WeighingMode::Flat => sample_prediction = average_leaves_flat(leaves,features)
                }
            },
        }
        // println!("Intervals: {:?}", sample_intervals);
        predictions.push(sample_prediction);
        // println!("{}",predictions.len());

    }

    PredictThreadPool::terminate(&mut prediction_pool);

    predictions
}

pub fn average_leaves_squared_gain(nodes: Vec<Vec<&StrippedNode>>,features:&HashMap<String,usize>) -> Vec<f64> {

    let predictions = raw_median_gain(nodes);

    let mut agg_predictions = vec![0.;features.len()];

    for (feature,values) in predictions {
        let sum = values.iter().fold((0.,0.),|acc,x| (acc.0 + (x.0 * (x.1.max(0.).powi(2))), acc.1 + (x.1.max(0.).powi(2))));
        let mut mean = sum.0 / sum.1;
        if mean.is_nan() {
            mean = 0.;
        }
        agg_predictions[features[feature]] = mean;
    }

    agg_predictions

}

pub fn average_leaves_flat(nodes: Vec<Vec<&StrippedNode>>,features:&HashMap<String,usize>) -> Vec<f64> {

    let predictions = raw_median_gain(nodes);

    let mut agg_predictions = vec![0.;features.len()];

    for (feature,values) in predictions {
        let mut mean = values.iter().fold(0.,|acc,x| acc + x.0) / values.len() as f64;
        if mean.is_nan() {
            mean = 0.;
        }
        agg_predictions[features[feature]] = mean;
    }

    agg_predictions

}

pub fn average_leaves_gain(nodes: Vec<Vec<&StrippedNode>>,features:&HashMap<String,usize>) -> Vec<f64> {

    let predictions = raw_median_gain(nodes);

    let mut agg_predictions = vec![0.;features.len()];

    for (feature,values) in predictions {
        let sum = values.iter().fold((0.,0.),|acc,x| (acc.0 + (x.0 * (x.1.max(0.))), acc.1 + (x.1.max(0.))));
        let mut mean = sum.0 / sum.1;
        if mean.is_nan() {
            mean = 0.;
        }
        agg_predictions[features[feature]] = mean;
    }

    agg_predictions

}

pub fn average_leaves_cov(nodes: Vec<Vec<&StrippedNode>>,features:&HashMap<String,usize>) -> Vec<f64> {

    let predictions = raw_median_cov(nodes);

    let mut agg_predictions = vec![0.;features.len()];

    for (feature,values) in predictions {
        let sum = values.iter().fold((0.,0.),|acc,x| (acc.0 + (x.0 * (1./x.1.max(0.))), acc.1 + (1./x.1.max(0.))));
        let mut mean = sum.0 / sum.1;
        if mean.is_nan() {
            mean = 0.;
        }
        agg_predictions[features[feature]] = mean;
    }

    agg_predictions

}

pub fn average_leaves_cov_squared(nodes: Vec<Vec<&StrippedNode>>,features:&HashMap<String,usize>) -> Vec<f64> {

    let predictions = raw_median_cov(nodes);

    let mut agg_predictions = vec![0.;features.len()];

    for (feature,values) in predictions {
        let sum = values.iter().fold((0.,0.),|acc,x| (acc.0 + (x.0 * (1./x.1.max(0.))).powi(2), acc.1 + (1./x.1.max(0.)).powi(2)));
        let mut mean = sum.0 / sum.1;
        if mean.is_nan() {
            mean = 0.;
        }
        agg_predictions[features[feature]] = mean;
    }

    agg_predictions

}


pub fn raw_median_gain(nodes: Vec<Vec<&StrippedNode>>) -> HashMap<&String,Vec<(f64,f64)>> {

    let flat_nodes: Vec<&StrippedNode> = nodes.into_iter().flat_map(|x| x).collect();

    let mut predictions: HashMap<&String,Vec<(f64,f64)>> = HashMap::new();

    for node in flat_nodes {
        for (feature,(median,gain)) in node.features().iter().zip(node.medians().iter().zip(node.absolute_gains().as_ref().unwrap_or(&vec![]).iter())) {
            predictions.entry(feature).or_insert(Vec::new()).push((*median,*gain));
        }
    }

    predictions

}

pub fn raw_median_cov(nodes: Vec<Vec<&StrippedNode>>) -> HashMap<&String,Vec<(f64,f64)>> {

    let flat_nodes: Vec<&StrippedNode> = nodes.into_iter().flat_map(|x| x).collect();

    let mut predictions: HashMap<&String,Vec<(f64,f64)>> = HashMap::new();

    for node in flat_nodes {
        for (feature,(median,cov)) in node.features().iter().zip(node.medians().iter().zip(node.covs().iter())) {
            predictions.entry(feature).or_insert(Vec::new()).push((*median,*cov));
        }
    }

    predictions

}

pub fn raw_medians(nodes: Vec<Vec<&StrippedNode>>) -> HashMap<&String,Vec<f64>> {

    let flat_nodes: Vec<&StrippedNode> = nodes.into_iter().flat_map(|x| x).collect();

    let mut predictions: HashMap<&String,Vec<f64>> = HashMap::new();

    for node in flat_nodes {
        for (feature,median) in node.features().iter().zip(node.medians()) {
            predictions.entry(feature).or_insert(Vec::new()).push(*median);
        }
    }

    predictions

}

pub fn intervals<'a>(nodes: Vec<Vec<&'a StrippedNode>>) -> HashMap<&String,Vec<(f64,f64,f64)>> {

    let flat_nodes: Vec<&StrippedNode> = nodes.into_iter().flat_map(|x| x).collect();

    let mut intervals: HashMap<&String,Vec<(f64,f64,f64)>> = HashMap::new();

    for node in flat_nodes {
        for (feature,((median,mad),gain)) in node.features().iter().zip(node.medians().iter().zip(node.mads().iter()).zip(node.absolute_gains().as_ref().unwrap().iter())) {
            intervals.entry(feature).or_insert(Vec::new()).push((*median-*mad,*median+*mad,1.0));
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

        // println!("Receiving feature: {}", feature);
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

pub fn node_sample_encoding(nodes: &Vec<&StrippedNode>,header: &HashMap<String,usize>) -> Vec<Vec<bool>> {
    let mut encoding = vec![vec![false; nodes.len()]; header.len()];
    for (node_index,node) in nodes.iter().enumerate() {
        for sample in node.samples() {
            if let Some(sample_index) = header.get(sample) {
                encoding[*sample_index][node_index] = true;
            }
        }
    }
    encoding
}

pub fn median_matrix(nodes: &Vec<&StrippedNode>) -> Vec<Vec<f64>> {
    let mut matrix = Vec::with_capacity(nodes.len());
    for node in nodes {
        matrix.push(node.medians().clone())
    }
    matrix
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
