use std::cmp::PartialOrd;
use std::cmp::Ordering;
use PredictionMode;
use std::collections::HashMap;

extern crate rand;


use node::Node;
use tree::Tree;

pub fn predict(trees: &Vec<Tree>, counts: &Vec<Vec<f64>>, features: &HashMap<String,usize>, prediction_mode: &PredictionMode) -> Vec<Vec<f64>> {
    let mut predictions: Vec<Vec<f64>> = Vec::with_capacity(counts.len());
    let feature_intervals: Vec<Vec<(f64,f64,f64)>> = Vec::with_capacity(features.len());
    for sample in counts {
        let mut leaves = Vec::with_capacity(trees.len());
        for tree in trees {
            leaves.push(node_predict_leaves(&tree.root,sample,features,prediction_mode));
        }
        let sample_intervals = intervals(leaves);
        let sample_prediction = aggregate_predictions(sample_intervals, features);
        predictions.push(sample_prediction);

    }
    predictions
}

pub fn node_predict_leaves<'a>(node: &'a Node, vector: &Vec<f64>, header: &HashMap<String,usize>, prediction_mode: &PredictionMode) -> Vec<&'a Node> {

    let mut leaves: Vec<&Node> = Vec::new();

    if let (&Some(ref feature),&Some(ref split)) = (&node.feature,&node.split) {
        if *vector.get(*header.get(feature).unwrap_or(&(vector.len()+1))).unwrap_or(&0.) != 0.  {
            if vector[header[feature]] > split.clone() {
                leaves.append(&mut node_predict_leaves(&node.children[1], vector, header, prediction_mode));
            }
            else {
                leaves.append(&mut node_predict_leaves(&node.children[0], vector, header, prediction_mode));
            }
        }
        else {
            match prediction_mode {
                &PredictionMode::Branch => {
                    leaves.append(&mut node_predict_leaves(&node.children[0], vector, header, prediction_mode));
                    leaves.append(&mut node_predict_leaves(&node.children[1], vector, header, prediction_mode));
                },
                &PredictionMode::Truncate => {
                    leaves.push(&node)
                },
                &PredictionMode::Abort => {},
                &PredictionMode::Auto => {
                    leaves.append(&mut node_predict_leaves(&node, vector, header, &PredictionMode::Branch));
                }
            }
        }
    }
    else {
        leaves.push(&node);
    }

    return leaves

}

pub fn intervals<'a>(nodes: Vec<Vec<&'a Node>>) -> HashMap<&String,Vec<(f64,f64,f64)>> {

    let flat_nodes: Vec<&Node> = nodes.into_iter().flat_map(|x| x).collect();

    let mut intervals: HashMap<&String,Vec<(f64,f64,f64)>> = HashMap::new();

    for node in flat_nodes {
        for (feature,(median,mad)) in node.features().iter().zip(node.medians().iter().zip(node.mads().iter())) {
            intervals.entry(feature).or_insert(Vec::new()).push((*median-*mad,*median+*mad,1.));
        }
    }

    intervals
}

pub fn aggregate_predictions(feature_intervals:HashMap<&String,Vec<(f64,f64,f64)>>,features: &HashMap<String,usize>) -> Vec<f64> {
    let mut predictions = vec![0.;features.len()];

    for (feature,intervals) in feature_intervals.into_iter() {
        predictions[features[feature]] = max_interval(interval_stack(intervals));
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
    let scored = aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).zip(aggregate_scores.into_iter()).map(|((begin,end),score)| (*begin,*end,score)).collect();
    scored
}

pub fn max_interval(intervals: Vec<(f64,f64,f64)>) -> f64 {
    intervals.into_iter().max_by(|a,b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Greater)).unwrap_or((0.,0.,0.)).2
}
