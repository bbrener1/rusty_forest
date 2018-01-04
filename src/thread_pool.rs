#![recursion_limit="128"]

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
use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;


use std::thread;

extern crate rand;
use rand::Rng;

use rank_table::RankTable;
use rank_table::RankTableSplitter;
use node::Node;


impl<'a,U:Clone + std::cmp::Eq + std::hash::Hash + Debug,T:Clone + std::cmp::Eq + std::hash::Hash + Debug> ThreadPool<'a,U,T>{
    pub fn new(size: usize) -> Sender<(&'a U,&'a Node<U,T>,Sender<(U,usize,T,usize,f64,f64,Vec<usize>)>)> {

        let (tx,rx) = mpsc::channel();

        let mut workers = Vec::with_capacity(size);

        for i in 0..size {
            workers.push(Worker::new(i))
        }

        tx
    }
}


pub struct ThreadPool<'a,U:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug,T:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug> {
    workers: Vec<Worker<'a,U,T>>,
    distributor_channel: Sender<(&'a U,&'a Node<U,T>,Sender<(U,usize,T,usize,f64,f64,Vec<usize>)>)>,
    worker_receiver_channel: Arc<Mutex<Receiver<(&'a U, &'a Node<U,T>, mpsc::Sender<(U,usize,T,usize,f64,f64,Vec<usize>)>)>>>
}


impl <'a,U:Clone + std::cmp::Eq + std::hash::Hash + Debug,T:Clone + std::cmp::Eq + std::hash::Hash + Debug> Worker<'a,U,T> {

    pub fn new(id:usize,channel:Arc<Mutex<Receiver<(&'a U, &'a RankTable<U,T>,&'a[f64], mpsc::Sender<(U,usize,T,usize,f64,f64,Vec<usize>)>)>>>) ->Worker<'a,U,T> {
        Worker{
            id: id,
            thread: std::thread::spawn(|| {
                while let Some((feature,rank_table, weights,sender)) = channel.lock().unwrap().recv().ok() {
                    sender.send(split(feature,rank_table,weights));
                }
            }),
            worker_receiver_channel: channel
        }
    }

    pub fn compute(feature: &'a U, node: &'a Node<U,T>, return_channel: mpsc::Sender<(U,usize,T,usize,f64,f64,Vec<usize>)>) {
        return_channel.send(node.split(feature))
    }
}

struct Worker<'a,U:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug,T:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug> {
    id: usize,
    thread: thread::JoinHandle<()>,
    worker_receiver_channel: Arc<Mutex<Receiver<(&'a U, &'a RankTable<U,T>, &'a[f64], mpsc::Sender<(U,usize,T,usize,f64,f64,Vec<usize>)>)>>>,
}



fn split<'a,U:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug,T:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug> (feature: &'a U,rank_table: &'a RankTable<U,T>, feature_weights:&[f64]) -> (U,usize,T,usize,f64,f64,Vec<usize>) {

    println!("Splitting a node");

    let feature_index = rank_table.feature_index(feature);

    let (forward,reverse,draw_order) = rank_table.split(feature);

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
                div.powi(2) * feature_weights[x.0] * ((x.0 != feature_index) as i32 as f64) + acc
            })
            .sqrt();

    }

    let mut rv_dsp = vec![0.;reverse.length];

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
                div.powi(2) * feature_weights[x.0] * ((x.0 != feature_index) as i32 as f64) + acc
            })
            .sqrt();


    }

    rv_dsp.reverse();

    for combo in fw_dsp.iter().zip(rv_dsp.iter()) {
        println!("{:?},{}", combo, combo.0 + combo.1);
    }

    let fw: &[f64];
    let rv: &[f64];

    if fw_dsp.len() > 6 && rv_dsp.len() > 6 {
        fw = &fw_dsp[3..(fw_dsp.len()-3)];
        rv = &rv_dsp[3..(rv_dsp.len()-3)];
    }
    else if fw_dsp.len() > 2 && rv_dsp.len() > 2 {
        fw = &fw_dsp[1..(fw_dsp.len()-1)];
        rv = &rv_dsp[1..(rv_dsp.len()-1)];
    }
    else {
        fw = &fw_dsp[..];
        rv = &rv_dsp[..]
    }


    let (split_index, split_dispersion) = fw.iter().zip(rv.iter()).map(|x| x.0 + x.1).enumerate().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or((0,0.));

    let split_sample_value = rank_table.feature_fetch(feature, draw_order[split_index]);

    let split_sample_index = draw_order[split_index];

    let split_sample_name = rank_table.sample_name(split_sample_index);

    let output = (feature.clone(),split_index,split_sample_name,split_sample_index,split_sample_value,split_dispersion,draw_order);

    println!("Split output: {:?}",output.clone());

    output

}

//
// struct Worker {
//     id: usize,
//     thread: thread::JoinHandle<()>,
// }
