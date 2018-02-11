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
use time;

use std::thread;

extern crate rand;
use rand::Rng;

use rank_vector::RankVector;
use node::Node;


impl FeatureThreadPool{
    pub fn new(size: usize) -> Sender<((RankVector,Arc<Vec<usize>>), mpsc::Sender<Vec<(f64,f64)>>)> {

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(size);

        for i in 0..size {

            workers.push(Worker::new(i,worker_receiver_channel.clone()))
        }

        tx
    }

}


pub struct FeatureThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<((RankVector,Arc<Vec<usize>>), mpsc::Sender<Vec<(f64,f64)>>)>>>
}


impl Worker {

    pub fn new(id:usize,channel:Arc<Mutex<Receiver<((RankVector,Arc<Vec<usize>>), mpsc::Sender<Vec<(f64,f64)>>)>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {
                loop{
                    let message = channel.lock().unwrap().recv().ok();
                    if let Some(((vector,draw_order),sender)) = message {
                        sender.send(compute(vector,draw_order));
                    }
                }
            }),
        }
    }
}

    // pub fn compute(feature: & str, rank_table: & RankTable, feature_weights: &[f64], return_channel: mpsc::Sender<(String,usize,String,usize,f64,f64,Vec<usize>)>) {
    //     return_channel.send(split(feature,rank_table,feature_weights));
    // }


struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
    // worker_receiver_channel: Arc<Mutex<Receiver<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>>>,
}



fn compute (vector: RankVector , draw_order: Arc<Vec<usize>>) -> Vec<(f64,f64)> {

    vector.ordered_mad(&*draw_order)

    // let end_time = time::PreciseTime::now();

    // println!("Single split time: {}", start_time.to(end_time).num_microseconds().unwrap_or(-1));
    //
    // println!("Split output: {}",&output.0);


}
