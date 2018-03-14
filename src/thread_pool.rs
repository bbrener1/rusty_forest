
use std;
use std::sync::Arc;
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


impl ThreadPool{
    pub fn new(size: usize) -> Sender<((usize, (RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)> {

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(size);

        for i in 0..size {

            workers.push(Worker::new(i,worker_receiver_channel.clone()))
        }

        tx
    }
}


pub struct ThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>>>
}


impl Worker {

    pub fn new(id:usize,channel:Arc<Mutex<Receiver<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>>>) ->Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {
                loop{
                    let message = channel.lock().unwrap().recv().ok();
                    if let Some(((feature_index,(forward,reverse,order), weights),sender)) = message {
                        sender.send(split(feature_index,forward,reverse,order,weights));
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



fn split (feature_index: usize, forward: RankTableSplitter, reverse: RankTableSplitter, draw_order: Vec<usize>, feature_weights:Vec<f64>) -> (usize,usize,f64,Vec<usize>) {

    // println!("Splitting a node");

    // let start_time = time::PreciseTime::now();

    if forward.length != reverse.length {
    // if true {
        println!("Parallel split, iterator check:");
        println!("{}",forward.length);
        println!("{}",reverse.length);
        println!("{:?}",forward.draw_order);
        println!("{:?}",reverse.draw_order);
        panic!("DESYNCED DRAW ORDER, PANICING");
    }

    let mut fw_dsp = vec![0.;forward.length as usize];

    // println!("Constructed zero length vector?");

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
                div.powi(2) * feature_weights[x.0] * ((x.0 != feature_index) as i32 as f64) + acc
            })
            .sqrt();


    }

    rv_dsp.reverse();



    let (mut split_index, mut split_dispersion) = (0,std::f64::INFINITY);

    for (i,(fw,rv)) in fw_dsp.iter().zip(rv_dsp).enumerate() {

        let weighted_split = fw * ((fw_dsp.len()-i) as f64 / fw_dsp.len() as f64) + rv * (i as f64/ fw_dsp.len() as f64);

        if fw_dsp.len() > 6 && i > 2 && i < fw_dsp.len() - 3 {
            if weighted_split < split_dispersion {
                split_index = i;
                split_dispersion = weighted_split;
            }
        }
        else if fw_dsp.len() > 3 && fw_dsp.len() < 6 && i > 1 && i < fw_dsp.len() -1 {
            if weighted_split < split_dispersion {
                split_index = i;
                split_dispersion = weighted_split;
            }
        }
        else if fw_dsp.len() < 3 {
            if weighted_split < split_dispersion {
                split_index = i;
                split_dispersion = weighted_split;
            }
        }
    }

    let split_sample_index = 0;

    if draw_order.len() > 0 {
        let split_sample_index = draw_order[split_index];
    }

    let output = (split_index,split_sample_index,split_dispersion,draw_order);

    // let end_time = time::PreciseTime::now();

    // println!("Single split time: {}", start_time.to(end_time).num_microseconds().unwrap_or(-1));
    //
    // println!("Split output: {}",&output.0);

    output

    //
    // for combo in fw_dsp.iter().zip(rv_dsp.iter()) {
    //     println!("{:?},{}", combo, combo.0 + combo.1);
    // }
    //
    // let fw: &[f64];
    // let rv: &[f64];
    //
    // if fw_dsp.len() > 6 && rv_dsp.len() > 6 {
    //     fw = &fw_dsp[3..(fw_dsp.len()-3)];
    //     rv = &rv_dsp[3..(rv_dsp.len()-3)];
    // }
    // else if fw_dsp.len() > 2 && rv_dsp.len() > 2 {
    //     fw = &fw_dsp[1..(fw_dsp.len()-1)];
    //     rv = &rv_dsp[1..(rv_dsp.len()-1)];
    // }
    // else {
    //     fw = &fw_dsp[..];
    //     rv = &rv_dsp[..]
    // }
    //
    //
    // let (split_index, split_dispersion) = fw.iter().zip(rv.iter()).map(|x| x.0 + x.1).enumerate().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater)).unwrap_or((0,0.));
    //
    // let split_sample_index = draw_order[split_index];
    //
    //
    // let output = (split_index,split_sample_index,split_dispersion,draw_order);
    //
    // println!("Split output: {:?}",output.clone());
    //
    // output

}

//
// struct Worker {
//     id: usize,
//     thread: thread::JoinHandle<()>,
// }
