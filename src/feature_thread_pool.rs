
use std;
use std::sync::Arc;

use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use std::thread;

extern crate rand;

use rank_vector::RankVector;


impl FeatureThreadPool{
    pub fn new(size: usize) -> Sender<((RankVector,Arc<Vec<usize>>), mpsc::Sender<(Vec<(f64,f64)>,RankVector)>)> {

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
    worker_receiver_channel: Arc<Mutex<Receiver<((RankVector,Arc<Vec<usize>>), mpsc::Sender<(Vec<(f64,f64)>,RankVector)>)>>>
}


impl Worker{

    pub fn new(id:usize,channel:Arc<Mutex<Receiver<((RankVector,Arc<Vec<usize>>), mpsc::Sender<(Vec<(f64,f64)>,RankVector)>)>>>) -> Worker {
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


struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
    // worker_receiver_channel: Arc<Mutex<Receiver<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>>>,
}



fn compute (mut vector: RankVector , draw_order: Arc<Vec<usize>>) -> (Vec<(f64,f64)>,RankVector) {

    let result = vector.ordered_mad(&*draw_order);
    vector.manual_reset();
    (result,vector)

}
