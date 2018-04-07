use std;
use std::sync::Arc;

use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::SyncSender;
use std::sync::mpsc::Sender;

use std::thread;

extern crate rand;

use tree::Tree;
use tree::PredictiveTree;

impl TreeThreadPool{
    pub fn new(prototype:&Tree,features_per_tree:usize,samples_per_tree:usize,input_features:usize,output_features:usize,processors: usize) -> Sender<(usize, mpsc::Sender<PredictiveTree>)> {

        println!("Initializing thread pool, args:");
        println!("{},{},{},{}",features_per_tree,samples_per_tree,input_features,output_features);

        println!("Prototype tree: {},{},{}", prototype.input_features().len(), prototype.output_features().len(),prototype.root.samples().len());

        if processors < 1 {
            panic!("Warning, no processors were allocated to the pool, quitting!");
        }

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(processors);

        for i in 0..processors {

            workers.push(Worker::new(i,prototype.clone(),features_per_tree,samples_per_tree,input_features,output_features, worker_receiver_channel.clone()))

        }

        tx
    }

}


pub struct TreeThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<(Tree, mpsc::Sender<PredictiveTree>)>>>,
}


impl Worker{

    pub fn new(id:usize,prototype:Tree,features_per_tree:usize,samples_per_tree:usize,input_features:usize,output_features:usize, channel:Arc<Mutex<Receiver<(usize, mpsc::Sender<PredictiveTree>)>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {
                loop{
                    let message = channel.lock().unwrap().recv().ok();
                    if let Some((tree_iter,sender)) = message {
                        println!("Tree Pool: Request for tree: {}",tree_iter);
                        println!("Tree Pool: Deriving {}", tree_iter);
                        let mut tree = prototype.derive_from_prototype(features_per_tree,samples_per_tree,input_features,output_features,tree_iter);
                        println!("Tree Pool: Growing {}", tree_iter);
                        tree.grow_branches();
                        println!("Tree Pool: Sending {}", tree_iter);
                        tree.serialize_compact();
                        let p_tree = tree.strip_consume();
                        sender.send(p_tree).expect("Tree worker thread error");
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
