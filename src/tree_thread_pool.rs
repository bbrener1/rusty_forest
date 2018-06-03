use std;
use std::sync::Arc;

use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use std::thread;

extern crate rand;

use tree::Tree;
use tree::PredictiveTree;
use Parameters;

impl TreeThreadPool{
    pub fn new(prototype:&Tree, parameters: Arc<Parameters>) -> Sender<(usize, mpsc::Sender<PredictiveTree>)> {

        println!("Prototype tree: {},{},{}", prototype.input_features().len(), prototype.output_features().len(),prototype.root.samples().len());
        println!("Parameters:{:?},{:?},{:?}", parameters.input_features,parameters.output_features,parameters.sample_subsample);

        let processors = parameters.processor_limit.unwrap_or(1);
        let samples_per_tree = parameters.sample_subsample.unwrap_or(1);
        let input_features = parameters.input_features.unwrap_or(1);
        let output_features = parameters.output_features.unwrap_or(1);

        if processors < 1 {
            panic!("Warning, no processors were allocated to the pool, quitting!");
        }

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(processors);

        // if processors > 6 {
        //     for i in 0..(processors/6) {
        //
        //         println!("Spawning tree pool worker");
        //         println!("Prototype tree has {} threads", processors/(processors/6));
        //
        //         workers.push(Worker::new(i,prototype.pool_switch_clone(processors/(processors/6)),samples_per_tree,input_features,output_features, worker_receiver_channel.clone()))
        //
        //     }
        // }
        // else {
        //     workers.push(Worker::new(0,prototype.pool_switch_clone(processors),samples_per_tree,input_features,output_features, worker_receiver_channel.clone()))
        // }

        let prototype = prototype.pool_switch_clone(processors);

        for i in 0..(processors/5) {

                println!("Spawning tree pool worker");
                workers.push(Worker::new(i,prototype.clone(),samples_per_tree,input_features,output_features, worker_receiver_channel.clone()))

        }


        tx
    }

    pub fn terminate(channel: &mut Sender<(usize, mpsc::Sender<PredictiveTree>)>) {
        while let Ok(()) = channel.send((0,mpsc::channel().0)) {}
    }

}

pub struct TreeThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<(Tree, mpsc::Sender<PredictiveTree>)>>>,
}


impl Worker{

    pub fn new(id:usize,mut prototype:Tree,samples_per_tree:usize,input_features:usize,output_features:usize, channel:Arc<Mutex<Receiver<(usize, mpsc::Sender<PredictiveTree>)>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {
                loop{
                    let message = channel.lock().unwrap().recv().ok();
                    if let Some((tree_iter,sender)) = message {
                        if tree_iter == 0 {
                            println!("Termination request");
                            prototype.terminate_pool();
                            break
                        }
                        println!("Tree Pool: Request for tree: {}",tree_iter);
                        println!("Tree Pool: Deriving {}", tree_iter);
                        let mut tree = prototype.derive_from_prototype(samples_per_tree,input_features,output_features,tree_iter);
                        println!("Tree Pool: Growing {}", tree_iter);
                        tree.grow_branches();
                        println!("Tree Pool: Sending {}", tree_iter);
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
