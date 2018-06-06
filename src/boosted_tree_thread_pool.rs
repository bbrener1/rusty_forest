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

impl BoostedTreeThreadPool{
    pub fn new(prototype:&Tree,processors: usize) -> Sender<BoostedMessage> {

        println!("Initializing boosted tree thread pool, processors:{}",processors);

        println!("Prototype tree: {},{},{}", prototype.input_features().len(), prototype.output_features().len(),prototype.root.samples().len());

        if processors < 1 {
            panic!("Warning, no processors were allocated to the pool, quitting!");
        }

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(processors);

        if processors > 30 {
            for i in 0..(processors/30) {

                println!("Spawning tree pool worker");
                println!("Prototype tree has {} threads", processors/(processors/30) - 1);

                workers.push(Worker::new(i,prototype.pool_switch_clone(processors/(processors/30) - 1 ),worker_receiver_channel.clone()))

            }
        }
        else {
            workers.push(Worker::new(0,prototype.pool_switch_clone(processors), worker_receiver_channel.clone()))
        }


        tx
    }

    pub fn terminate(channel: &mut Sender<BoostedMessage>) {
        while let Ok(()) = channel.send(BoostedMessage::Terminate) {}
    }

}

pub struct BoostedTreeThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<BoostedMessage>>>,
}


impl Worker{

    pub fn new(id:usize,mut prototype:Tree, channel:Arc<Mutex<Receiver<BoostedMessage>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {
                loop{
                    let message_option = channel.lock().unwrap().recv().ok();
                    if let Some(message) = message_option {
                        match message {
                            BoostedMessage::Selections(tree_iter,input_features,output_features,samples,sender) => {
                                println!("Tree Pool: Request for tree: {}",tree_iter);
                                println!("Tree Pool: Deriving {}", tree_iter);
                                let mut tree = prototype.derive_specified(&samples.iter().collect(),&input_features.iter().collect(),&output_features.iter().collect(),tree_iter);
                                println!("Tree Pool: Growing {}", tree_iter);
                                tree.grow_branches();
                                println!("Tree Pool: Sending {}", tree_iter);
                                let p_tree = tree.strip_consume();
                                sender.send(p_tree).expect("Tree worker thread error");
                            }
                            BoostedMessage::Terminate => {
                                println!("Termination request");
                                prototype.terminate_pool();
                                break
                            }
                        }
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

pub enum BoostedMessage {
    Selections(usize,Vec<String>,Vec<String>,Vec<String>,Sender<PredictiveTree>),
    Terminate
}
