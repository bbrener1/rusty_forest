use std;
use std::collections::HashSet;
use std::mem::replace;
use std::sync::Arc;

use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use std::thread;

extern crate rand;

use smallvec::SmallVec;

use rv3::RankVector;
use rv3::Node;
use SplitMode;

impl FeatureThreadPool{
    pub fn new(size: usize) -> Sender<FeatureMessage> {

        if size < 1 {
            panic!("Warning, no processors were allocated to the pool, quitting!");
        }

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(size);

        for i in 0..size {

            workers.push(Worker::new(i,worker_receiver_channel.clone()))

        }

        tx
    }

    pub fn terminate(channel: &mut Sender<FeatureMessage>) {
        while let Ok(()) = channel.send(FeatureMessage::Terminate) {};
    }

}


pub struct FeatureThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<FeatureMessage>>>,
    sender: Sender<FeatureMessage>
}


impl Worker{

    pub fn new(id:usize,channel:Arc<Mutex<Receiver<FeatureMessage>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {

                let mut local_container: Option<SmallVec<[Node;1024]>> = Some(SmallVec::new());

                loop{
                    let message_option = channel.lock().unwrap().recv().ok();
                    if let Some(message) = message_option {
                        match message {
                            FeatureMessage::Message((vector,draw_order,drop_set,split_mode),sender) => {
                                sender.send(compute(vector,draw_order,drop_set,split_mode,&mut local_container)).expect("Failed to send feature result");
                            },
                            FeatureMessage::Terminate => break
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


pub enum FeatureMessage {
    Message((RankVector<Vec<Node>>,Arc<Vec<usize>>,Arc<HashSet<usize>>,SplitMode), mpsc::Sender<(Vec<f64>,RankVector<Vec<Node>>)>),
    Terminate
}

fn compute (prot_vector: RankVector<Vec<Node>> , draw_order: Arc<Vec<usize>> , drop_set: Arc<HashSet<usize>>, split_mode:SplitMode, local_container:&mut Option<SmallVec<[Node;1024]>>) -> (Vec<f64>,RankVector<Vec<Node>>) {

    let mut vector = prot_vector.clone_to_container(replace(local_container, None).unwrap());

    let result = match split_mode {
        SplitMode::Cov => vector.ordered_covs(&draw_order,&drop_set),
        SplitMode::MAD => vector.ordered_mads(&draw_order,&drop_set),
        SplitMode::CovSquared => vector.ordered_covs(&draw_order,&drop_set),
        SplitMode::MADSquared => vector.ordered_mads(&draw_order,&drop_set),
    };

    replace(local_container, Some(vector.return_container()));

    // println!("parallel: {:?}", result);

    (result,prot_vector)

}
