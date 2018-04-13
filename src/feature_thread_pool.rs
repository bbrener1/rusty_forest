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
                loop{
                    let message_option = channel.lock().unwrap().recv().ok();
                    if let Some(message) = message_option {
                        match message {
                            FeatureMessage::Message((vector,draw_order),sender) => {
                                sender.send(compute(vector,draw_order)).expect("Failed to send feature result");
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
    Message((RankVector,Arc<Vec<usize>>), mpsc::Sender<(Vec<(f64,f64)>,RankVector)>),
    Terminate
}

fn compute (mut vector: RankVector , draw_order: Arc<Vec<usize>>) -> (Vec<(f64,f64)>,RankVector) {

    let result = vector.ordered_mad(&*draw_order);
    vector.manual_reset();
    (result,vector)

}
