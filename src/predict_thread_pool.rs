use std;
use std::sync::Arc;

use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use std::thread;

extern crate rand;

use std::cmp::Ordering;

use compact_predictor::interval_stack;
use compact_predictor::max_interval;

impl PredictThreadPool{
    pub fn new(processors: usize) -> Sender<PredictionMessage> {

        println!("Initializing predictor thread pool, processors:{}", processors);

        if processors < 1 {
            panic!("Warning, no processors were allocated to the pool, quitting!");
        }

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(processors);

        for i in 0..(processors) {

            workers.push(Worker::new(i, worker_receiver_channel.clone()))

        }

        println!("Workers initialized");

        tx
    }

    pub fn terminate(channel: &mut Sender<PredictionMessage>) {
        while let Ok(()) = channel.send(PredictionMessage::Terminate) {}
    }

}


pub struct PredictThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<PredictionMessage>>>,
}


impl Worker{

    pub fn new(id:usize,channel:Arc<Mutex<Receiver<PredictionMessage>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {
                loop{
                    println!("Trying to acquire lock");
                    let message_option = channel.lock().unwrap().recv().ok();
                    println!("Received a message");
                    if let Some(PredictionMessage::Message(intervals,sender)) = message_option {
                        sender.send(max_interval(interval_stack(intervals))).expect("Tree worker thread error");
                    }
                    else if let Some(PredictionMessage::Terminate) = message_option {
                        break
                    }
                }
            }),
        }
    }
}

pub enum PredictionMessage {
    Message(Vec<(f64,f64,f64)>,Sender<f64>),
    Terminate
}

struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
    // worker_receiver_channel: Arc<Mutex<Receiver<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>>>,
}

// pub fn interval_stack(intervals: Vec<(f64,f64,f64)>) -> Vec<(f64,f64,f64)> {
//     let mut aggregate_intervals: Vec<f64> = intervals.iter().fold(Vec::with_capacity(intervals.len()*2), |mut acc,x| {acc.push(x.0); acc.push(x.1); acc});
//     aggregate_intervals.sort_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));
//     let mut aggregate_scores = vec![0.;aggregate_intervals.len()-1];
//     for (s_start,s_end,score) in intervals {
//         for (i,(w_start,w_end)) in aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).enumerate() {
//             if (*w_start >= s_start) && (*w_end <= s_end) {
//                 aggregate_scores[i] += score;
//             }
//             // else {
//             //     aggregate_scores[i] -= score;
//             // }
//         }
//     }
//     let scored: Vec<(f64,f64,f64)> = aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).zip(aggregate_scores.into_iter()).map(|((begin,end),score)| (*begin,*end,score)).collect();
//     let filtered: Vec<(f64,f64,f64)> = scored.into_iter().filter(|x| x.0 != x.1 && x.2 != 0.).collect();
//     filtered
// }
