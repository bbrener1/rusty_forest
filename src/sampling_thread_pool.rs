// use std;
// use std::sync::Arc;
//
// use std::sync::mpsc;
// use std::sync::Mutex;
// use std::sync::mpsc::Receiver;
// use std::sync::mpsc::Sender;
//
// use std::thread;
//
// extern crate rand;
//
// impl<T:Clone> SamplingThreadPool<T>{
//     pub fn new(weight_vec:Vec<f64>,processors: usize) -> Sender<SamplingMessage<T>> {
//
//         let weights = Arc::new(Mutex::new(weight_vec));
//
//         if processors < 1 {
//             panic!("Warning, no processors were allocated to the pool, quitting!");
//         }
//
//         let (tx,rx) = mpsc::channel();
//
//         let worker_receiver_channel = Arc::new(Mutex::new(rx));
//
//         let mut workers = Vec::with_capacity(processors);
//
//         for i in 0..(processors/30) {
//
//             println!("Spawning tree pool worker");
//             println!("Prototype tree has {} threads", processors/(processors/30) - 1);
//
//             workers.push(Worker::new(weights.clone(),worker_receiver_channel.clone()))
//
//
//         }
//
//         tx
//     }
//
//     pub fn terminate(channel: &mut Sender<SamplingMessage<T>>) {
//         while let Ok(()) = channel.send(SamplingMessage::Terminate) {}
//     }
//
// }
//
// pub struct SamplingThreadPool<T> {
//     workers: Vec<Worker<T>>,
//     worker_receiver_channel: Arc<Mutex<Receiver<SamplingMessage<T>>>>,
// }
//
//
// impl<T: Clone> Worker<T>{
//
//     pub fn new(id:usize,mut weights: Arc<Mutex<Vec<f64>>>, channel:Arc<Mutex<Receiver<SamplingMessage<T>>>>) -> Worker<T> {
//         Worker{
//             id: id,
//             thread: std::thread::spawn(move || {
//                 loop{
//                     let message_option = channel.lock().unwrap().recv().ok();
//                     if let Some(message) = message_option {
//                         match message {
//                             SamplingMessage::Selections(tree_iter,input_features,output_features,samples,sender) => {
//                                 println!("Tree Pool: Request for tree: {}",tree_iter);
//                                 println!("Tree Pool: Deriving {}", tree_iter);
//                                 let mut tree = prototype.derive_specified(&samples.iter().collect(),&input_features.iter().collect(),&output_features.iter().collect(),tree_iter);
//                                 println!("Tree Pool: Growing {}", tree_iter);
//                                 tree.grow_branches();
//                                 println!("Tree Pool: Sending {}", tree_iter);
//                                 let p_tree = tree.strip_consume();
//                                 sender.send(p_tree).expect("Tree worker thread error");
//                             }
//                             SamplingMessage::Terminate => {
//                                 println!("Termination request");
//                                 break
//                             }
//                         }
//                     }
//                 }
//             }),
//         }
//     }
// }
//
//
// struct Worker<T> {
//     id: usize,
//     thread: thread::JoinHandle<()>,
//     // worker_receiver_channel: Arc<Mutex<Receiver<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>>>,
// }
//
// pub enum SamplingMessage<T> {
//     Inputs(usize,Vec<T>,Sender<Vec<T>>),
//     Outputs(usize,Vec<T>,Sender<Vec<T>>),
//     Terminate
// }
