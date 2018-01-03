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

use std::thread;

extern crate rand;
use rand::Rng;

use rank_table::RankTable;
use rank_table::RankTableSplitter;
use node::Node;

impl ThreadPool{

}

pub struct ThreadPool {
    workers: Vec<Worker>
}

struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
}
