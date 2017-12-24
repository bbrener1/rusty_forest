use std;
use std::marker::PhantomData;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::cmp::max;
use std::collections::HashSet;
use std::collections::HashMap;

extern crate rand;
use rand::Rng;

use raw_vector::RawVector;

impl<U,T> RankVector<U,T> {

    pub fn new(in_vec:&Vec<f64>, drop:f64, feature_name: U, sample_names: Vec<T>) -> RankVector<U,T> {

        let vector = RawVector::raw_vector(in_vec);

        let median = RankVector::<U,T>::describe(&vector);

        let left_boundary = 0;
        let right_boundary = 0;
        let length = vector.len();

        let zones = RankVector::<U,T>::empty_zones();

        RankVector{

            length: length,

            source_vector: vector.clone(),
            vector: vector,

            draw_order: (0..length).collect(),

            drop: false,
            num_dropped: 0,

            median: median,

            left_zone:zones.0,
            median_zone:zones.1,
            right_zone:zones.2,

            left_boundary: left_boundary,
            right_boundary: right_boundary,

            finger: 0,

            feature_name: feature_name,
            sample_names: sample_names
        }

    }

    // pub fn desrciption(&mut self) {
    //
    // }

    pub fn describe(input:&RawVector) -> (usize,usize,f64) {

        let non_zero_length = input.vector.len() - input.drop_set.len();

        let median_target = ((non_zero_length as f64 / 2.).trunc() + 1.) as i32;



        let mut median: (usize,usize,f64);

        match non_zero_length%2 {
            0 => median = ((non_zero_length/2)-1,non_zero_length/2,0.),
            1 => median = (non_zero_length/2,non_zero_length/2,0.),
            _ => median = (0,0,0.)
        }

        median = (input.left_to_right().nth(median.0).unwrap().1,input.drop_skip().nth(median.0).unwrap().1,0.);
        median = (median.0,median.1,(input.vector[median.0].3 + input.vector[median.0].3)/2.);

        (0,0,0.)
    }

    fn empty_zones() -> (LeftZone,MedianZone,RightZone) {

        let median_zone = MedianZone {
            size: 0,
            dead_center: DeadCenter{left:None,right:None},
            left: None,
            right: None,
            index_set: HashSet::new()
        };
        let left_zone = LeftZone {
            size: 0,
            right: None,
            index_set: HashSet::new()
        };
        let right_zone = RightZone{
            size: 0,
            left: None,
            index_set: HashSet::new()
        };
        (left_zone,median_zone,right_zone)
    }

    pub fn initialize(&mut self) {
        let mut dead_center = DeadCenter::center(&self.vector);
        let mut leftward = self.vector.crawl_left(dead_center.left.unwrap().1).cloned();
        let mut rightward = self.vector.crawl_right(dead_center.right.unwrap().1).cloned();

        let mut left = leftward.next().unwrap();
        let mut right = rightward.next().unwrap();

        let median = dead_center.median();

        let mut left_zone = 0;
        let mut median_zone = 0;
        let mut right_zone = 0;

        let mut left_set: HashSet<usize> = HashSet::new();
        let mut middle_set: HashSet<usize> = HashSet::new();
        let mut right_set: HashSet<usize> = HashSet::new();

        let left_object: LeftZone;
        let median_object: MedianZone;
        let right_object: RightZone;

        if left == right {
            middle_set.insert(left.1);
            median_zone = 1;
            median_object = MedianZone{ size:1 ,dead_center:dead_center,left:Some(left),right:Some(right), index_set: middle_set};
        }
        else {
            middle_set.insert(left.1);
            middle_set.insert(right.1);
            // median_zone = 2;
            median_object = MedianZone{ size:2 ,dead_center:dead_center,left:Some(left),right:Some(right), index_set: middle_set};
        }

        for sample in leftward {
            left_set.insert(sample.1);
            left_zone += 1;
        }
        for sample in rightward {
            right_set.insert(sample.1);
            right_zone += 1;
        }

        left_object = LeftZone{size: left_zone, right:self.vector.left(left.1), index_set: left_set};
        right_object = RightZone{size: right_zone, left: self.vector.right(right.1), index_set: right_set};

        self.left_zone = left_object;
        self.median_zone = median_object;
        self.right_zone = right_object;

    }

    pub fn drop(&mut self, target: usize) {
        self.vector.drop(target);
        self.source_vector.drop(target);
        if self.median_zone.index_set.contains(&target) {
            self.median_zone.size -= 1;
            self.median_zone.index_set.remove(&target);
        }
        if self.left_zone.index_set.contains(&target) {
            self.left_zone.size -= 1;
            self.left_zone.index_set.remove(&target);
        }
        if self.right_zone.index_set.contains(&target) {
            self.right_zone.size -= 1;
            self.right_zone.index_set.remove(&target);
        }
    }

    pub fn drop_zeroes(&mut self) {

        let samples_to_drop: Vec<usize> = self.vector.iter_full().filter(|x| x.3 == 0.).map(|x| x.1).collect();

        for sample in samples_to_drop {
            self.drop(sample);
        }
    }

    pub fn pop(&mut self, target: usize) -> (usize,usize,usize,f64,usize) {
        if self.vector.drop_set.contains(&target) {
            return self.vector[target]
        }
        if self.median_zone.index_set.contains(&target) {
            if self.median_zone.left.unwrap().1 == target {
                self.median_zone.left = self.vector.right(target);
            }
            if self.median_zone.right.unwrap().1 == target {
                self.median_zone.right = self.vector.left(target);
            }
            self.median_zone.size -= 1;
            self.median_zone.index_set.remove(&target);
        }
        if self.left_zone.index_set.contains(&target) {
            if self.left_zone.right.unwrap().1 == target {
                self.left_zone.right = self.vector.left(target);
            }
            self.left_zone.size -= 1;
            self.left_zone.index_set.remove(&target);
        }
        if self.right_zone.index_set.contains(&target) {
            if self.right_zone.left.unwrap().1 == target {
                self.right_zone.left = self.vector.right(target);
            }
            self.right_zone.size -= 1;
            self.right_zone.index_set.remove(&target);
        }
        let removed = self.vector.pop(target);
        self.median_zone.dead_center.re_center(&removed, &self.vector);
        self.zone_balance();
        removed
    }

    pub fn median(&self) -> f64 {
        self.median_zone.dead_center.median()
    }

    pub fn set_boundaries(&mut self) {
        while (self.left_zone.size + self.right_zone.size) > (self.median_zone.size - 1) {
            println!("Moving:{:?}",self.expand_by_1());
            println!("Zones: {:?}",self.zones());
        }
    }

    pub fn zone_balance(&mut self) {
        while (self.left_zone.size + self.right_zone.size) < (self.median_zone.size - 2) {
            // println!("Zones: {},{},{},{},{},{}",self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len());
            self.contract_by_1();
            // println!("Zones: {},{},{},{},{},{}",self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len())
        }
        while (self.left_zone.size + self.right_zone.size) >= (self.median_zone.size) {
            // println!("Zones: {},{},{},{},{},{}",self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len());
            self.expand_by_1();
            // println!("Zones: {},{},{},{},{},{}",self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len())
        }
    }

    fn expand_by_1(&mut self) -> (usize,usize) {

        if self.vector.len() < 2 {
            panic!("Asked to expand a singleton zone!");
        }

        let mut moved_index = (0,0);

        let left = self.median_zone.left;
        let right = self.median_zone.right;

        let outer_right = self.right_zone.left;
        let outer_left = self.left_zone.right;

        let median = self.median_zone.dead_center.median();

        if let (Some(x),Some(y)) = (outer_left,outer_right) {
            if (y.3 - median).abs() < (x.3 - median).abs() {
                moved_index.0 = self.median_zone.expand_right(&self.vector);
                moved_index.1 = self.right_zone.contract(&self.vector);
            }
            else {
                moved_index.0 = self.median_zone.expand_left(&self.vector);
                moved_index.1 = self.left_zone.contract(&self.vector);
            }
        }
        else {
            if outer_left.is_none() {
                moved_index.0 = self.median_zone.expand_right(&self.vector);
                moved_index.1 = self.right_zone.contract(&self.vector);
            }
            if outer_right.is_none() {
                moved_index.0 = self.median_zone.expand_left(&self.vector);
                moved_index.1 = self.left_zone.contract(&self.vector);
            }
        }

        moved_index

    }

    fn contract_by_1(&mut self) -> (usize,usize) {

        let mut moved_index = (0,0);

        if self.vector.len() < 2 {
            panic!("Asked to contract a singleton zone!");
        }

        let left = self.median_zone.left.unwrap();
        let right = self.median_zone.right.unwrap();

        let median = self.median_zone.dead_center.median();

        if (left.3 - median).abs() < (right.3 - median).abs() {
            moved_index.0 = self.median_zone.contract_right(&self.vector);
            moved_index.1 = self.right_zone.expand(&self.vector);
        }
        else {
            moved_index.0 = self.median_zone.contract_left(&self.vector);
            moved_index.1 = self.left_zone.expand(&self.vector);
        }

        moved_index

    }

    fn boundaries (&self) -> (usize,f64,usize,f64) {
        (self.median_zone.left.unwrap().1,self.median_zone.left.unwrap().3,self.median_zone.right.unwrap().1,self.median_zone.right.unwrap().3)
    }

    pub fn zones(&self) -> (usize,usize,usize,usize,usize,usize) {
        (self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len())
    }

    pub fn indecies(&self) -> (HashSet<usize>,HashSet<usize>,HashSet<usize>) {
        (self.left_zone.index_set.clone(), self.median_zone.index_set.clone(), self.right_zone.index_set.clone())
    }

    pub fn derive() {

    }

    pub fn index() {

    }

}

#[derive(Debug,Clone)]
pub struct RankVector<U,T> {
    pub vector: RawVector,
    source_vector: RawVector,
    draw_order : Vec<usize>,

    pub left_zone: LeftZone,
    pub median_zone: MedianZone,
    pub right_zone: RightZone,

    length: usize,
    drop: bool,
    num_dropped: usize,

    median: (usize,usize,f64),

    left_boundary: usize,
    right_boundary: usize,

    finger: usize,

    feature_name: U,
    sample_names: Vec<T>

}

impl MedianZone {

    fn expand_left(&mut self, raw_vector: &RawVector) -> usize {
        let left = self.left.unwrap();
        let new_left = raw_vector.left(left.1).unwrap();
        self.size += 1;
        self.index_set.insert(new_left.1);
        self.left = Some(new_left);
        new_left.1
    }

    fn expand_right(&mut self, raw_vector: &RawVector) -> usize {
        let right = self.right.unwrap();
        let new_right = raw_vector.right(right.1).unwrap();
        self.size += 1;
        self.index_set.insert(new_right.1);
        self.right = Some(new_right);
        new_right.1
    }

    fn contract_left(&mut self, raw_vector: &RawVector) -> usize {
        let left = self.left.unwrap();
        let new_left = raw_vector.right(left.1).unwrap();
        self.size -= 1;
        self.index_set.remove(&left.1);
        self.left = Some(new_left);
        left.1
    }
    fn contract_right(&mut self, raw_vector: &RawVector) -> usize {
        let right = self.right.unwrap();
        let new_right = raw_vector.left(right.1).unwrap();
        self.size -= 1;
        self.index_set.remove(&right.1);
        self.right = Some(new_right);
        right.1
    }

    fn mad(&self) -> f64 {
        (self.left.unwrap().3 - self.dead_center.median()).abs().max((self.right.unwrap().3 - self.dead_center.median()).abs())
    }


}

#[derive(Debug,Clone)]
pub struct MedianZone{
    pub size: usize,
    dead_center: DeadCenter,
    left: Option<usize>,
    right: Option<usize>,
    pub index_set: HashSet<usize>

}

impl DeadCenter {
    pub fn center(raw :&RawVector) -> DeadCenter {
        let length = raw.len();

        let mut left_zone= -1i32;
        let mut right_zone = (length+1) as i32;

        let mut left = None;
        let mut right = Some(raw.first());

        for sample in raw.left_to_right() {

            println!("Center debug: {},{}", left_zone, right_zone);

            left = right;
            left_zone += 1;

            if left_zone == right_zone {
                break
            }

            right = Some(sample.clone());
            right_zone -= 1;

            if left_zone == right_zone {
                break
            }

        }

        DeadCenter {
            left: left,
            right: right,
        }
    }

    pub fn re_center(&mut self, removed:&(usize,usize,usize,f64,usize), raw_vector: &RawVector) {

        if raw_vector.len() < 1 {
            panic!("Tried to re-center an empty vector.")
        }

        if self.left == self.right {
            match removed.2.cmp(&self.left.unwrap().2) {
                Ordering::Greater => {
                    self.left = Some(raw_vector[raw_vector.left_ind(self.right.unwrap().1).unwrap()]);
                },
                Ordering::Less => {
                    self.right = Some(raw_vector[raw_vector.right_ind(self.left.unwrap().1).unwrap()]);
                },
                Ordering::Equal => {
                    self.left = Some(raw_vector[removed.0]);
                    self.right = Some(raw_vector[removed.4]);
                }
            }
        }
        else {
            if removed.2 > self.left.unwrap().2 {
                self.right = self.left;
            }
            if removed.2 < self.right.unwrap().2 {
                self.left = self.right;
            }

        }
    }

    pub fn median(&self) -> f64 {
        (self.left.unwrap().3 + self.right.unwrap().3)/2.
    }

}

#[derive(Debug,Clone)]
pub struct DeadCenter {
    left: Option<usize>,
    right: Option<usize>
}

impl LeftZone {
    fn expand(&mut self,raw_vector:&RawVector) -> usize {
        if let Some(right) = self.right {
            let new_right = raw_vector.right(right.1).unwrap();
            self.size += 1;
            self.index_set.insert(new_right.1);
            self.right = Some(new_right);
        }
        else {
            let new_right = raw_vector.first();
            self.size += 1;
            self.index_set.insert(new_right.1);
            self.right = Some(new_right);
        }
        self.right.unwrap().1
    }

    fn contract(&mut self, raw_vector: &RawVector) -> usize {
        let right = self.right.unwrap();
        let new_right = raw_vector.left(right.1);
        println!("Left contract debug");
        println!("{:?},{:?}", right,new_right);
        self.size -= 1;
        self.index_set.remove(&right.1);
        self.right = new_right;
        right.1
    }
}

#[derive(Debug,Clone)]
pub struct LeftZone{
    pub size: usize,
    // left: Option<(usize,usize,usize,f64,usize)>,
    right: Option<usize>,
    pub index_set: HashSet<usize>
}

impl RightZone {
    fn expand(&mut self,raw_vector:&RawVector) -> usize {
        if let Some(left) = self.left {
            let new_left = raw_vector.left(left.1).unwrap();
            self.size += 1;
            self.index_set.insert(new_left.1);
            self.left = Some(new_left);
        }
        else {
            let new_left = raw_vector.last();
            self.size += 1;
            self.left = Some(new_left);
            self.index_set.insert(new_left.1);
        }
        self.left.unwrap().1
    }

    fn contract(&mut self, raw_vector: &RawVector) -> usize {
        let left = self.left.unwrap();
        let new_left = raw_vector.right(left.1);
        self.size -= 1;
        self.index_set.remove(&left.1);
        self.left = new_left;
        left.1
    }
}

#[derive(Debug,Clone)]
pub struct RightZone {
    pub size: usize,
    left: Option<usize>,
    // right: Option<&'a (usize,usize,usize,f64,usize)>,
    pub index_set: HashSet<usize>,
}


impl<'a,U,T> OrderedDraw<'a,U,T> {
    pub fn new(vector : &'a mut RankVector<U,T>) -> OrderedDraw<'a,U,T> {
        OrderedDraw{vector: vector, index:0}
    }
}

impl<'a,U,T> Iterator for OrderedDraw<'a,U,T> {
    type Item = (f64,f64);

    fn next(&mut self) -> Option<(f64,f64)> {
        if self.vector.vector.len() < 2 {
            return None
        }
        let draw = self.vector.draw_order[self.index];
        println!("Popping: {}",draw);
        println!("Remaining samples: {}", self.vector.vector.len());
        println!("Zones: {:?}", self.vector.zones());
        println!("Boundaries: {:?}", self.vector.boundaries());
        println!("Zone boundaries: {:?},{:?}", self.vector.left_zone.right.unwrap(), self.vector.right_zone.left.unwrap());
        println!("Indecies: {:?}", self.vector.indecies());
        println!("{:?}",self.vector.pop(draw));

        println!("{:?}", self.vector.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());

        self.index +=1;

        Some((self.vector.median_zone.dead_center.median(),self.vector.median_zone.mad()))
    }
}

pub struct OrderedDraw<'a,U:'a,T:'a>{
    pub vector: &'a mut RankVector<U,T>,
    index: usize,
}
