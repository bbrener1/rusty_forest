use std;
use std::marker::PhantomData;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::collections::HashMap;

extern crate rand;
use rand::Rng;

use raw_vector::RawVector;

impl<U,T> RankVector<U,T> {

    fn new(in_vec:&Vec<f64>, drop:f64, feature_name: U, sample_names: Vec<T>) -> RankVector<U,T> {

        let vector = RawVector::raw_vector(in_vec);

        let median = RankVector::<U,T>::describe(&vector);

        let left_boundary = 0;
        let right_boundary = 0;


        RankVector{

            length: vector.vector.len(),

            source_vector: vector.clone(),
            vector: vector,

            drop: false,
            num_dropped: 0,

            median: median,

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

    pub fn median() {

    }

    pub fn boundaries() {

    }

    fn expand_by_1() {

    }

    fn contract_by_1() {

    }

    pub fn derive() {

    }

    pub fn index() {

    }

}

pub struct RankVector<U,T> {
    vector: RawVector,
    source_vector: RawVector,

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

}

pub struct MedianZone {
    dead_center: DeadCenter,
    left: usize,
    right: usize,
    size: usize,
}

impl DeadCenter {
    pub fn center(raw :&RawVector) -> DeadCenter {
        let length = raw.len();

        let mut left_zone:i32 = -1;
        let mut right_zone:i32 = (length +1) as i32;

        let mut left = None;
        let mut center = None;
        let mut right = None;

        for sample in raw.left_to_right() {

            println!("Center debug: {},{}", left_zone, right_zone);

            center = right;
            right = Some(sample.clone());
            right_zone -= 1;

            if left_zone == right_zone {
                break
            }

            left = center;
            center = None;
            left_zone += 1;

            if left_zone == right_zone {
                break
            }

        }

        DeadCenter {
            left: left,
            center: center,
            right: right,
        }
    }

    pub fn re_center(&mut self, target: usize, raw_vector: &RawVector) {

        let removed = raw_vector.vector[target].clone();

        if raw_vector.len() < 2 {
            self.center = Some(raw_vector.first().clone());
            self.left = None;
            self.right = None;
            return
        }

        if let Some(center) = self.center {
            match removed.2.cmp(&center.2) {
                Ordering::Greater => {
                    self.right = self.center;
                    self.center = None;
                    self.left = Some(raw_vector[raw_vector.left(self.right.unwrap().1).unwrap()]);
                },
                Ordering::Less => {
                    self.left = self.center;
                    self.center = None;
                    self.right = Some(raw_vector[raw_vector.right(self.left.unwrap().1).unwrap()]);
                },
                Ordering::Equal => {}
            }
        }
        else {
            if removed.2 > self.left.unwrap().2 {
                self.center = self.left;
                self.left = Some(raw_vector[raw_vector.left(self.center.unwrap().1).unwrap()]);
                self.right = Some(raw_vector[raw_vector.right(self.center.unwrap().1).unwrap()]);
            }
            if removed.2 < self.right.unwrap().2 {
                self.center = self.right;
                self.left = Some(raw_vector[raw_vector.left(self.center.unwrap().1).unwrap()]);
                self.right = Some(raw_vector[raw_vector.right(self.center.unwrap().1).unwrap()]);
            }

        }
    }

    pub fn median(&self) -> f64 {
        if self.center.is_some(){
            return self.center.unwrap().3
        }
        else {
            return (self.left.unwrap().3 + self.right.unwrap().3)/2.
        }
    }
}

#[derive(Debug)]
pub struct DeadCenter {
    left: Option<(usize,usize,usize,f64,usize)>,
    center: Option<(usize,usize,usize,f64,usize)>,
    right: Option<(usize,usize,usize,f64,usize)>
}

pub struct LeftZone {
    size: usize,
    left: Option<usize>,
    right: Option<usize>,
}

pub struct RightZone {
    size: usize,
    left: Option<usize>,
    right: Option<usize>
}
