use std;
use std::marker::PhantomData;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::collections::HashMap;
use std::ops::Index;

extern crate rand;
use rand::Rng;

impl RawVector {
    pub fn raw_vector(in_vec:&Vec<f64>) -> RawVector {

        let mut vector = Vec::new();

        let mut sorted_invec = in_vec.iter().enumerate().collect::<Vec<(usize,&f64)>>();
        sorted_invec.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));

        for (i,sample) in sorted_invec.iter().enumerate() {

            // eprintln!("{}", i);

            if i == 0 {
                vector.push((sample.0,sample.0,i,*sample.1,sorted_invec[i+1].0));
            }
            if i == (sorted_invec.len() - 1) {
                vector.push((sorted_invec[i-1].0,sample.0,i,*sample.1,sample.0));
            }
            if {i != 0} && {i < (sorted_invec.len()-1)} {
                vector.push((sorted_invec[i-1].0,sample.0,i,*sample.1,sorted_invec[i+1].0));
            }
        }

        let first = vector[0].1;
        let last = vector[vector.len()-1].1;
        let length = vector.len();

        vector.sort_by_key(|x| x.1);

        let draw_order = (0..vector.len()).collect();
        let drop_set = HashSet::new();

        RawVector {
            first: first,
            len: length,
            last: last,
            vector: vector,
            draw_order: draw_order,
            drop_set: drop_set,
            drop: false
        }

    }

    pub fn pop(&mut self, i:usize) -> (usize,usize,usize,f64,usize) {
        let left = self.left_ind(i);
        let right = self.right_ind(i);

        let target = self.vector[i].clone();

        println!("Pop debug 1: {:?}\t{:?}\t{:?}",self.vector[left.unwrap_or(i)], target, self.vector[right.unwrap_or(i)]);

        if left.is_none() && right.is_none() {
            return target
        }

        self.vector[i] = (i,i,target.2,target.3,i);

        match left {
            Some(x) => self.vector[x].4 = right.unwrap_or(x),
            None => {}
        }
        match right {
            Some(x) => self.vector[x].0 = left.unwrap_or(x),
            None => {}
        }

        self.len -= 1;

        if self.first == i {
            self.first = right.unwrap();
        }
        if self.last == i {
            self.last = left.unwrap();
        }

        println!("Pop debug 2: {:?}\t{:?}\t{:?}",self.vector[left.unwrap_or(i)], target, self.vector[right.unwrap_or(i)]);

        target
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn left(&self, i:usize) -> Option<(usize,usize,usize,f64,usize)> {
        let sample = self.vector[i];
        if sample.0 != sample.1 {
            Some(self[sample.0])
        }
        else {
            None
        }
    }

    pub fn right(&self, i:usize) -> Option<(usize,usize,usize,f64,usize)> {
        let sample = self.vector[i];
        if sample.4 != sample.1 {
            Some(self[sample.4])
        }
        else {
            None
        }
    }

    pub fn left_ind(&self, i:usize) -> Option<usize> {
        let sample = self.vector[i];
        if sample.0 != sample.1 {
            Some(sample.0)
        }
        else {
            None
        }
    }

    pub fn right_ind(&self, i:usize) -> Option<usize> {
        let sample = self.vector[i];
        if sample.4 != sample.1 {
            Some(sample.4)
        }
        else {
            None
        }
    }

    pub fn drop(&mut self, target: usize) {
        self.drop_set.insert(target);
        self.pop(target);
        self.drop = true;
    }

    pub fn drop_zeroes(&mut self) {
        self.drop_set = self.iter_full().filter(|x| x.3 == 0.).map(|x| x.1).collect();
        for i in self.drop_set.clone() {
            self.pop(i);
        }
        self.drop = true;
        eprintln!("Drop set");
        eprintln!("{:?}", self.drop_set);
        eprintln!("{:?}", self.vector)
    }

    pub fn iter(&self) -> RawVectDropNone {
        self.drop_none()
    }

    pub fn crawl_right(&self,first:usize) -> RightVectCrawler {
        println!("{:?}",self.first);
        RightVectCrawler{index : Some(first), vector : &self.vector}
    }

    pub fn crawl_left(&self, first:usize) -> LeftVectCrawler {
        LeftVectCrawler{index : Some(first), vector : &self.vector}
    }

    pub fn left_to_right(&self) -> RightVectCrawler {
        self.crawl_right(self.first)
    }

    pub fn right_to_left(&self) -> LeftVectCrawler {
        self.crawl_left(self.last)
    }

    pub fn iter_full(&self) -> RawVectIterFull {
        RawVectIterFull::new(&self.vector,&self.draw_order)
    }

    pub fn drop_skip<'a>(&'a self) -> RawVectDropSkip<'a> {
        RawVectDropSkip::new(self.iter_full(), &self.drop_set)
    }

    pub fn drop_none<'a>(&'a self) -> RawVectDropNone<'a> {
        RawVectDropNone::new(self.iter_full(), &self.drop_set)
    }

    pub fn set_draw(& mut self, order: Vec<usize>) {
        self.draw_order = order
    }

    pub fn first(&self) -> (usize,usize,usize,f64,usize) {
        self.vector[self.first].clone()
    }

    pub fn last(&self) -> (usize,usize,usize,f64,usize) {
        self.vector[self.last].clone()
    }

    pub fn seek(&self, ind: usize) -> Option<(usize,usize,usize,f64,usize)> {
        if ind < self.vector.len() {
            Some(self.vector[ind].clone())
        }
        else {
            None
        }
    }

    // pub fn drop_skip<'a,'b>(&'a self, drop_set: &'b HashSet<usize>) -> RawVectDropSkip<'a,'b> {
    //     RawVectDropSkip::new(&self.vector, &drop_set)
    // }
    //
    // pub fn none_skip<'a,'b>(&'a self, drop_set: &'b HashSet<usize>) -> RawVectDropNone<'a,'b> {
    //     RawVectDropNone::new(&self.vector, &drop_set)
    // }
}

impl Index<usize> for RawVector {
    type Output = (usize,usize,usize,f64,usize);

    fn index(& self, ind: usize) -> &(usize,usize,usize,f64,usize) {
        &self.vector[ind]
    }
}

////////////////////////
// Iterator based on a set draw order from the raw vector

#[derive(Debug,Clone)]
pub struct RawVector {
    pub vector: Vec<(usize,usize,usize,f64,usize)>,
    pub first: usize,
    pub last: usize,
    pub len: usize,
    pub drop_set : HashSet<usize>,
    pub draw_order : Vec<usize>,
    pub drop: bool
}

////////////////////////
// Meta-iterator wrap, decides whether to drop, skip, or none


////////////////////////
// Iterator that draws in order


impl<'a> RawVectIterFull<'a> {
    fn new(input: &'a Vec<(usize,usize,usize,f64,usize)>, draw_order: &'a Vec<usize>) -> RawVectIterFull<'a> {
        RawVectIterFull{vector: input, index:0,draw_order:draw_order}
    }
}

impl<'a> Iterator for RawVectIterFull<'a> {
    type Item = &'a (usize,usize,usize,f64,usize);

    fn next(&mut self) -> Option<&'a (usize,usize,usize,f64,usize)> {

        self.index += 1;
        if self.index <= self.vector.len() {
            Some(& self.vector[self.draw_order[self.index-1]])
        }
        else {
            None
        }
    }
}


pub struct RawVectIterFull<'a> {
    vector: &'a Vec<(usize,usize,usize,f64,usize)>,
    index: usize,
    draw_order: &'a Vec<usize>
}


////////////////////////
// Iterator that skips items that are present in the drop set

impl<'a> RawVectDropSkip<'a> {
    fn new(input: RawVectIterFull<'a>, drop_index : &'a HashSet<usize>) -> RawVectDropSkip<'a> {
        RawVectDropSkip{draw: input, index:0, drop_set: drop_index}
    }
}

impl<'a> Iterator for RawVectDropSkip<'a> {
    type Item = &'a (usize,usize,usize,f64,usize);

    fn next(&mut self) -> Option<&'a (usize,usize,usize,f64,usize)> {

        let mut result = self.draw.next();
        loop {
            if result.is_none() {
                return None
            }
            if !self.drop_set.contains(&result.unwrap().1) {
                break
            }
            else {
                result = self.draw.next();
            }
        }
        result
    }
}


pub struct RawVectDropSkip<'a> {
    draw: RawVectIterFull<'a>,
    drop_set: &'a HashSet<usize>,
    index: usize
}
//
////////////////////////

////////////////////////
// Iterates through ranked vector, replaces dropped values with None, ordinary values are Some

impl<'a> RawVectDropNone<'a> {
    fn new(input: RawVectIterFull<'a>, drop_index : &'a HashSet<usize>) -> RawVectDropNone<'a> {
        RawVectDropNone{draw: input, index:0, drop_set: drop_index}
    }
}

impl<'a> Iterator for RawVectDropNone<'a> {
    type Item = (usize,usize,usize,Option<f64>,usize);

    fn next(&mut self) -> Option<(usize,usize,usize,Option<f64>,usize)> {

        let mut result = self.draw.next();

        if result.is_none() {
            return None
        }

        let output = result.unwrap().clone();

        if self.drop_set.contains(&output.1) {
            Some((output.0,output.1,output.2,None,output.4))
        }
        else {
            Some((output.0,output.1,output.2,Some(output.3),output.4))
        }

    }
}


pub struct RawVectDropNone<'a> {
    draw: RawVectIterFull<'a>,
    drop_set: &'a HashSet<usize>,
    index: usize
}


////////////////////////
// Crawls right from a given node

impl<'a> RightVectCrawler<'a> {
    fn new(input: &'a Vec<(usize,usize,usize,f64,usize)>, first: usize) -> RightVectCrawler {
        RightVectCrawler{vector: input, index: Some(first)}
    }
}

impl<'a> Iterator for RightVectCrawler<'a> {
    type Item = &'a (usize,usize,usize,f64,usize);

    fn next(&mut self) -> Option<&'a (usize,usize,usize,f64,usize)> {

        let index: usize;
        if self.index.is_some(){
            index = self.index.unwrap();
        }
        else{
            return None
        }

        if self.vector[index].1 == self.vector[index].4 {
            self.index = None;
            return Some(& self.vector[index])
        }
        let current = & self.vector[index];
        self.index = Some(current.4);
        Some(& current)

    }
}

pub struct RightVectCrawler<'a> {
    vector: &'a Vec<(usize,usize,usize,f64,usize)>,
    index: Option<usize>,
}

////////////////////////
// Crawls left from a given node

impl<'a> LeftVectCrawler<'a> {
    fn new(input: &'a Vec<(usize,usize,usize,f64,usize)>, first: usize) -> LeftVectCrawler {
        LeftVectCrawler{vector: input, index: Some(first)}
    }
}

impl<'a> Iterator for LeftVectCrawler<'a> {
    type Item = &'a (usize,usize,usize,f64,usize);

    fn next(&mut self) -> Option<&'a (usize,usize,usize,f64,usize)> {

        let index: usize;
        if self.index.is_some(){
            index = self.index.unwrap();
        }
        else{
            return None
        }

        if self.vector[index].1 == self.vector[index].0 {
            self.index = None;
            return Some(& self.vector[index])
        }
        let current = & self.vector[index];
        self.index = Some(current.0);
        Some(& current)

    }
}

pub struct LeftVectCrawler<'a> {
    vector: &'a Vec<(usize,usize,usize,f64,usize)>,
    index: Option<usize>
}



// impl<'a> Iterator for GenericIter<'a> {
//     type Item = &'a tuple;
//
//     fn next(&mut self) -> &'a T {
//         if self.none.is_some()
//     }
// }
//
// pub struct GenericIter<'a> {
//
//     full: Option<RawVectIterFull<'a>>,
//     skip: Option<RawVectDropSkip<'a>>,
//     none: Option<RawVectDropNone<'a>>,
//
// }
