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

        let mut vector = Vec::with_capacity(in_vec.len());
        let mut draw_order = Vec::with_capacity(in_vec.len());

        let mut sorted_invec = in_vec.iter().enumerate().collect::<Vec<(usize,&f64)>>();
        sorted_invec.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));

        for (i,sample) in sorted_invec.iter().enumerate() {

            // eprintln!("{}", i);

            if i == 0 {
                vector.push((sample.0,sample.0,i,*sample.1,sorted_invec[i+1].0));
                draw_order.push(sample.0);
            }
            if i == (sorted_invec.len() - 1) {
                vector.push((sorted_invec[i-1].0,sample.0,i,*sample.1,sample.0));
                draw_order.push(sample.0);
            }
            if {i != 0} && {i < (sorted_invec.len()-1)} {
                vector.push((sorted_invec[i-1].0,sample.0,i,*sample.1,sorted_invec[i+1].0));
                draw_order.push(sample.0);
            }
        }

        let first: usize;
        let last: usize;
        let length: usize;

        if vector.len() > 0 {
            first = vector[0].1;
            last = vector[vector.len()-1].1;
            length = vector.len();
        }
        else {
            first = 0;
            last = 0;
            length = 0;
        }


        vector.sort_by_key(|x| x.1);

        let drop_set = HashSet::new();

        RawVector {
            first: Some(first),
            len: length,
            last: Some(last),
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

        // println!("Pop debug 1: {:?}\t{:?}\t{:?}",self.vector[left.unwrap_or(i)], target, self.vector[right.unwrap_or(i)]);

        if left.is_none() && right.is_none() {
            if self.len() == 1 {
                if let (Some(first),Some(last)) = (self.first,self.last) {
                    if (first == last) && (first == target.1) {
                        self.first = None;
                        self.last = None;
                        self.len = 0;
                        return target
                    }
                }
            }
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

        if self.first.unwrap_or(i) == i {
            self.first = right;
        }
        if self.last.unwrap_or(i) == i {
            self.last = left;
        }

        // println!("Pop debug 2: {:?}\t{:?}\t{:?}",self.vector[left.unwrap_or(i)], target, self.vector[right.unwrap_or(i)]);

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
        // println!("{:?}",self.first);
        RightVectCrawler{index : Some(first), vector : &self.vector}
    }

    pub fn crawl_left(&self, first:usize) -> LeftVectCrawler {
        LeftVectCrawler{index : Some(first), vector : &self.vector}
    }

    pub fn left_to_right(&self) -> RightVectCrawler {
        if let Some(first) = self.first {
            return self.crawl_right(first)
        }
        return RightVectCrawler::empty(&self.vector)

    }

    pub fn right_to_left(&self) -> LeftVectCrawler {
        if let Some(last) = self.last {
            return self.crawl_left(last)
        }
        return LeftVectCrawler::empty(&self.vector)
    }

    pub fn iter_full(&self) -> RawVectIterFull {
        RawVectIterFull::new(&self.vector,&self.draw_order)
    }

    pub fn iter_ordered(&self) -> Vec<f64> {
        self.vector.iter().cloned().map(|x| x.3).collect()
    }

    pub fn drop_skip<'a>(&'a self) -> RawVectDropSkip<'a> {
        RawVectDropSkip::new(self.iter_full(), &self.drop_set)
    }

    pub fn drop_none<'a>(&'a self) -> RawVectDropNone<'a> {
        RawVectDropNone::new(self.iter_full(), &self.drop_set)
    }

    pub fn dropped_draw_order(&self) -> Vec<usize> {
        self.left_to_right().map(|x| x.1).collect()
    }

    pub fn set_draw(& mut self, order: Vec<usize>) {
        self.draw_order = order
    }

    pub fn first(&self) -> Option<(usize,usize,usize,f64,usize)> {
        if let Some(first) = self.first {
            return Some(self.vector[first].clone())
        }
        None
    }

    pub fn last(&self) -> Option<(usize,usize,usize,f64,usize)> {
        if let Some(last) = self.last {
            return Some(self.vector[last].clone())
        }
        None
    }

    pub fn seek(&self, ind: usize) -> Option<(usize,usize,usize,f64,usize)> {
        if ind < self.vector.len() {
            Some(self.vector[ind].clone())
        }
        else {
            None
        }
    }

    // pub fn derive(&self, indecies:Vec<usize>) -> RawVector {
    pub fn derive(&self, indecies:&[usize]) -> RawVector {

        let derived_set: HashSet<usize> = indecies.iter().cloned().collect();
        let index_map: HashMap<usize,usize> = self.vector.iter().map(|x| x.1).filter(|x| derived_set.contains(x)).enumerate().map(|x| (x.1,x.0)).collect();

        let mut intermediate = vec![(0,0,0,0.,0);derived_set.len()];
        let mut new_draw_order = Vec::new();

        let mut i = 0;
        let mut previous = 0;
        let mut first = 0;
        let mut new_index = 0;


        for sample in self.iter_full() {
            if derived_set.contains(&sample.1) {

                let new: (usize,usize,usize,f64,usize);

                new_index = index_map[&sample.1];

                if i == 0 {
                    new = (new_index,new_index,i,sample.3,new_index);
                    previous = new_index;
                    first = new_index;
                }
                else {
                    new = (previous,new_index,i,sample.3,new_index);
                    intermediate[previous].4 = new_index;
                    previous = new_index;
                }


                intermediate[index_map[&sample.1]] = new;
                new_draw_order.push(index_map[&sample.1]);

                i += 1;
            }
        }

        let last = new_index;

        let mut new_drop_set = HashSet::new();

        // if self.drop {
        //     new_drop_set = derived_set.intersection(&self.drop_set).cloned().collect();
        // }

        let new_raw = RawVector {
            first: Some(first),
            len: intermediate.len(),
            last: Some(last),
            vector: intermediate,
            draw_order: new_draw_order,
            drop_set: new_drop_set,
            drop: false
        };

        new_raw

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
    pub first: Option<usize>,
    pub last: Option<usize>,
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
    pub fn empty(input: &'a Vec<(usize,usize,usize,f64,usize)>) -> RightVectCrawler {
        RightVectCrawler{vector: input, index: None}
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
    pub fn empty(input: &'a Vec<(usize,usize,usize,f64,usize)>) -> LeftVectCrawler {
        LeftVectCrawler{vector: input, index: None}
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
