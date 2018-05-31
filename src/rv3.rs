use std::f64;
use std::collections::HashSet;
use std::collections::HashMap;
use std::cmp::Ordering;
use smallvec::SmallVec;
use std::ops::Index;
use std::ops::IndexMut;
use std::fmt::Debug;
use std::clone::Clone;
use std::borrow::{Borrow,BorrowMut};
use DropMode;

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct RankVector<T> {
    drop_set: Option<HashSet<usize>>,
    dirty_set: Option<HashSet<usize>>,
    rank_order: Option<Vec<usize>>,
    drop: DropMode,
    zones: [usize;4],
    offset: usize,
    median: (usize,usize),
    left: usize,
    right: usize,
    nodes: T,
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Node {
    data: f64,
    index: usize,
    rank: usize,
    previous: usize,
    next: usize,
    zone: usize,
}

impl Node {
    pub fn blank() -> Node{
        Node {
            data: f64::NAN,
            index: 0,
            rank: 0,
            previous: 0,
            next: 0,
            zone: 0,
        }
    }
}

impl<T: Borrow<[Node]> + BorrowMut<[Node]> + Index<usize,Output=Node> + IndexMut<usize,Output=Node> + Clone + Debug > RankVector<T> {

    pub fn empty() -> RankVector<Vec<Node>> {
        RankVector::<Vec<Node>>::link(&vec![])
    }

    pub fn link(in_vec: &Vec<f64>) -> RankVector<Vec<Node>> {

        let mut vector: Vec<Node> = vec![Node::blank();in_vec.len()+2];

        let left = vector.len() - 2;
        let right = vector.len() -1;

        vector[left] = Node {
            data:0.,
            index:left,
            rank:0,
            previous:left,
            next: right,
            zone:0,
        };

        vector[right] = Node {
            data:0.,
            index:right,
            rank:0,
            previous:left,
            next:right,
            zone:0,
        };

        let mut zones = [0;4];

        let (clean_vector,dirty_set) = sanitize_vector(in_vec);

        let mut sorted_invec = clean_vector.into_iter().enumerate().collect::<Vec<(usize,f64)>>();

        sorted_invec.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));

        let mut rank_order = Vec::with_capacity(sorted_invec.len());

        let mut previous = left;

        let tail_node_index = right;

        for (ranking,(index,data)) in sorted_invec.into_iter().enumerate() {

            let node = Node {
                data: data,
                index: index,
                previous: vector[previous].index,
                next: tail_node_index,
                zone: 2,
                rank: ranking,
            };

            // println!("{:?}", vector);

            vector[previous].next = index;

            previous = index;

            rank_order.push(index);

            vector[index] = node;

            zones[2] += 1;

        };

        vector[right].previous = previous;

        let median = (vector.len()-2,vector.len()-2);

        let left = *rank_order.get(0).unwrap_or(&0);
        let right = *rank_order.last().unwrap_or(&0);

        let drop_set = HashSet::with_capacity(0);

        let mut prototype = RankVector::<Vec<Node>> {
            nodes: vector,
            drop_set: Some(drop_set),
            dirty_set: Some(dirty_set),
            rank_order: Some(rank_order),
            drop: DropMode::No,
            zones: zones,
            offset: 2,
            median: median,
            left: left,
            right: right,
        };

        // println!("Linking");

        // println!("{:?}", prototype);

        prototype.establish_median();

        // println!("{:?}", prototype);

        prototype.establish_zones();

        // println!("{:?}", prototype);

        prototype

    }


    #[inline]
    pub fn g_left(&self,index:usize) -> usize {
        self.nodes[index].previous
    }

    #[inline]
    pub fn g_right(&self, index:usize) -> usize {
        self.nodes[index].next
    }

    pub fn pop(&mut self, target: usize) -> f64 {

        println!("{:?}", self);

        let target_zone = self.nodes[target].zone;
        //
        println!("Popping {}", target);

        if target_zone != 0 {
            //
            // println!("Popping internal");
            //
            self.unlink(target);

            self.zones[target_zone] -= 1;
            self.zones[0] += 1;
            self.nodes[target].zone = 0;

            self.check_boundaries(target);

            // println!("Balancing");
            // println!("{:?}", self.zones);
            // println!("{:?}", self.median);

            self.balance_zones(target);

            // println!("{:?}", self.zones);
            // println!("{:?}", self.median);
            //
            // println!("Recentering");
            // println!("{:?}", self.median);
            let (old_median,new_median) = self.recenter_median(target);
            // println!("{:?}", self.median);


            // println!("Shifting zones");
            //
            self.shift_zones(old_median,new_median);

        }

        // if (self.median() - slow_median(self.ordered_values())).abs() > 0.00001 {
        //     println!("{:?}", self);
        //     println!("{:?}", self.ordered_values());
        //     println!("{:?}", slow_median(self.ordered_values()));
        //     panic!("Failed to adjust median!");
        // };
        //
        // if (self.mad() - slow_mad(self.ordered_values())).abs() > 0.00001 {
        //     println!("{:?}", self);
        //     println!("{:?}", self.ordered_values());
        //     println!("{:?}", self.mad());
        //     println!("{:?}", slow_mad(self.ordered_values()));
        //     panic!("Failed to adjust mad");
        // };

        println!("{:?}", self);

        self.nodes[target].data

    }

    // This method acts directly on the internal linked list, bypassing the node at a given index.

    // #[inline]
    // fn pop_internal(&mut self, target: usize) -> &Node {
    //
    //     let left = self.nodes[target].previous;
    //     let right = self.nodes[target].next;
    //
    //     self.nodes[left].next = self.nodes[target].next;
    //     self.nodes[right].previous = self.nodes[target].previous;
    //
    //     self.zones[self.nodes[target].zone].length -= 1;
    //     self.zones[0].length += 1;
    //
    //     self.nodes[target].zone = 0;
    //
    //     &self.nodes[target]
    // }

    fn unlink(&mut self, target: usize) {

        let left = self.nodes[target].previous;
        let right = self.nodes[target].next;

        self.nodes[left].next = self.nodes[target].next;
        self.nodes[right].previous = self.nodes[target].previous;

    }

    fn check_boundaries(&mut self, target: usize) {
        match target {
            left if left == self.left => {
                self.left = self.nodes[target].next;
            },
            right if right == self.right => {
                self.right = self.nodes[target].previous;
            },
            _ => {},
        }
    }

    //

    #[inline]
    pub fn establish_median(&mut self) {

        let order = self.left_to_right();

        match order.len() % 2 {
            0 => {
                if order.len() == 0 {
                    self.median = (0,1)
                }
                else {
                    self.median = (order[(order.len()/2)-1],order[order.len()/2]);
                }
            },
            1 => {
                self.median = (order[order.len()/2],order[order.len()/2]);
            },
            _ => unreachable!(),
        }

        // if (self.median() - slow_median(self.ordered_values())) > 0.00001 {
        //     println!("{:?}", self);
        //     println!("{:?}", self.ordered_values());
        //     println!("{:?}", slow_median(self.ordered_values()));
        //     panic!("Failed to establish median!");
        // }
    }

    #[inline]
    pub fn establish_zones(&mut self) {

        // for _ in 0..((self.len() - self.len()%2)/2) {
        for _ in 0..(((self.len())/2).max(1) - (1 - self.len()%2)) {
            self.contract_1();
        };

        // if (self.mad() - slow_mad(self.ordered_values())) > 0.00001 {
        //     println!("{:?}", self);
        //     println!("{:?}", self.ordered_values());
        //     println!("{:?}", self.mad());
        //     println!("{:?}", slow_mad(self.ordered_values()));
        //     panic!("Failed to establish mad");
        // };

    }

    #[inline]
    pub fn len(&self) -> usize {
        self.zones[1] + self.zones[2] + self.zones[3]
    }

    #[inline]
    pub fn raw_len(&self) -> usize {
        self.zones[0] + self.zones[1] + self.zones[2] + self.zones[3]
    }

    #[inline]
    pub fn drop_f(&mut self, f: f64) {

        let mut drop_set: HashSet<usize> = self.left_to_right().iter().map(|x| &self.nodes[*x]).filter(|y| y.data == f).map(|x| x.index).collect();

        for index in drop_set.iter() {
            self.pop(*index);
        };
        drop_set.shrink_to_fit();
        self.drop_set.as_mut().unwrap().extend(drop_set.iter());

    }

    #[inline]
    pub fn contract_left(&mut self) {
        self.zones[1] += 1;
        self.zones[2] -= 1;

        self.nodes[self.left].zone = 1;
        self.left = self.nodes[self.left].next;
    }

    #[inline]
    pub fn contract_right(&mut self) {
        self.zones[3] += 1;
        self.zones[2] -= 1;

        self.nodes[self.right].zone = 3;
        self.right = self.nodes[self.right].previous;
    }


    #[inline]
    pub fn expand_left(&mut self) {
        self.zones[1] -= 1;
        self.zones[2] += 1;

        self.left = self.nodes[self.left].previous;
        self.nodes[self.left].zone = 2;
    }

    #[inline]
    pub fn expand_right(&mut self) {
        self.zones[3] -= 1;
        self.zones[2] += 1;

        self.right = self.nodes[self.right].next;
        self.nodes[self.right].zone = 2;
    }

    #[inline]
    pub fn move_left(&mut self) {
        self.expand_left();
        self.contract_right();
    }

    #[inline]
    pub fn move_right(&mut self) {
        self.expand_right();
        self.contract_left();
    }

    #[inline]
    pub fn expand_1(&mut self) {

        let median = self.median();

        if self.zones[1] > 0 && self.zones[3] > 0 {

            let left = self.nodes[self.left].data;
            let right = self.nodes[self.right].data;

            if (right - median).abs() > (median - left).abs() {
                self.expand_left();
            }
            else {
                self.expand_right();
            }

        }
        else {
            if self.zones[1] == 0 {
                self.expand_right();
            }
            else if self.zones[3] == 0 {
                self.expand_left();
            }
            else {
                panic!("Tried to expand into empty boundary zones!")
            }
        }

    }


    #[inline]
    pub fn contract_1(&mut self) {

        let median = self.median();

        let left = self.nodes[self.left].data;
        let right = self.nodes[self.right].data;

        // println!("{:?}", self);
        //
        // println!("{},{},{}", left, median, right);
        //
        // println!("Comparison {},{}", left-median, right-median);

        if (right - median).abs() > (left - median).abs() {
            // println!("Right");
            self.contract_right();
        }
        else {
            // println!("Left");
            self.contract_left();
        }
    }

    #[inline]
    pub fn balance_zones(&mut self,target:usize) {

        if self.len() > 0 {

            match self.len() %2 {
                1 => {
                    match self.zones[2].cmp(&(self.zones[1] + self.zones[3] + 1)) {
                        Ordering::Greater => self.contract_1(),
                        Ordering::Less => self.expand_1(),
                        Ordering::Equal => {},
                    }
                },
                0 => {
                    match self.zones[2].cmp(&(self.zones[1] + self.zones[3] + 2)) {
                        Ordering::Greater => self.contract_1(),
                        Ordering::Less => self.expand_1(),
                        Ordering::Equal => {},
                    }
                }
                _ => unreachable!(),
            }

        }

    }

    #[inline]
    pub fn median(&self) -> f64 {

        (self.nodes[self.median.0].data + self.nodes[self.median.1].data) / 2.

    }

    #[inline]
    pub fn shift_median_left(&mut self) {
        match self.median.0 == self.median.1 {
            false => {
                self.median = (self.nodes[self.median.1].previous,self.nodes[self.median.1].previous)
            },
            true => {
                self.median = (self.nodes[self.median.1].previous,self.median.1)
            }
        }
    }

    #[inline]
    pub fn shift_median_right(&mut self) {
        match self.median.0 == self.median.1 {
            false => {
                self.median = (self.nodes[self.median.0].next,self.nodes[self.median.0].next)
            },
            true => {
                self.median = (self.median.0,self.nodes[self.median.0].next)
            }
        }
    }

    #[inline]
    pub fn recenter_median(&mut self, target:usize) -> (f64,f64) {

        let old_median = self.median();

        // println!("Recentering");
        // println!("{:?}", self.nodes[self.median.0]);
        // println!("{:?}", self.nodes[self.median.1]);

        let target_rank = self.nodes[target].rank;
        let left_rank = self.nodes[self.median.0].rank;
        let right_rank = self.nodes[self.median.1].rank;

        if target_rank > left_rank {
                self.shift_median_left();
        }
        else if target_rank < right_rank {
                self.shift_median_right();
        }
        else {
            self.median.0 = self.nodes[target].previous;
            self.median.1 = self.nodes[target].next;
        }

        let new_median = self.median();

        (old_median,new_median)

    }

    #[inline]
    pub fn shift_zones(&mut self,old_median:f64,new_median:f64) {

        let change = new_median - old_median;


        if change > 0. {

            for i in 0..self.zones[3] {

                let left = self.nodes[self.left].data;
                let right = self.nodes[self.nodes[self.right].next].data;

                // println!("Moving right");
                // println!("{},{},{}",left,new_median,right);
                // println!("Comparison: {},{}",(left - new_median).abs(), (right - new_median).abs());

                if (right - new_median).abs() > (left - new_median).abs() {
                    // println!("Finished");
                    break
                }

                self.move_right()

            }
        }
        if change < 0. {

            for i in 0..self.zones[1] {

                let left = self.nodes[self.nodes[self.left].previous].data;
                let right = self.nodes[self.right].data;

                // println!("Moving left");
                // println!("{},{},{}",left,new_median,right);
                // println!("Comparison: {},{}",(left - new_median).abs(), (right - new_median).abs());

                if (left - new_median).abs() > (right - new_median).abs() {
                    // println!("Finished");
                    break
                }

                self.move_left()

            }
        }

    }


    #[inline]
    pub fn mad(&self) -> f64 {

        if self.len() < 2 {return 0.}

        let left_i = self.left;
        let right_i = self.right;

        let inner_left_i = self.nodes[left_i].next;
        let inner_right_i = self.nodes[right_i].previous;

        let left = self.nodes[left_i].data;
        let right = self.nodes[right_i].data;
        let inner_left = self.nodes[inner_left_i].data;
        let inner_right = self.nodes[inner_right_i].data;

        let median = self.median();

        let mut distance_to_median = [(left - median).abs(), (inner_left - median).abs(), (inner_right - median).abs(), (right - median).abs()];

        println!("MAD debug");
        println!("{:?}", self);
        // println!("{}",median);
        // println!("{:?},{:?}",left,right);
        // println!("{:?}",distance_to_median);

        distance_to_median.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));
        distance_to_median.reverse();

        if self.len() % 2 == 1 {
            return distance_to_median[0]
        }
        else {
            return (distance_to_median[0] + distance_to_median[1]) / 2.
        }
    }

    #[inline]
    pub fn left_to_right(&self) -> Vec<usize> {
        GRVCrawler::new(self, self.nodes[self.raw_len()].next).take(self.len()).collect()
    }

    #[inline]
    pub fn ordered_values(&self) -> Vec<f64> {
        // println!("Ordered values");
        // println!("{:?}", self.left_to_right());
        self.left_to_right().iter().map(|x| self.nodes[*x].data).collect()
    }

    pub fn full_values(&self) -> Vec<f64> {
        (0..self.raw_len()).map(|x| self.nodes[x].data).collect()
    }

    pub fn ordered_meds_mads(&mut self,draw_order: &Vec<usize>,drop_set: HashSet<usize>) -> Vec<(f64,f64)> {

        for dropped_sample in drop_set {
            self.pop(dropped_sample);
        }

        let mut meds_mads = Vec::with_capacity(draw_order.len());
        meds_mads.push((self.median(),self.mad()));
        for draw in draw_order {
            self.pop(*draw);
            meds_mads.push((self.median(),self.mad()))
        }

        meds_mads
    }

    pub fn ordered_mad_gains(&mut self,draw_order: &Vec<usize>, drop_set: &HashSet<usize>) -> Vec<f64> {

        for dropped_sample in drop_set {
            self.pop(*dropped_sample);
        }

        let start_mad = self.mad();

        let mut mad_gains = Vec::with_capacity(draw_order.len());

        mad_gains.push(0.);
        for draw in draw_order {
            self.pop(*draw);
            mad_gains.push((start_mad - self.mad()).max(0.));
        }

        mad_gains

    }

    pub fn ordered_mads(&mut self,draw_order: &Vec<usize>,drop_set: &HashSet<usize>) -> Vec<f64> {

        for dropped_sample in drop_set {
            self.pop(*dropped_sample);
        }

        let mut mads = Vec::with_capacity(draw_order.len());
        mads.push(self.mad());
        for draw in draw_order {
            self.pop(*draw);
            mads.push(self.mad());
            // if (self.mad() - slow_mad(self.ordered_values())).abs() > 0.00001 {
            //     println!("{:?}", self);
            //     println!("{:?}", self.ordered_values());
            //     println!("{:?}", self.mad());
            //     println!("{:?}", slow_mad(self.ordered_values()));
            //     panic!("Mad mismatch");
            // }
        }

        mads
    }

    pub fn ordered_covs(&mut self,draw_order: &Vec<usize>,drop_set: &HashSet<usize>) -> Vec<f64> {

        for dropped_sample in drop_set {
            self.pop(*dropped_sample);
        }

        let mut covs = Vec::with_capacity(draw_order.len());

        covs.push(self.mad()/self.median());

        for draw in draw_order {
            self.pop(*draw);
            let mut cov = (self.mad()/self.median()).abs();
            covs.push(cov);
        }

        // println!("innermost pre-filter: {:?}", covs);

        for element in covs.iter_mut() {
            if !element.is_normal() {
                *element = 0.;
            }
        }

        // println!("innermost: {:?}", covs);

        covs
    }

    #[inline]
    pub fn draw_order(&self) -> Vec<usize> {
        self.left_to_right()
    }

    #[inline]
    pub fn draw_and_drop(&self) -> (Vec<usize>,HashSet<usize>) {
        let draw_order = self.left_to_right();
        let drop_set = self.drop_set.as_ref().unwrap_or(&HashSet::with_capacity(0)).clone();
        (draw_order,drop_set)
    }

    #[inline]
    pub fn split_indecies(&self, split:&f64) -> (Vec<usize>,Vec<usize>) {

        let (mut left,mut right) = (Vec::with_capacity(self.len()),Vec::with_capacity(self.len()));

        for (i,sample) in self.ordered_values().iter().enumerate() {
            if sample <= &split {
                left.push(i);
            }
            else {
                right.push(i);
            }
        }

        left.shrink_to_fit();
        right.shrink_to_fit();

        (vec![],vec![])
    }

    pub fn ordered_cov_gains(&mut self,draw_order: &Vec<usize>,drop_set: &HashSet<usize>) -> Vec<f64> {

        for dropped_sample in drop_set {
            self.pop(*dropped_sample);
        }

        let mut cov_gains = Vec::with_capacity(draw_order.len());

        let mut start_cov = self.mad()/self.median();

        if !start_cov.is_normal() {
            start_cov = 0.;
        }

        cov_gains.push(0.);

        for draw in draw_order {

            let mut cov = self.mad()/self.median();

            if !cov.is_normal() {
                cov = 0.;
            }

            self.pop(*draw);
            cov_gains.push(start_cov - cov);
        }

        cov_gains
    }

    #[inline]
    pub fn fetch(&self, index:usize) -> f64 {
        self.nodes[index].data
    }

    #[inline]
    pub fn boundaries(&self) -> ((usize,f64),(usize,f64)) {
        ((self.left,self.nodes[self.left].data),(self.right,self.nodes[self.right].data))
    }

    #[inline]
    pub fn drop_using_mode(&mut self,drop_mode: DropMode) {
        let cmp = drop_mode.cmp();
        self.drop_f(cmp);
        self.drop = drop_mode;
    }

}

impl RankVector<Vec<Node>> {

    #[inline]
    pub fn derive(&self, indecies:&[usize]) -> RankVector<Vec<Node>> {

        let mut new_nodes: Vec<Node> = vec![Node::blank();indecies.len() + self.offset];
        let mut index_map = HashMap::with_capacity(indecies.len());
        let mut new_rank_order = Vec::with_capacity(indecies.len());
        let mut derived_rank_order = Vec::with_capacity(indecies.len());

        let left = new_nodes.len() - 2;
        let right = new_nodes.len() -1;

        new_nodes[left] = Node {
            data:0.,
            index:left,
            rank:0,
            previous:left,
            next:right,
            zone:0,
        };

        new_nodes[right] = Node {
            data:0.,
            index:right,
            rank:0,
            previous:left,
            next:right,
            zone:0,
        };

        let mut new_zones = [0;4];

        let derived_set: HashSet<usize> = indecies.iter().cloned().collect();

        index_map.extend(
                (0..self.raw_len())
                .filter(|x| derived_set.contains(x))
                .enumerate()
                .map(|(i,x)| (x,i))
            );

        for n in self.rank_order.as_ref().unwrap() {
            if derived_set.contains(&n) {
                derived_rank_order.push(*n);
                new_rank_order.push(index_map[n]);
            }
        }

        let new_dirty_set: HashSet<usize> = self.dirty_set.as_ref().unwrap().iter().filter(|x| derived_set.contains(x)).map(|y| index_map[y]).collect();

        let mut previous = left;

        for (i,(&old_index,&new_index)) in derived_rank_order.iter().zip(new_rank_order.iter()).enumerate() {

            let data = self.nodes[old_index].data;

            let new_node = Node {
                data: data,
                index: new_index,
                rank: i,
                previous: previous,
                next: right,
                zone: 2,
            };

            new_nodes[previous].next = new_index;
            new_nodes[new_index] = new_node;
            new_zones[2] += 1;

            previous = new_index;

        }

        new_nodes[right].previous = previous;

        let left = *new_rank_order.get(0).unwrap_or(&0);
        let right = *new_rank_order.last().unwrap_or(&0);

        let mut new_vector = RankVector {
            drop_set: Some(HashSet::with_capacity(0)),
            dirty_set: Some(new_dirty_set),
            rank_order: Some(new_rank_order),
            drop: self.drop,
            zones: new_zones,
            offset: self.offset,
            median: (4,4),
            nodes: new_nodes,
            left: left,
            right: right,

        };

        new_vector.establish_median();
        new_vector.establish_zones();

        new_vector.drop_using_mode(self.drop);

        // println!("{:?}", new_vector);

        new_vector

    }

    #[inline]
    pub fn clone_to_container(&self, mut local_node_vector: SmallVec<[Node;1024]>) -> RankVector<SmallVec<[Node;1024]>> {

        local_node_vector.clear();

        for node in &self.nodes {
            local_node_vector.push(node.clone());
        }

        RankVector {
            nodes: local_node_vector,
            drop_set: None,
            dirty_set: None,
            rank_order: None,
            drop: self.drop.clone(),
            zones: self.zones.clone(),
            offset: self.offset,
            median: self.median,
            left: self.left,
            right: self.right,
        }

    }

}

impl RankVector<SmallVec<[Node;1024]>> {

    pub fn return_container(self) -> SmallVec<[Node;1024]> {

        self.nodes

    }
}


pub fn sanitize_vector(in_vec:&Vec<f64>) -> (Vec<f64>,HashSet<usize>) {

    (
        in_vec.iter().map(|x| if !x.is_normal() {0.} else {*x}).collect(),

        in_vec.iter().enumerate().filter(|x| !x.1.is_normal() && *x.1 != 0.).map(|x| x.0).collect()
    )

}



impl<'a,T: Borrow<[Node]> + BorrowMut<[Node]> + Index<usize,Output=Node> + IndexMut<usize,Output=Node> + Clone + Debug> GRVCrawler<'a,T> {

    #[inline]
    fn new(input: &'a RankVector<T>, first: usize) -> GRVCrawler<'a,T> {
        GRVCrawler{vector: input, index: first}
    }
}

impl<'a,T: Borrow<[Node]> + BorrowMut<[Node]> + Index<usize,Output=Node> + IndexMut<usize,Output=Node> + Clone + Debug> Iterator for GRVCrawler<'a,T> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {

        let Node{next:next,index:index,..} = self.vector.nodes[self.index];
        self.index = next;
        return Some(index)
    }
}


pub struct GRVCrawler<'a, T: 'a + Borrow<[Node]> + BorrowMut<[Node]> + Index<usize,Output=Node> + IndexMut<usize,Output=Node> + Clone + Debug> {
    vector: &'a RankVector<T>,
    index: usize,
}

impl<'a,T:Borrow<[Node]> + BorrowMut<[Node]> + Index<usize,Output=Node> + IndexMut<usize,Output=Node> + Clone + Debug> GLVCrawler<'a,T> {

    #[inline]
    fn new(input: &'a RankVector<T>, first: usize) -> GLVCrawler<'a,T> {
        GLVCrawler{vector: input, index: first}
    }
}

impl<'a,T: Borrow<[Node]> + BorrowMut<[Node]> + Index<usize,Output=Node> + IndexMut<usize,Output=Node> + Clone + Debug> Iterator for GLVCrawler<'a,T> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {

        let Node{previous:previous,index:index,..} = self.vector.nodes[self.index];
        self.index = previous;
        return Some(index)
    }
}

pub struct GLVCrawler<'a, T:'a + Borrow<[Node]> + BorrowMut<[Node]> + Index<usize,Output=Node> + IndexMut<usize,Output=Node> + Clone + Debug> {
    vector: &'a RankVector<T>,
    index: usize,
}

fn slow_median(values: Vec<f64>) -> f64 {
    let median: f64;
    if values.len() < 1 {
        return 0.
    }

    if values.len()%2==0 {
        median = (values[values.len()/2] + values[values.len()/2 - 1]) as f64 / 2.;
    }
    else {
        median = values[(values.len()-1)/2];
    }

    median

}

fn slow_mad(values: Vec<f64>) -> f64 {
    let median: f64;
    if values.len() < 1 {
        return 0.
    }
    if values.len()%2==0 {
        median = (values[values.len()/2] + values[values.len()/2 - 1]) as f64 / 2.;
    }
    else {
        median = values[(values.len()-1)/2];
    }

    let mut abs_deviations: Vec<f64> = values.iter().map(|x| (x-median).abs()).collect();

    abs_deviations.sort_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

    let mad: f64;
    if abs_deviations.len()%2==0 {
        mad = (abs_deviations[abs_deviations.len()/2] + abs_deviations[abs_deviations.len()/2 - 1]) as f64 / 2.;
    }
    else {
        mad = abs_deviations[(abs_deviations.len()-1)/2];
    }

    mad

}

#[cfg(test)]
mod rank_vector_tests {

    use super::*;
    use rand::{thread_rng,Rng};
    use rand::distributions::Standard;
    use rand::seq::sample_indices;

    #[test]
    fn create_trivial() {
        let mut vector = RankVector::<Vec<Node>>::link(&vec![]);
        vector.drop_f(0.);
    }

    #[test]
    fn create_very_simple_drop() {
        let mut vector = RankVector::<Vec<Node>>::link(&vec![0.]);
        println!("{:?}",vector);
        vector.drop_f(0.);
    }


    #[test]
    fn create_very_simple_nan_drop() {
        let mut vector = RankVector::<Vec<Node>>::link(&vec![f64::NAN]);
        println!("{:?}",vector);
        vector.drop_f(f64::NAN);
    }


    #[test]
    fn create_simple() {
        let mut vector = RankVector::<Vec<Node>>::link(&vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]);
        vector.drop_f(0.);
        println!("{:?}",vector);
        assert_eq!(vector.ordered_values(),vec![-3.,-2.,-1.,5.,10.,15.,20.]);
        assert_eq!(vector.median(),slow_median(vector.ordered_values()));
        assert_eq!(slow_mad(vector.ordered_values()),vector.mad());

    }

    #[test]
    fn create_repetitive() {
        let mut vector = RankVector::<Vec<Node>>::link(&vec![0.,0.,0.,-5.,-5.,-5.,10.,10.,10.,10.,10.]);
        vector.drop_f(0.);
        println!("{:?}",vector);
        assert_eq!(vector.ordered_values(),vec![-5.,-5.,-5.,10.,10.,10.,10.,10.]);
        assert_eq!(vector.median(),10.);
        assert_eq!(vector.mad(),0.);
    }

    #[test]
    fn sequential_mad_simple() {
        let mut vector = RankVector::<Vec<Node>>::link(&vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],);
        vector.drop_f(0.);

        let mut vm = vector.clone();


        for draw in vector.draw_order() {
            println!("{:?}",vm.ordered_values());
            println!("Median:{},{}",vm.median(),slow_median(vm.ordered_values()));
            println!("MAD:{},{}",vm.mad(),slow_mad(vm.ordered_values()));
            println!("Boundaries:{:?}", vm.boundaries());
            println!("{:?}",vm.pop(draw));
            println!("{:?}",vm.ordered_values());
            println!("Median:{},{}",vm.median(),slow_median(vm.ordered_values()));
            println!("MAD:{},{}",vm.mad(),slow_mad(vm.ordered_values()));
            println!("Boundaries:{:?}", vm.boundaries());
            assert_eq!(vm.median(),slow_median(vm.ordered_values()));
            assert_eq!(vm.mad(),slow_mad(vm.ordered_values()));
        }

    }

    #[test]
    fn fetch_test() {
        let mut vector = RankVector::<Vec<Node>>::link(&vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],);
        vector.drop_f(0.);
        assert_eq!(vector.fetch(0),10.);
        assert_eq!(vector.fetch(1),-3.);
        assert_eq!(vector.fetch(2),0.);
        assert_eq!(vector.fetch(3),5.);
    }

}
