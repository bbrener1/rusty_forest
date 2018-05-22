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
    zones: [Zone;4],
    zone_offset: usize,
    median: (usize,usize),
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

#[derive(Clone,Copy,Debug,Serialize,Deserialize)]
struct Zone {
    head: usize,
    tail: usize,
    length: usize,
    id: usize,
}

impl Zone {

    pub fn initialize(container: &mut Vec<Node>,id:usize) -> Zone{

        let head = Node {
            data: 0.,
            index: container.len(),
            previous: container.len(),
            next: container.len() + 1,
            zone: id,
            rank: 0,
        };

        let tail = Node {
            data: 0.,
            index: container.len() + 1,
            previous: container.len(),
            next: container.len() + 1,
            zone: id,
            rank: 0,
        };

        let zone = Zone {
            head: head.index,
            tail: tail.index,
            length: 0,
            id: id,
        };

        container.push(head);
        container.push(tail);

        zone

    }

    pub fn blank() -> Zone {

        Zone {
            head: 0,
            tail: 0,
            length: 0,
            id: 0,
        }

    }

}

impl<T: Borrow<[Node]> + BorrowMut<[Node]> + Index<usize,Output=Node> + IndexMut<usize,Output=Node> + Clone + Debug > RankVector<T> {

    pub fn with_capacity(capacity:usize) -> RankVector<Vec<Node>> {
        let mut vector: Vec<Node> = Vec::with_capacity(capacity+8);

        let drop = Zone::initialize(&mut vector, 0);
        let left = Zone::initialize(&mut vector, 1);
        let median = Zone::initialize(&mut vector, 2);
        let right = Zone::initialize(&mut vector, 3);

        let zones = [drop,left,median,right];

        let dirty_set = HashSet::with_capacity(0);
        let median = (4,4);
        let drop_set = HashSet::with_capacity(0);
        let rank_order = Vec::with_capacity(0);

        RankVector::<Vec<Node>> {
            nodes: vector,
            drop_set: Some(drop_set),
            dirty_set: Some(dirty_set),
            rank_order: Some(rank_order),
            drop: DropMode::No,
            zones: zones,
            zone_offset: 8,
            median: median,
        }

    }

    pub fn link(in_vec: &Vec<f64>) -> RankVector<Vec<Node>> {

        let mut vector: Vec<Node> = Vec::with_capacity(in_vec.len()+8);

        let drop = Zone::initialize(&mut vector, 0);
        let left = Zone::initialize(&mut vector, 1);
        let median = Zone::initialize(&mut vector, 2);
        let right = Zone::initialize(&mut vector, 3);

        let mut zones = [drop,left,median,right];

        let (clean_vector,dirty_set) = sanitize_vector(in_vec);

        let mut sorted_invec = clean_vector.into_iter().enumerate().collect::<Vec<(usize,f64)>>();

        sorted_invec.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));

        let mut rank_order = Vec::with_capacity(sorted_invec.len());

        let mut previous_node_local_index = zones[2].head;
        let tail_node_index = zones[2].tail;

        for (ranking,&(index,data)) in sorted_invec.iter().enumerate() {

            let node = Node {
                data: data,
                index: index+8,
                previous: vector[previous_node_local_index].index,
                next: tail_node_index,
                zone: 2,
                rank: ranking,
            };

            // println!("{:?}", vector);

            vector[previous_node_local_index].next = index+8;

            previous_node_local_index = vector.len();

            vector[tail_node_index].previous = index+8;

            rank_order.push(index);

            vector.push(node);

            zones[2].length += 1;

        };

        let median = (4,4);

        vector.sort_unstable_by_key(|x| x.index);

        let drop_set = HashSet::with_capacity(0);

        let mut prototype = RankVector::<Vec<Node>> {
            nodes: vector,
            drop_set: Some(drop_set),
            dirty_set: Some(dirty_set),
            rank_order: Some(rank_order),
            drop: DropMode::No,
            zones: zones,
            zone_offset: 8,
            median: median,
        };

        // println!("Linking");

        prototype.establish_median();
        prototype.establish_zones();

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

    pub fn pop(&mut self, nominal_index: usize) -> f64 {

        // println!("{:?}", self);

        let target = nominal_index + self.zone_offset;

        let target_zone = self.nodes[target].zone;

        // println!("Popping {}", target);

        if self.nodes[target].zone != 0 {

            // println!("Popping internal");

            self.zones[target_zone].length -= 1;
            self.zones[0].length += 1;
            self.nodes[target].zone = 0;

            // println!("Balancing");
            // println!("{:?}", self.zones);
            // println!("{:?}", self.median);

            self.balance_zones(target);

            // println!("{:?}", self.zones);
            // println!("{:?}", self.median);

            self.unlink(target);

            // println!("Recentering");
            // println!("{:?}", self.median);
            let (old_median,new_median) = self.recenter_median(target);
            // println!("{:?}", self.median);

            //
            // println!("Shifting zones");

            self.shift_zones(old_median,new_median);

        }

        if (self.median() - slow_median(self.ordered_values())).abs() > 0.00001 {
            println!("{:?}", self);
            println!("{:?}", self.ordered_values());
            println!("{:?}", slow_median(self.ordered_values()));
            panic!("Failed to adjust median!");
        };

        if (self.mad() - slow_mad(self.ordered_values())).abs() > 0.00001 {
            println!("{:?}", self);
            println!("{:?}", self.ordered_values());
            println!("{:?}", self.mad());
            println!("{:?}", slow_mad(self.ordered_values()));
            panic!("Failed to adjust mad");
        };

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

    //

    #[inline]
    fn push_next_internal(&mut self, left: usize, target: usize) {

        let right = self.nodes[left].next;

        self.zones[self.nodes[target].zone].length -= 1;

        self.nodes[left].next = target;
        self.nodes[target].previous = left;
        self.nodes[target].zone = self.nodes[left].zone;
        self.nodes[target].next = right;
        self.nodes[right].previous = target;

        self.zones[self.nodes[target].zone].length += 1;


    }

    #[inline]
    pub fn establish_median(&mut self) {

        let order = self.left_to_right();

        match order.len() % 2 {
            0 => {
                self.median = (order[(order.len()/2)-1],order[order.len()/2]);
            },
            1 => {
                self.median = (order[order.len()/2],order[order.len()/2]);
            },
            _ => unreachable!(),
        }

        if (self.median() - slow_median(self.ordered_values())) > 0.00001 {
            println!("{:?}", self);
            println!("{:?}", self.ordered_values());
            println!("{:?}", slow_median(self.ordered_values()));
            panic!("Failed to establish median!");
        }
    }

    #[inline]
    pub fn establish_zones(&mut self) {
        for _ in 0..(((self.len())/2).max(1) - (1 - self.len()%2)) {
            self.contract_1();
        };

        if (self.mad() - slow_mad(self.ordered_values())) > 0.00001 {
            println!("{:?}", self);
            println!("{:?}", self.ordered_values());
            println!("{:?}", self.mad());
            println!("{:?}", slow_mad(self.ordered_values()));
            panic!("Failed to establish mad");
        };

    }

    #[inline]
    fn push_previous_internal(&mut self, right: usize, target: usize) {

        let left = self.nodes[right].previous;

        self.zones[self.nodes[target].zone].length -= 1;

        self.nodes[left].next = target;
        self.nodes[target].previous = left;
        self.nodes[target].zone = self.nodes[right].zone;
        self.nodes[target].next = right;
        self.nodes[right].previous = target;

        self.zones[self.nodes[target].zone].length += 1;

    }

    #[inline]
    pub fn len(&self) -> usize {
        self.zones[1].length + self.zones[2].length + self.zones[3].length
    }

    #[inline]
    pub fn raw_len(&self) -> usize {
        self.zones[0].length + self.zones[1].length + self.zones[2].length + self.zones[3].length
    }

    #[inline]
    pub fn drop_f(&mut self, f: f64) {

        let mut drop_set: HashSet<usize> = HashSet::with_capacity(self.len());
        drop_set = self.left_to_right().iter().map(|x| &self.nodes[*x]).filter(|y| y.data == f).map(|x| x.index).collect();
        let offset = self.zone_offset;
        for index in drop_set.iter().map(|x| *x - offset) {
            self.pop(index);
        };
        drop_set.shrink_to_fit();
        self.drop_set.as_mut().unwrap().extend(drop_set.iter());
        self.drop = DropMode::Zeros;
    }

    #[inline]
    pub fn push_tail(&mut self, input: usize, zone: usize) {

        let tail = self.zones[zone].tail;

        self.push_previous_internal(tail , input);

    }

    #[inline]
    pub fn push_head(&mut self, input: usize, zone: usize) {

        let head = self.zones[zone].head;

        self.push_next_internal(head,input);

    }

    #[inline]
    pub fn pop_head(&mut self, zone: usize) -> usize {

        let head = self.nodes[self.zones[zone].head].next;

        self.unlink(head);

        self.zones[self.nodes[head].zone].length -= 1;
        self.zones[0].length += 1;

        self.nodes[head].zone = 0;

        head

    }

    #[inline]
    pub fn pop_tail(&mut self, zone: usize) -> usize {

        let tail = self.nodes[self.zones[zone].tail].previous;

        self.unlink(tail);

        self.zones[self.nodes[tail].zone].length -= 1;
        self.zones[0].length += 1;

        self.nodes[tail].zone = 0;

        tail

    }

    #[inline]
    pub fn contract_left(&mut self) {
        let passed = self.pop_head(2);
        self.push_tail(passed,1)
    }

    #[inline]
    pub fn contract_right(&mut self) {
        let passed = self.pop_tail(2);
        self.push_head(passed,3)
    }


    #[inline]
    pub fn expand_left(&mut self) {
        let passed = self.pop_tail(1);
        self.push_head(passed,2)
    }

    #[inline]
    pub fn expand_right(&mut self) {
        let passed = self.pop_head(3);
        self.push_tail(passed,2)
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

        if self.zones[1].length > 0 && self.zones[3].length > 0 {

            let left = self.nodes[self.nodes[self.zones[1].tail].previous].data;
            let right = self.nodes[self.nodes[self.zones[3].head].next].data;

            if (right - median).abs() > (median - left).abs() {
                self.expand_left();
            }
            else {
                self.expand_right();
            }

        }
        else {
            if self.zones[1].length == 0 {
                self.expand_right();
            }
            else if self.zones[3].length == 0 {
                self.expand_left();
            }
        }

    }


    #[inline]
    pub fn contract_1(&mut self) {

        let left = self.nodes[self.nodes[self.zones[2].head].next].data;
        let median = self.median();
        let right = self.nodes[self.nodes[self.zones[2].tail].previous].data;

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
                    match self.zones[2].length.cmp(&(self.zones[1].length + self.zones[3].length + 1)) {
                        Ordering::Greater => self.contract_1(),
                        Ordering::Less => self.expand_1(),
                        Ordering::Equal => {},
                    }
                },
                0 => {
                    match self.zones[2].length.cmp(&(self.zones[1].length + self.zones[3].length + 2)) {
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
                self.median = (self.median.0,self.median.0)
            },
            true => {
                self.median = (self.nodes[self.median.0].previous,self.median.0)
            }
        }
    }

    #[inline]
    pub fn shift_median_right(&mut self) {
        match self.median.0 == self.median.1 {
            false => {
                self.median = (self.median.1,self.median.1)
            },
            true => {
                self.median = (self.median.1,self.nodes[self.median.1].next)
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

            for i in 0..self.zones[3].length {

                let left = self.nodes[self.nodes[self.zones[2].head].next].data;
                let right = self.nodes[self.nodes[self.zones[3].head].next].data;

                if (right - new_median).abs() > (left - new_median).abs() {

                    break
                }

                self.move_right()

            }
        }
        if change < 0. {

            for i in 0..self.zones[1].length {

                let left = self.nodes[self.nodes[self.zones[1].tail].previous].data;
                let right = self.nodes[self.nodes[self.zones[2].tail].previous].data;

                if (left - new_median).abs() > (right - new_median).abs() {
                    break
                }

                self.move_left()

            }
        }

    }


    #[inline]
    pub fn mad(&self) -> f64 {

        if self.len() < 2 {return 0.}

        let left_i = self.nodes[self.zones[2].head].next;
        let right_i = self.nodes[self.zones[2].tail].previous;

        let inner_left_i = self.nodes[left_i].next;
        let inner_right_i = self.nodes[right_i].previous;

        let left = self.nodes[left_i].data;
        let right = self.nodes[right_i].data;
        let inner_left = self.nodes[inner_left_i].data;
        let inner_right = self.nodes[inner_right_i].data;

        let median = self.median();

        let mut distance_to_median = [(left - median).abs(), (inner_left - median).abs(), (inner_right - median).abs(), (right - median).abs()];

        // println!("MAD debug");
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
        let mut indecies = Vec::with_capacity(self.len());
        indecies.extend(GRVCrawler::new(&self, self.nodes[self.zones[1].head].next).take(self.zones[1].length));
        indecies.extend(GRVCrawler::new(&self, self.nodes[self.zones[2].head].next).take(self.zones[2].length));
        indecies.extend(GRVCrawler::new(&self, self.nodes[self.zones[3].head].next).take(self.zones[3].length));

        indecies
    }

    #[inline]
    pub fn ordered_values(&self) -> Vec<f64> {
        self.left_to_right().iter().map(|x| self.nodes[*x].data).collect()
    }

    pub fn full_values(&self) -> Vec<f64> {
        (self.zone_offset..(self.len() + self.zone_offset)).map(|x| self.nodes[x].data).collect()
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
            mad_gains.push(start_mad - self.mad())
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
            if (self.mad() - slow_mad(self.ordered_values())).abs() > 0.00001 {
                println!("{:?}", self);
                println!("{:?}", self.ordered_values());
                println!("{:?}", self.mad());
                println!("{:?}", slow_mad(self.ordered_values()));
                panic!("Mad mismatch");
            }
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
        self.left_to_right().into_iter().map(|x| x-self.zone_offset).collect()
    }

    #[inline]
    pub fn draw_and_drop(&self) -> (Vec<usize>,HashSet<usize>) {
        let offset = self.zone_offset;
        let draw_order = self.left_to_right().into_iter().map(|x| x - offset).collect();
        let offset_drop_set = self.drop_set.as_ref().unwrap().iter().map(|x| *x - offset).collect();
        (draw_order,offset_drop_set)
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
    pub fn fetch(&self, nominal_index:usize) -> f64 {
        self.nodes[nominal_index + self.zone_offset].data
    }

    #[inline]
    pub fn boundaries(&self) -> ((usize,f64),(usize,f64)) {
        let (left,right) = (self.nodes[4].next, self.nodes[5].previous);
        ((left,self.nodes[left].data),(right,self.nodes[right].data))
    }

    pub fn drop_using_mode(&mut self) {
        let cmp = self.drop.cmp();
        self.drop_f(cmp);
    }

}

impl RankVector<Vec<Node>> {

    pub fn derive(&self, indecies:&[usize]) -> RankVector<Vec<Node>> {

        let mut new_vector = RankVector::<Vec<Node>>::with_capacity(indecies.len());

        let mut new_rank_order = Vec::with_capacity(indecies.len());

        let derived_set: HashSet<usize> = indecies.iter().cloned().collect();

        let index_map: HashMap<usize,usize> =
            ((self.zone_offset)..(self.raw_len()+self.zone_offset))
                .filter(|x| derived_set.contains(&(*x-self.zone_offset)))
                .enumerate()
                .map(|(i,x)| (x,i+self.zone_offset))
                .collect();


        let mut new_dirty_set: HashSet<usize> = HashSet::with_capacity(indecies.len());

        let mut previous = 4;
        let mut new_index = 0;
        let mut last_local_index = 4;
        let dirty_indirect = self.dirty_set.as_ref().unwrap();

        for (i,node) in self.rank_order.as_ref().unwrap_or(&vec![]).iter().filter(|y| derived_set.contains(y)).map(|x| &self.nodes[*x+self.zone_offset]).enumerate() {

                new_index = index_map[&node.index];

                if dirty_indirect.contains(&node.index) {
                    new_dirty_set.insert(new_index);
                };

                let new_node = Node {
                    data: node.data,
                    index: new_index,
                    rank: i,
                    previous: previous,
                    next: 5,
                    zone: 2,
                };

                new_vector.nodes[last_local_index].next = new_index;
                new_vector.nodes.push(new_node);
                new_rank_order.push(new_index - self.zone_offset);
                new_vector.zones[2].length += 1;

                last_local_index = new_vector.nodes.len()-1;

                previous = new_index;

        }

        new_vector.nodes[5].previous = new_index;

        new_vector.nodes.sort_unstable_by_key(|x| x.index);

        new_vector.rank_order = Some(new_rank_order);

        new_vector.establish_median();
        new_vector.establish_zones();
        new_vector.drop_f(0.);

        // println!("{:?}", new_vector);

        new_vector

    }

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
            drop: self.drop,
            zones: self.zones,
            zone_offset: self.zone_offset,
            median: self.median,
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
    use std::f64::NAN;

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

    // #[test]
    // fn sequential_mad_simple_nan() {
    //     let mut vector = RankVector::new(&vec![10.,-3.,NAN,5.,-2.,-1.,15.,20.],"test".to_string(),Arc::new(Parameters::empty()));
    //     vector.drop_zeroes();
    //     vector.initialize();
    //     vector.set_boundaries();
    //
    //     let mut vm = vector.clone();
    //
    //
    //     for draw in vector.draw_order {
    //         println!("{:?}",vm.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
    //         println!("Median:{},{}",vm.median(),slow_median(&vm.vector));
    //         println!("MAD:{},{}",vm.mad(),slow_mad(&vm.vector));
    //         println!("{:?}",vm.pop(draw));
    //         println!("{:?}",vm.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
    //         println!("Median:{},{}",vm.median(),slow_median(&vm.vector));
    //         println!("MAD:{},{}",vm.mad(),slow_mad(&vm.vector));
    //         assert_eq!(vm.median(),slow_median(&vm.vector));
    //         assert_eq!(vm.mad(),slow_mad(&vm.vector));
    //     }
    //
    // }
    //
    // #[test]
    // fn odds_ratio() {
    //     let mut vector = RankVector::new(&vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],"test".to_string());
    //     vector.drop_zeroes();
    //     vector.initialize();
    //     vector.set_boundaries();
    //
    // }



}
