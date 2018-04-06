
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::collections::HashSet;

extern crate rand;

use raw_vector::RawVector;
use raw_vector::LeftVectCrawler;
use raw_vector::RightVectCrawler;
use DropMode;

impl RankVector {

    pub fn new(in_vec:&Vec<f64>, feature_name: String,drop: DropMode) -> RankVector {

        let vector = RawVector::raw_vector(in_vec);

        // println!("Made raw vector object!");

        let empty_backup = RawVector::raw_vector(&Vec::with_capacity(0));

        // println!("Made an empty backup!");

        let length = vector.len();

        // println!("Trying to intialize ranked object");

        let zones = RankVector::empty_zones();

        RankVector{

            length: length,

            backup: false,

            backup_vector: empty_backup,
            backup_left: zones.0.clone(),
            backup_median: zones.1.clone(),
            backup_right: zones.2.clone(),

            vector: vector,

            draw_order: (0..length).collect(),

            drop: drop,
            num_dropped: 0,

            left_zone:zones.0,
            median_zone:zones.1,
            right_zone:zones.2,


            feature_name: feature_name,
        }

    }

    pub fn empty() -> RankVector {
        RankVector::new(&vec![], "".to_string(), DropMode::No)
    }

    fn empty_zones() -> (LeftZone,MedianZone,RightZone) {

        let median_zone = MedianZone {
            size: 0,
            dead_center: DeadCenter{left:Some((0,0,0,0.,0)),right:Some((0,0,0,0.,0))},
            left: None,
            right: None,
            index_set: HashSet::with_capacity(0)
        };
        let left_zone = LeftZone {
            size: 0,
            right: None,
            index_set: HashSet::with_capacity(0)
        };
        let right_zone = RightZone{
            size: 0,
            left: None,
            index_set: HashSet::with_capacity(0)
        };
        (left_zone,median_zone,right_zone)
    }

    pub fn initialize(&mut self) {

        // println!("Initializing!");

        let dead_center = DeadCenter::center(&self.vector);

        let mut leftward: LeftVectCrawler;
        let mut rightward: RightVectCrawler;


        if let (Some(d_l),Some(d_r)) = (dead_center.left,dead_center.right) {
            leftward = self.vector.crawl_left(d_l.1);
            rightward = self.vector.crawl_right(d_r.1);
        }
        else {
            leftward = LeftVectCrawler::empty(&self.vector.vector);
            rightward = RightVectCrawler::empty(&self.vector.vector);
        }

        let left_option = leftward.next();
        let right_option = rightward.next();

        let mut left_zone = 0;
        let mut right_zone = 0;

        let mut left_set: HashSet<usize> = HashSet::with_capacity(self.vector.vector.len()/2);
        let mut middle_set: HashSet<usize> = HashSet::with_capacity(self.vector.vector.len()/2);
        let mut right_set: HashSet<usize> = HashSet::with_capacity(self.vector.vector.len()/2);

        let left_object: LeftZone;
        let median_object: MedianZone;
        let right_object: RightZone;

        if let (Some(left),Some(right)) = (left_option,right_option) {
            if left == right {
                middle_set.insert(left.1);
                median_object = MedianZone{ size:1 ,dead_center:dead_center,left:Some(left.1),right:Some(right.1), index_set: middle_set};
            }
            else {
                middle_set.insert(left.1);
                middle_set.insert(right.1);
                median_object = MedianZone{ size:2 ,dead_center:dead_center,left:Some(left.1),right:Some(right.1), index_set: middle_set};
            }

            for sample in leftward {
                left_set.insert(sample.1);
                left_zone += 1;
            }
            for sample in rightward {
                right_set.insert(sample.1);
                right_zone += 1;
            }
            left_object = LeftZone{size: left_zone, right:self.vector.left_ind(left.1), index_set: left_set};
            right_object = RightZone{size: right_zone, left: self.vector.right_ind(right.1), index_set: right_set};

        }
        else{
            median_object = MedianZone{ size: 0, dead_center:dead_center,left:None,right:None,index_set:middle_set};
            left_object = LeftZone{size: left_zone, right:None, index_set: left_set};
            right_object = RightZone{size: right_zone, left: None, index_set: right_set};
        }


        self.left_zone = left_object;
        self.median_zone = median_object;
        self.right_zone = right_object;

    }

    pub fn drop_target(&mut self, target: usize) {

        if self.vector.drop(target) {

            if self.backup {
                self.backup_vector.drop(target);
            }

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

            self.num_dropped += 1;
        }

    }

    pub fn drop_zeroes(&mut self) {

        let samples_to_drop: Vec<usize> = self.vector.iter_full().filter(|x| x.3 == 0.).map(|x| x.1).collect();

        for sample in samples_to_drop {
            self.drop_target(sample);
        }

    }

    pub fn drop_nan(&mut self) {

        let samples_to_drop: HashSet<usize> = self.vector.dirty_set.clone();

        for sample in samples_to_drop {
            self.drop_target(sample);
        }

    }

    pub fn drop(&mut self) {

        match self.drop {
            DropMode::Zeros => self.drop_zeroes(),
            DropMode::NaNs => self.drop_nan(),
            DropMode::No => {},
        }

    }

    pub fn reset(&mut self) {

        self.vector = self.backup_vector.clone();

        self.left_zone = self.backup_left.clone();
        self.median_zone = self.backup_median.clone();
        self.right_zone = self.backup_right.clone();
        self.length = self.vector.len();

    }

    pub fn manual_reset(&mut self) {

        self.vector.reset_by_reference(&self.backup_vector);
        self.left_zone.reset_by_reference(&self.backup_left);
        self.right_zone.reset_by_reference(&self.backup_right);
        self.median_zone.reset_by_reference(&self.backup_median);
        self.length = self.vector.len();

    }

    pub fn backup(&mut self) {
        self.backup = true;
        self.backup_vector = self.vector.clone();
        self.backup_left = self.left_zone.clone();
        self.backup_median = self.median_zone.clone();
        self.backup_right = self.right_zone.clone();
    }

    pub fn pop(&mut self, target: usize) -> (usize,usize,usize,f64,usize) {

        // let start_time = time::PreciseTime::now();


        if self.vector.drop_set.contains(&target) {
            return self.vector[target]
        }
        if self.median_zone.index_set.contains(&target) {
            if self.vector.len() < 2 {
                self.median_zone.dead_center.left = Some((0,0,0,0.,0));
                self.median_zone.dead_center.right = Some((0,0,0,0.,0));
                self.median_zone.left = Some(target);
                self.median_zone.right = Some(target);
                if self.vector.len() == 1 {
                    return self.vector.pop(target)
                }
                else {
                    return (0,0,0,0.,0)
                }
            }
            if self.median_zone.left.unwrap() == target {
                self.median_zone.left = self.vector.right_ind(target);
            }
            if self.median_zone.right.unwrap() == target {
                self.median_zone.right = self.vector.left_ind(target);
            }
            self.median_zone.size -= 1;
            self.median_zone.index_set.remove(&target);
        }
        if self.left_zone.index_set.contains(&target) {
            if self.left_zone.right.unwrap() == target {
                self.left_zone.right = self.vector.left_ind(target);
            }
            self.left_zone.size -= 1;
            self.left_zone.index_set.remove(&target);
        }
        if self.right_zone.index_set.contains(&target) {
            if self.right_zone.left.unwrap() == target {
                self.right_zone.left = self.vector.right_ind(target);
            }
            self.right_zone.size -= 1;
            self.right_zone.index_set.remove(&target);
        }
        let medians = self.median_zone.dead_center.re_center(&target, &self.vector);
        let removed = self.vector.pop(target);

        // let start_time = time::PreciseTime::now();
        self.zone_balance();
        // let end_time = time::PreciseTime::now();
        // if target == 300 {
        //     println!("Time for a zone balance: {}ns", start_time.to(end_time).num_nanoseconds().unwrap_or(-1));
        // }

        // let start_time = time::PreciseTime::now();
        self.zone_shift(medians.0,medians.1);
        // let end_time = time::PreciseTime::now();
        // if target == 300 {
        //     println!("Time for a zone shift: {}ns", start_time.to(end_time).num_nanoseconds().unwrap_or(-1));
        // }


        if (self.vector.len() != (self.median_zone.size+self.left_zone.size+self.right_zone.size)) ||
            (self.vector.len() > 0 && (self.median_zone.dead_center.left.is_none() || self.median_zone.dead_center.right.is_none())) ||
            (self.vector.len() > 0 && (self.median_zone.left.is_none() || self.median_zone.right.is_none())) {
                println!("The state is fucked!");
                println!("Own length: {}", self.vector.len());
                println!("{:?}", self.zones());
                println!("{:?}", self.boundaries());
                println!("{:?}", self);
                panic!("State de-sync");
            }

        // let end_time = time::PreciseTime::now();
        //
        // if target == 300 {
        //     println!("Time for a single pop: {}ns", start_time.to(end_time).num_nanoseconds().unwrap_or(-1));
        // }


        removed
    }

    pub fn median(&self) -> f64 {
        self.median_zone.dead_center.median()
    }

    pub fn mad(&self) -> f64 {
        self.median_zone.mad(&self.vector)
        // let fmad = self.median_zone.mad(&self.vector);
        // if slow_mad(&self.vector) != fmad {
        //     println!("{:?}", self);
        //     panic!("MAD De-sync")
        // }
        // fmad
    }

    pub fn set_boundaries(&mut self) {
        // println!("Setting boundaries!");
        if self.vector.len() < 1 {
            return
        }
        while (self.left_zone.size + self.right_zone.size) > (self.median_zone.size - 1) {
            self.expand_by_1();
            // println!("Moving:{:?}",self.expand_by_1());
            // println!("Zones: {:?}",self.zones());
        }
    }

    pub fn set_draw(&mut self, order:Vec<usize>) {
        self.vector.set_draw(order);
    }

    pub fn zone_shift(&mut self, old_median: f64, new_median: f64) {
        let change = new_median - old_median;

        if change < 0. {
            while let Some(left) = self.vector.left(self.median_zone.left.unwrap()) {
                let right = self.vector[self.median_zone.right.unwrap()];

                // println!("Zone shift left debug:");
                // println!("{}", new_median);
                // println!("{:?},{:?}",left,right);
                // println!("Comparison: {},{}",(left.3 - new_median).abs(),(right.3 - new_median).abs());

                if (left.3 - new_median).abs() > (right.3 - new_median).abs() {
                    break
                }
                self.median_zone.expand_left(&self.vector);
                self.median_zone.contract_right(&self.vector);
                self.left_zone.contract(&self.vector);
                self.right_zone.expand(&self.vector);
            }
        }
        if change > 0. {
            while let Some(right) = self.vector.right(self.median_zone.right.unwrap()) {
                let left = self.vector[self.median_zone.left.unwrap()];

                // println!("Zone shift right debug:");
                // println!("{}", new_median);
                // println!("{:?},{:?}",left,right);
                // println!("Comparison: {},{}",(left.3 - new_median).abs(),(right.3 - new_median).abs());

                if (right.3 - new_median).abs() > (left.3 - new_median).abs() {
                    break
                }
                self.median_zone.expand_right(&self.vector);
                self.median_zone.contract_left(&self.vector);
                self.left_zone.expand(&self.vector);
                self.right_zone.contract(&self.vector);
            }
        }
    }

    pub fn zone_balance(&mut self) {
        while (((self.left_zone.size + self.right_zone.size) as i32) < (self.median_zone.size as i32 - 2) && self.vector.len()%2 == 0) ||
        (((self.left_zone.size + self.right_zone.size) as i32) < (self.median_zone.size as i32 - 1) && self.vector.len()%2 == 1)
        {
            // println!("Zones: {},{},{},{},{},{}",self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len());
            self.contract_by_1();
            // println!("Zones: {},{},{},{},{},{}",self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len())
        }
        while ((self.left_zone.size + self.right_zone.size) >= (self.median_zone.size) && self.vector.len()%2 == 1) ||
            ((self.left_zone.size + self.right_zone.size) >= (self.median_zone.size - 1) && self.vector.len()%2 == 0)
        {
            // println!("Zones: {},{},{},{},{},{}",self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len());
            self.expand_by_1();
            // println!("Zones: {},{},{},{},{},{}",self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len())
        }
    }

    fn expand_by_1(&mut self) -> Option<usize> {

        // if self.vector.len() < 2 {
        //     panic!("Asked to expand a singleton zone!");
        // }

        let mut moved_index = (Some(0),Some(0));

        if self.vector.len() < 1 {
            eprintln!("Asked to expand an empty zone!");
            return None
        }

        let left = self.median_zone.left.unwrap();
        let right = self.median_zone.right.unwrap();

        let outer_right = self.vector.right(right);
        let outer_left = self.vector.left(left);

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

        // if let (Some(m1),Some(m2)) = moved_index {
        //     if m1 == m2 {
        //         return Some(m1)
        //     }
        //     else {
        //         eprintln!("{},{}",m1,m2);
        //         panic!{"Mismatch of moved indecies, zones are out of sync!"}
        //     }
        //
        // }

        None


    }

    fn contract_by_1(&mut self) -> Option<usize> {

        let mut moved_index = (Some(0),Some(0));

        if self.vector.len() < 1 {
            eprintln!("Asked to contract an empty zone!");
            return None
        }

        if self.vector.len() == 1 {
            if let Some(_output) = self.median_zone.contract_left(&self.vector) {
                self.median_zone.left = None;
                self.median_zone.right = None;
                self.median_zone.dead_center.left = None;
                self.median_zone.dead_center.right = None;
            }

        }

        let left = self.vector[self.median_zone.left.unwrap()];
        let right = self.vector[self.median_zone.right.unwrap()];

        let median = self.median_zone.dead_center.median();

        if (left.3 - median).abs() < (right.3 - median).abs() {
            moved_index.0 = self.median_zone.contract_right(&self.vector);
            moved_index.1 = self.right_zone.expand(&self.vector);
        }
        else {
            moved_index.0 = self.median_zone.contract_left(&self.vector);
            moved_index.1 = self.left_zone.expand(&self.vector);
        }

        // if let (Some(m1),Some(m2)) = moved_index {
        //     if m1 == m2 {
        //         return Some(m1)
        //     }
        //     else {
        //         eprintln!("{},{}",m1,m2);
        //         panic!{"Mismatch of moved indecies, zones are out of sync!"}
        //     }
        //
        // }

        None

    }

    fn boundaries (&self) -> (usize,f64,usize,f64) {
        (self.median_zone.left.unwrap(),self.vector[self.median_zone.left.unwrap()].3,self.median_zone.right.unwrap(),self.vector[self.median_zone.right.unwrap()].3)
    }

    pub fn zones(&self) -> (usize,usize,usize,usize,usize,usize) {
        (self.left_zone.size,self.left_zone.index_set.len(),self.median_zone.size,self.median_zone.index_set.len(),self.right_zone.size,self.right_zone.index_set.len())
    }

    pub fn indecies(&self) -> (HashSet<usize>,HashSet<usize>,HashSet<usize>) {
        (self.left_zone.index_set.clone(), self.median_zone.index_set.clone(), self.right_zone.index_set.clone())
    }

    pub fn ordered_draw(&mut self) -> OrderedDraw {
        OrderedDraw::new(self)
    }

    pub fn consumed_draw(self) -> ProceduralDraw {
        ProceduralDraw::new(self)
    }

    pub fn procedural_draw(&mut self) -> OrderedDraw {
        OrderedDraw::new(self)
    }

    pub fn ordered_mad(&mut self,draw_order: &Vec<usize>) -> Vec<(f64,f64)> {



        let mut meds_mads = Vec::with_capacity(draw_order.len());
        for draw in draw_order {
            self.pop(*draw);
            meds_mads.push((self.median(),self.mad()))
        }

        meds_mads
    }

    pub fn give_draw_order(&self) -> Vec<usize> {
        self.vector.draw_order.clone()
    }

    pub fn give_dropped_order(&self) -> Vec<usize> {
        self.vector.dropped_draw_order()
    }

    pub fn between_original(&self, begin:usize,end:usize) -> usize {
        let mut number = 0;
        if self.drop.bool() && (self.vector.drop_set.contains(&begin) || self.vector.drop_set.contains(&end) || (self.vector[begin].3 <= 0. && self.vector[end].3 >= 0.)) {
            if self.vector[end].3 > 0. {
                number = self.vector.crawl_left(end).enumerate().find(|x| (x.1).1 < 0).unwrap_or((1,&(0,0,0,0.,0))).0 + 1;
            }
            else if self.vector[begin].3 < 0. {
                number = self.vector.crawl_right(begin).enumerate().find(|x| (x.1).1 > 0).unwrap_or((1,&(0,0,0,0.,0))).0 + 1;
            }
        }
        else {
            number = self.vector[end].2 - self.vector[begin].2;
        }
        number
    }

    pub fn crawl_between(&self, begin:usize,end:usize) -> usize {
        for (count,sample) in self.vector.crawl_right(begin).enumerate() {
            if sample.1 == end {
                return count
            }
        }
        return 0
    }

    pub fn log_odds(&self, begin:usize,end:usize) -> f64 {
        let interval = self.between_original(begin,end) as f64;
        (interval / (self.vector.len() as f64 - interval)).log10()
    }

    pub fn draw_values(&self) -> Vec<f64> {
        self.vector.iter_full().map(|x| x.3.clone()).collect()
    }

    pub fn draw_ordered_values(&self) -> Vec<f64> {
        self.vector.iter_ordered()
    }

    pub fn split_indecies(&self, split: &f64) -> (Vec<usize>,Vec<usize>) {
        let index = self.vector.drop_skip().enumerate().find(|x| (x.1).3 > *split).unwrap_or((0,&(0,0,0,0.,0))).0;
        let draw_order = self.vector.dropped_draw_order();
        (draw_order[..index].to_vec(),draw_order[index..].to_vec())
    }

    pub fn derive(&self, indecies:&[usize],) -> RankVector {

        let vector = self.vector.derive(indecies);

        let empty_backup = RawVector::raw_vector(&Vec::new());

        let length = vector.len();

        let zones = RankVector::empty_zones();

        let mut derived = RankVector{

            length: length,

            backup: false,

            backup_vector: empty_backup,
            backup_left: zones.0.clone(),
            backup_median: zones.1.clone(),
            backup_right: zones.2.clone(),

            vector: vector,

            draw_order: (0..length).collect(),

            drop: self.drop,
            num_dropped: 0,

            left_zone:zones.0,
            median_zone:zones.1,
            right_zone:zones.2,


            feature_name: self.feature_name.clone(),
            // sample_names: new_sample_names
        };

        derived.drop();

        derived.initialize();
        derived.set_boundaries();

        if self.backup {
            derived.backup();
        }

        derived

    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn index() {

    }


}

#[derive(Serialize,Deserialize,Debug,Clone)]
pub struct RankVector {
    pub vector: RawVector,

    backup: bool,
    backup_vector: RawVector,
    backup_left: LeftZone,
    backup_median: MedianZone,
    backup_right: RightZone,

    draw_order : Vec<usize>,

    pub left_zone: LeftZone,
    pub median_zone: MedianZone,
    pub right_zone: RightZone,

    length: usize,
    drop: DropMode,
    num_dropped: usize,

    feature_name: String,

}

impl MedianZone {

    fn expand_left(&mut self, raw_vector: &RawVector) -> Option<usize> {
        if let Some(left) = self.left {
            if let Some(new_left) = raw_vector.left(left) {
                self.size += 1;
                self.index_set.insert(new_left.1);
                self.left = Some(new_left.1);
                return self.left
            };
        };
        return None
    }

    fn expand_right(&mut self, raw_vector: &RawVector) -> Option<usize> {
        if let Some(right) = self.right {
            if let Some(new_right) = raw_vector.right(right) {
                self.size += 1;
                self.index_set.insert(new_right.1);
                self.right = Some(new_right.1);
                return self.right
            }
        }
        return None
    }

    fn contract_left(&mut self, raw_vector: &RawVector) -> Option<usize> {
        if let Some(left) = self.left {
            if let Some(new_left) = raw_vector.right(left) {
                self.size -= 1;
                self.index_set.remove(&left);
                self.left = Some(new_left.1);
                return self.left
            }
        }
        return None

    }

    fn contract_right(&mut self, raw_vector: &RawVector) -> Option<usize> {
        if let Some(right) = self.right {
            if let Some(new_right) = raw_vector.left(right) {
                self.size -= 1;
                self.index_set.remove(&right);
                self.right = Some(new_right.1);
                return self.right
            }
        }
        return None
    }

    fn mad(&self, raw_vector: &RawVector) -> f64 {
        if raw_vector.len() > 1 {

            let left = raw_vector[self.left.unwrap()];
            let right = raw_vector[self.right.unwrap()];

            let inner_left = raw_vector.right(left.1).unwrap();
            let inner_right = raw_vector.left(right.1).unwrap();

            let median = self.dead_center.median();

            let mut distance_to_median = vec![(left.3 - median).abs(), (inner_left.3 - median).abs(), (inner_right.3 - median).abs(), (right.3 - median).abs()];

            // println!("MAD debug");
            // println!("{}",median);
            // println!("{:?},{:?}",left,right);
            // println!("{:?}",distance_to_median);

            distance_to_median.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));
            distance_to_median.reverse();

            if raw_vector.len%2 == 0 {
                return (distance_to_median[0] + distance_to_median[1]) / 2.
            }
            return distance_to_median[0]
        }
        else {
            return 0.
        }
    }

    fn reset_by_reference(&mut self, backup: &MedianZone) {

        self.size = backup.size;
        self.dead_center.reset_by_reference(&backup.dead_center);
        self.left = backup.left;
        self.right = backup.right;

        self.index_set.clear();
        self.index_set.extend(backup.index_set.iter());
    }


}

#[derive(Serialize,Deserialize,Debug,Clone)]
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

        if length < 1 {
            return DeadCenter{left:None,right:None}
        }

        let mut left_zone= -1i32;
        let mut right_zone = (length) as i32;

        let mut left = None;
        let mut right = None;

        for sample in raw.left_to_right() {

            // println!("Center debug: {:?},{:?}", left,right);
            // println!("Center debug: {},{}", left_zone, right_zone);

            right = Some(sample.clone());
            right_zone -= 1;

            if left_zone == right_zone {
                break
            }

            left = right;
            left_zone += 1;

            if left_zone == right_zone {
                break
            }

        }

        DeadCenter {
            left: left,
            right: right,
        }
    }

    pub fn move_left(&mut self, raw_vector: &RawVector) {
        if self.left == self.right {
            if let Some(left) = self.left{
                self.left = raw_vector.left(left.1);
            }
        }
        else {
            self.right = self.left;
        }
    }

    pub fn move_right (&mut self, raw_vector: &RawVector) {
        if self.left == self.right {
            if let Some(right) = self.right {
                self.right = raw_vector.right(right.1);
            }
        }
        else {
            self.left = self.right;
        }
    }

    pub fn re_center(&mut self, target:&usize, raw_vector: &RawVector) -> (f64,f64) {

        // println!("Re-center debug 1: {},{}", self.left.unwrap_or((0,0,0,0.,0)).1,self.right.unwrap_or((0,0,0,0.,0)).1);

        let removed = raw_vector[*target];

        if raw_vector.len() < 1 {
            return (0.,0.)
        }

        let old_median = self.median();

        if let (Some(left),Some(right)) = (self.left, self.right) {

            if removed.2 > left.2 {
                    self.move_left(raw_vector);
            }
            else if removed.2 < right.2 {
                    self.move_right(raw_vector);
            }
            else {
                self.left = raw_vector.left(removed.1);
                self.right = raw_vector.right(removed.1);
            }
        }
        else {
            panic!("Dead center de-synced");
        }

        // println!("Re-center debug 2: {},{}", self.left.unwrap_or((0,0,0,0.,0)).1,self.right.unwrap_or((0,0,0,0.,0)).1);

        let new_median = self.median();

        (old_median,new_median)
    }

    pub fn median(&self) -> f64 {
        (self.left.unwrap_or((0,0,0,0.,0)).3 + self.right.unwrap_or((0,0,0,0.,0)).3)/2.
    }

    pub fn reset_by_reference(&mut self,backup:&DeadCenter) {
        self.left = backup.left;
        self.right = backup.right;
    }

}

#[derive(Serialize,Deserialize,Debug,Clone)]
pub struct DeadCenter {
    left: Option<(usize,usize,usize,f64,usize)>,
    right: Option<(usize,usize,usize,f64,usize)>
}

impl LeftZone {
    fn expand(&mut self,raw_vector:&RawVector) -> Option<usize> {
        if let Some(right) = self.right {
            if let Some(new_right) = raw_vector.right_ind(right) {
                self.size += 1;
                self.index_set.insert(new_right);
                self.right = Some(new_right);
                return self.right
            }
            else {
                return None
            }
        }
        else {
            if let Some(new_right) = raw_vector.first() {
                self.size += 1;
                self.index_set.insert(new_right.1);
                self.right = Some(new_right.1);
                return self.right
            }
            else {
                return None
            }
        }
    }

    fn contract(&mut self, raw_vector: &RawVector) -> Option<usize> {
        if let Some(right) = self.right{
            let new_right = raw_vector.left_ind(right);
            // println!("Left contract debug");
            // println!("{:?},{:?}", right,new_right);
            self.size -= 1;
            self.index_set.remove(&right);
            self.right = new_right;
            return self.right
        }
        None
    }

    fn reset_by_reference(&mut self, backup: &LeftZone) {
        self.index_set.clear();
        self.index_set.extend(backup.index_set.iter());

        self.size = backup.size;
        self.right = backup.right;
    }

}

#[derive(Serialize,Deserialize,Debug,Clone)]
pub struct LeftZone{
    pub size: usize,
    // left: Option<(usize,usize,usize,f64,usize)>,
    right: Option<usize>,
    pub index_set: HashSet<usize>
}

impl RightZone {
    fn expand(&mut self,raw_vector:&RawVector) -> Option<usize> {
        if let Some(left) = self.left {
            if let Some(new_left) = raw_vector.left_ind(left) {
                self.size += 1;
                self.index_set.insert(new_left);
                self.left = Some(new_left);
                return self.left
            }
            else {
                return None
            }
        }
        else {
            if let Some(new_left) = raw_vector.last() {
                self.size += 1;
                self.left = Some(new_left.1);
                self.index_set.insert(new_left.1);
                return self.left
            }
            else {
                return None
            }
        }
    }

    fn contract(&mut self, raw_vector: &RawVector) -> Option<usize> {
        if let Some(left) = self.left {
            let new_left = raw_vector.right_ind(left);
            self.size -= 1;
            self.index_set.remove(&left);
            self.left = new_left;
            return self.left
        }
        None
    }

    fn reset_by_reference(&mut self, backup: &RightZone) {

        self.index_set.clear();
        self.index_set.extend(backup.index_set.iter());

        self.size = backup.size;
        self.left = backup.left;
    }
}

#[derive(Serialize,Deserialize,Debug,Clone)]
pub struct RightZone {
    pub size: usize,
    left: Option<usize>,
    // right: Option<&'a (usize,usize,usize,f64,usize)>,
    pub index_set: HashSet<usize>,
}


impl<'a> OrderedDraw<'a> {
    pub fn new(vector : &'a mut RankVector) -> OrderedDraw<'a> {
        vector.backup();
        OrderedDraw{vector: vector, index:0}
    }
}

impl<'a> Iterator for OrderedDraw<'a> {
    type Item = (f64,f64);

    fn next(&mut self) -> Option<(f64,f64)> {
        if self.index >= self.vector.draw_order.len() {
            self.vector.reset();
            return None
        }
        let draw = self.vector.draw_order[self.index];
        // println!("Popping: {}",draw);
        // println!("Remaining samples: {}", self.vector.vector.len());
        // println!("Zones: {:?}", self.vector.zones());
        // println!("Boundaries: {:?}", self.vector.boundaries());
        // println!("Zone boundaries: {:?},{:?}", self.vector.left_zone.right.unwrap_or(0), self.vector.right_zone.left.unwrap_or(0));
        // println!("Indecies: {:?}", self.vector.indecies());
        // println!("{:?}",self.vector.pop(draw));
        self.vector.pop(draw);

        // println!("{:?}", self.vector.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());

        self.index +=1;

        Some((self.vector.median_zone.dead_center.median(),self.vector.median_zone.mad(&self.vector.vector)))
    }
}

pub struct OrderedDraw<'a>{
    pub vector: &'a mut RankVector,
    index: usize,
}

impl ProceduralDraw {

    pub fn new(vector :RankVector) -> ProceduralDraw {
        ProceduralDraw{vector: vector, index:0}
    }

    pub fn next(&mut self,target:usize) -> Option<(f64,f64)> {

        // let start_time = time::PreciseTime::now();

        if self.vector.vector.len() < 1{
            return None
        }

        let state = (self.vector.median_zone.dead_center.median(),self.vector.median_zone.mad(&self.vector.vector));
        // println!("Popping: {}",target);
        // println!("Remaining samples: {}", self.vector.vector.len());
        // println!("Zones: {:?}", self.vector.zones());
        // println!("Boundaries: {:?}", self.vector.boundaries());
        // println!("Zone boundaries: {:?},{:?}", self.vector.left_zone.right.unwrap_or(0), self.vector.right_zone.left.unwrap_or(0));
        // println!("Indecies: {:?}", self.vector.indecies());
        // println!("{:?}",self.vector.pop(target).3);
        self.vector.pop(target);

        // println!("{:?}", self.vector.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());

        self.index +=1;

        // let end_time = time::PreciseTime::now();
        //
        // if self.index == 300 {
        //     println!("Time for a single iter: {}ns", start_time.to(end_time).num_nanoseconds().unwrap_or(-1));
        // }

        Some(state)
    }
}

#[derive(Clone)]
pub struct ProceduralDraw{
    pub vector: RankVector,
    index: usize,
}



#[cfg(test)]
mod rank_vector_tests {

    use super::*;
    use std::f64::NAN;

    fn slow_median(input: &RawVector) -> f64 {
        let values: Vec<f64> = input.left_to_right().map(|x| x.3).collect();
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

    fn slow_mad(input: &RawVector) -> f64 {
        let values: Vec<f64> = input.left_to_right().map(|x| x.3).collect();
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

    #[test]
    fn create_trivial() {
        let mut vector = RankVector::new(&vec![],"".to_string(),DropMode::No);
        vector.drop_zeroes();
        vector.initialize();
        vector.set_boundaries();
    }

    #[test]
    fn create_very_simple_drop() {
        let mut vector = RankVector::new(&vec![0.],"test".to_string(),DropMode::Zeros);
        vector.drop_zeroes();
    }

    #[test]
    fn create_very_simple_initialize() {
        let mut vector = RankVector::new(&vec![0.],"test".to_string(),DropMode::Zeros);
        vector.drop_zeroes();
        vector.initialize();
        vector.set_boundaries();
    }

    #[test]
    fn create_very_simple_nan_drop() {
        let mut vector = RankVector::new(&vec![NAN],"test".to_string(), DropMode::NaNs);
        vector.drop_zeroes();
        vector.initialize();
        vector.set_boundaries();
    }

    #[test]
    fn create_simple() {
        let mut vector = RankVector::new(&vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],"test".to_string(),DropMode::Zeros);
        vector.drop_zeroes();
        vector.initialize();
        vector.set_boundaries();
        assert_eq!(vector.draw_values(),vec![-3.,-2.,-1.,0.,5.,10.,15.,20.]);
        assert_eq!(vector.vector.drop_skip().map(|x| x.3.clone()).collect::<Vec<f64>>(),vec![-3.,-2.,-1.,5.,10.,15.,20.])
    }

    #[test]
    fn create_repetitive() {
        let mut vector = RankVector::new(&vec![0.,0.,0.,-5.,-5.,-5.,10.,10.,10.,10.,10.],"test".to_string(),DropMode::Zeros);
        vector.drop_zeroes();
        vector.initialize();
        vector.set_boundaries();
        assert_eq!(vector.draw_values(),vec![-5.,-5.,-5.,0.,0.,0.,10.,10.,10.,10.,10.]);
        assert_eq!(vector.vector.drop_skip().map(|x| x.3.clone()).collect::<Vec<f64>>(),vec![-5.,-5.,-5.,10.,10.,10.,10.,10.])
    }

    #[test]
    fn sequential_mad_simple() {
        let mut vector = RankVector::new(&vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],"test".to_string(),DropMode::Zeros);
        vector.drop_zeroes();
        vector.initialize();
        vector.set_boundaries();

        let mut vm = vector.clone();


        for draw in vector.draw_order {
            println!("{:?}",vm.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
            println!("Median:{},{}",vm.median(),slow_median(&vm.vector));
            println!("MAD:{},{}",vm.mad(),slow_mad(&vm.vector));
            println!("{:?}",vm.pop(draw));
            println!("{:?}",vm.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
            println!("Median:{},{}",vm.median(),slow_median(&vm.vector));
            println!("MAD:{},{}",vm.mad(),slow_mad(&vm.vector));
            assert_eq!(vm.median(),slow_median(&vm.vector));
            assert_eq!(vm.mad(),slow_mad(&vm.vector));
        }

    }

    #[test]
    fn sequential_mad_simple_nan() {
        let mut vector = RankVector::new(&vec![10.,-3.,NAN,5.,-2.,-1.,15.,20.],"test".to_string(),DropMode::NaNs);
        vector.drop_zeroes();
        vector.initialize();
        vector.set_boundaries();

        let mut vm = vector.clone();


        for draw in vector.draw_order {
            println!("{:?}",vm.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
            println!("Median:{},{}",vm.median(),slow_median(&vm.vector));
            println!("MAD:{},{}",vm.mad(),slow_mad(&vm.vector));
            println!("{:?}",vm.pop(draw));
            println!("{:?}",vm.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
            println!("Median:{},{}",vm.median(),slow_median(&vm.vector));
            println!("MAD:{},{}",vm.mad(),slow_mad(&vm.vector));
            assert_eq!(vm.median(),slow_median(&vm.vector));
            assert_eq!(vm.mad(),slow_mad(&vm.vector));
        }

    }

    // #[test]
    // fn odds_ratio() {
    //     let mut vector = RankVector::new(&vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],"test".to_string());
    //     vector.drop_zeroes();
    //     vector.initialize();
    //     vector.set_boundaries();
    //
    // }



}
