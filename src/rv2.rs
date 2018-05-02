use std::f64;
use std::collections::HashSet;
use std::cmp::Ordering;

struct RawVector {
    length: usize,
    nodes: Vec<Node>,
    drop_set: HashSet<usize>,
    dirty_set: HashSet<usize>,
    drop:bool
}

struct Node {
    data: f64,
    index: usize,
    previous: usize,
    next: usize,
}

impl Node {

    pub fn head() -> Node {
        Node {
            data: f64::NAN,
            index: 0,
            previous: 0,
            next: 1,
        }
    }

    pub fn tail() -> Node {
        Node {
            data: f64::NAN,
            index: 1,
            previous: 1,
            next: 0
        }
    }

}

impl RawVector {

    pub fn empty() -> RawVector {

        let head = Node {
            data: f64::NAN,
            index: 0,
            previous: 0,
            next: 1,
        };
        let tail = Node {
            data: f64::NAN,
            index: 1,
            previous: 1,
            next: 0
        };

        RawVector {
            length: 0,
            nodes: vec![head,tail],
            drop_set: HashSet::with_capacity(0),
            dirty_set: HashSet::with_capacity(0),
            drop: false,
        }

    }

    pub fn link(in_vec: &Vec<f64>) -> RawVector {

        let mut vector: Vec<Node> = Vec::with_capacity(in_vec.len());
        // let mut draw_order = Vec::with_capacity(in_vec.len());

        let (clean_vector,dirty_set) = sanitize_vector(in_vec);

        let mut sorted_invec = clean_vector.into_iter().enumerate().collect::<Vec<(usize,f64)>>();

        sorted_invec.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));

        vector.push(Node::head());
        vector.push(Node::tail());

        let mut previous_node_index = 0;

        for (ranking,&(index,data)) in sorted_invec.iter().enumerate() {

            let node = Node {
                data: data,
                index: index,
                previous: previous_node_index,
                next: 1
            };

            vector[previous_node_index].next = index;

            previous_node_index = index;

            vector.push(node);

        }

        // let draw_order = vector.iter().map(|x| x.index).collect();

        vector.sort_unstable_by_key(|x| x.index);

        let drop_set = HashSet::new();

        RawVector {
            length: vector.len(),
            nodes: vector,
            drop_set: drop_set,
            dirty_set: dirty_set,
            drop: false
        }


    }

    pub fn pop(&mut self, nominal_index: usize) -> f64 {

        let Node{data: data,index: index,previous: mut previous,next: mut next} = self.nodes[nominal_index + 2];

        self.nodes[previous].next = next;
        self.nodes[next].previous = previous;

        previous = index;
        next = index;

        data
    }

    fn pop_internal(&mut self, passed_index: usize) -> f64 {

        let Node{data: data,index: index,previous: mut previous,next: mut next} = self.nodes[passed_index];

        self.nodes[previous].next = next;
        self.nodes[next].previous = previous;

        previous = index;
        next = index;

        data
    }

    pub fn len(&self) -> usize {
        self.length
    }

    fn g_left(&self, target:usize) -> &Node {
        &self.nodes[self.nodes[target].previous]
    }

    fn g_right(&self, target:usize) -> &Node {
        &self.nodes[self.nodes[target].next]
    }

    fn left(&self, target:usize) -> Option<&Node> {
        if target != 0 && target != 1 {
            Some(&self.nodes[self.nodes[target].previous])
        }
        else { return None }
    }

    fn right(&self, target:usize) -> Option<&Node> {
        if target != 0 && target != 1 {
            Some(&self.nodes[self.nodes[target].next])
        }
        else { return None }
    }

    fn left_to_right(&self) -> Vec<&Node> {
        let mut output = Vec::with_capacity(self.len());
        let mut index = 0;
        for i in 0..self.len() {
            index = self.nodes[index].next;
            output.push(&self.nodes[index]);
        }
        output
    }

    pub fn drop_f(&mut self, f: f64) {
        let drop_set: HashSet<usize> = self.left_to_right().iter().filter(|x| x.data == f).map(|x| x.index).collect();
        for index in &drop_set {
            self.pop_internal(*index);
        };
        self.drop_set = drop_set;

    }


}





pub fn sanitize_vector(in_vec:&Vec<f64>) -> (Vec<f64>,HashSet<usize>) {

    (
        in_vec.iter().map(|x| if !x.is_normal() {0.} else {*x}).collect(),

        in_vec.iter().enumerate().filter(|x| !x.1.is_normal()).map(|x| x.0).collect()
    )

}


// pub fn push(&mut self, data: f64) {
//
//     let node = Node{
//         data: data,
//         index: self.nodes.len(),
//         previous: self.nodes[1].previous,
//         next: 1,
//     };
//
//     let previous_tail = self.nodes[1].previous;
//     self.nodes[1].previous = node.index;
//     self.nodes[previous_tail].next = node.index;
//
// }

//
// impl<'a> LeftVectCrawler<'a> {
//
//     #[inline]
//     fn new(input: &'a Vec<Node>, first: usize) -> LeftVectCrawler {
//         LeftVectCrawler{vector: input, index: first}
//     }
// }
//
// impl<'a> Iterator for LeftVectCrawler<'a> {
//     type Item = &'a Node;
//
//     #[inline]
//     fn next(&mut self) -> Option<&'a Node> {
//
//         if let Some(index) = self.index {
//             if self.vector[index].1 == self.vector[index].0 {
//                 self.index = None;
//                 return Some(& self.vector[index])
//             }
//             let current = & self.vector[index];
//             self.index = Some(current.0);
//             return Some(& current)
//         }
//         else{
//             return None
//         }
//
//     }
// }
//
// pub struct LeftVectCrawler<'a> {
//     vector: &'a Vec<Node>,
//     index: usize
// }
