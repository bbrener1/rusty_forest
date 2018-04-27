// use std::f64;
//
//
// struct RawVector {
//     length: usize,
//     nodes: Vec<Node>
// }
//
// struct Node {
//     data: f64,
//     index: usize,
//     previous: usize,
//     next: usize,
// }
//
// impl RawVector {
//
//     pub fn empty() -> RawVector {
//
//         let head = Node {
//             data: f64::NAN,
//             index: 0,
//             previous: 0,
//             next: 1,
//         };
//         let tail = Node {
//             data: f64::NAN,
//             index: 1,
//             previous: 1,
//             next: 0
//         };
//
//         RawVector {
//             length: 0,
//             nodes: vec![head,tail],
//         }
//
//     }
//
//     pub fn link(&Vec<f64>) -> RawVector {
//
//         let mut vector = Vec::with_capacity(in_vec.len());
//         let mut draw_order = Vec::with_capacity(in_vec.len());
//
//         let (clean_vector,dirty_set) = sanitize_vector(in_vec);
//
//         let mut sorted_invec = clean_vector.into_iter().enumerate().collect::<Vec<(usize,f64)>>();
//         sorted_invec.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//
//         for (i,sample) in sorted_invec.iter().enumerate() {
//
//             // eprintln!("{}", i);
//
//             if i == 0 {
//                 vector.push((sample.0,sample.0,i,sample.1,sample.0));
//                 draw_order.push(sample.0);
//             }
//             else if i == (sorted_invec.len() - 1) {
//                 vector[i-1].4 = sample.0;
//                 vector.push((sorted_invec[i-1].0,sample.0,i,sample.1,sample.0));
//                 draw_order.push(sample.0);
//             }
//             if {i != 0} && {i < (sorted_invec.len()-1)} {
//                 vector[i-1].4 = sample.0;
//                 vector.push((sorted_invec[i-1].0,sample.0,i,sample.1,sample.0));
//                 draw_order.push(sample.0);
//             }
//         }
//
//         let first: usize;
//         let last: usize;
//         let length: usize;
//
//         if vector.len() > 0 {
//             first = vector[0].1;
//             last = vector[vector.len()-1].1;
//             length = vector.len();
//         }
//         else {
//             first = 0;
//             last = 0;
//             length = 0;
//         }
//
//
//         vector.sort_by_key(|x| x.1);
//
//         let drop_set = HashSet::new();
//
//         RawVector {
//             first: Some(first),
//             len: length,
//             last: Some(last),
//             vector: vector,
//             draw_order: draw_order,
//             drop_set: drop_set,
//             dirty_set: dirty_set,
//             drop: false
//         }
//
//
//     }
//
//     pub fn pop(&mut self, nominal_index: usize) -> f64 {
//
//         let Node{data: data,index: index,previous: mut previous,next: mut next} = self.nodes[nominal_index + 2];
//
//         self.nodes[previous].next = next;
//         self.nodes[next].previous = previous;
//
//         previous = index;
//         next = index;
//
//         data
//     }
//
//     pub fn push(&mut self, data: f64) {
//
//         let node = Node{
//             data: data,
//             index: self.nodes.len(),
//             previous: self.nodes[1].previous,
//             next: 1,
//         };
//
//         let previous_tail = self.nodes[1].previous;
//         self.nodes[1].previous = node.index;
//         self.nodes[previous_tail].next = node.index;
//
//     }
//
//     pub fn len(&self) -> usize {
//         self.length
//     }
//
//
//
// }
//
// pub fn sanitize_vector(in_vec:&Vec<f64>) -> (Vec<f64>,HashSet<usize>) {
//
//     (
//         in_vec.iter().map(|x| if !x.is_normal() {0.} else {*x}).collect(),
//
//         in_vec.iter().enumerate().filter(|x| !x.1.is_normal()).map(|x| x.0).collect()
//     )
//
// }
