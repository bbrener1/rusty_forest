use tree::Tree;
use rank_table::RankTable;
use node::Node;



pub fn fragment_nodes(node: &mut Node,prototype_node: &mut Node) -> Vec<Node> {

    Vec::new()
}

// pub fn cascading_interaction<'a>(node:&'a mut Node,parents:&mut Vec<&'a RankTable>) -> Vec<(&'a str,Vec<(&'a str,Vec<(&'a str,f64)>)>)> {
//
//     parents.push(&node.rank_table);
//
//     let mut interactions = Vec::new();
//
//     for child in node.children.iter_mut() {
//         interactions.extend(cascading_interaction(child, parents));
//     }
//
//
//
//     interactions
// }
