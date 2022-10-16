use std::collections::HashSet;

use super::*;

use datafrog::{Iteration, Relation, Variable};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Link {
    from: NodeId,
    output_name: String,
    to: NodeId,
    input_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceSource {
    res_id: u32,
    node: NodeId,
    output_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceHandle {
    res_id: u32,
    handle_id: u32,
    parent: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Socket {
    node: NodeId,
    socket_name: String,
}

pub fn datafrog_test(graph: &Graph) -> Result<()> {
    let mut links: Vec<Link> = Vec::new();
    let mut resource_ids: Vec<u32> = Vec::new();

    let mut iteration = Iteration::new();

    let links_var = iteration.variable::<Link>("links");
    let resources_var = iteration.variable::<u32>("resources");
    // let input_sockets = iteration.variable::<Socket>("input_sockets");
    // let output_sockets = iteration.variable::<Socket>("output_sockets");

    // let rel = Relation::from_map(input, logic)

    let mut sources_set = HashSet::new();
    let mut handles_set = HashSet::new();
    let mut links_set = HashSet::new();

    let mut next_res_id = 0;
    let mut next_handle_id = 0;

    for node in graph.nodes.iter() {
        for (out_name, out) in node.outputs.iter() {
            let from = node.id;
            let output_name = out_name.to_string();

            if let OutputSource::PrepareAllocation { .. } = &out.source {
                let res_id = next_res_id;
                let res_source = ResourceSource {
                    res_id,
                    node: node.id,
                    output_name: out_name.to_string(),
                };
                next_res_id += 1;
                sources_set.insert(res_source);

                let handle_id = next_handle_id;

                let handle = ResourceHandle {
                    res_id,
                    handle_id,
                    parent: None,
                };
                handles_set.insert(handle);
                next_handle_id += 1;
            }


            if let Some((other, in_name)) = out.link.as_ref() {
                let link = Link {
                    from: node.id,
                    output_name: out_name.to_string(),
                    to: *other,
                    input_name: in_name.to_string(),
                };

                // let output = Socket {
                //     node: link.from,
                //     socket_name: link.output_name.clone(),
                // };

                // let input = Socket {
                //     node: link.to,
                //     socket_name: link.input_name.clone(),
                // };

                links_set.insert(link);
                // output_sockets.extend(Some(output));
                // input_sockets.extend(Some(input));
            }
        }
    }

    links_var.extend(links_set.iter().cloned());

    while iteration.changed() {
        // 


    }

    Ok(())
}
