#include "connectivity.h"

namespace libtensor {

namespace expr {

const char* connectivity::k_clazz = "connectivity";

connectivity::connectivity(const expr_tree& tree)
{
    expr_tree::node_id_t root_id = tree.get_root();
    const node& root = tree.get_vertex(root_id);

    if(!root.check_type<node_assign>())
    {
        throw bad_parameter(g_ns,k_clazz,"batch_provider(...)",__FILE__, __LINE__,
            "Invalid root node");
    }

    expr_tree::node_id_t op_node_id = tree.get_edges_out(root_id)[1];
    expr_tree::edge_list_t op_children = tree.get_edges_out(op_node_id);

    const node& op_node = tree.get_vertex(op_node_id);
    if(op_node.check_type<node_add>())
    {
        m_conn.resize(1+op_children.size(),std::vector<idx_pair_list>(root.get_n()));
        for(size_t tensor_idx = 0; tensor_idx < m_conn.size(); ++tensor_idx)
        {
            for(size_t subspace_idx = 0; subspace_idx < m_conn[tensor_idx].size(); ++subspace_idx)
            {
                for(size_t other_tensor_idx = 0; other_tensor_idx < m_conn.size(); ++other_tensor_idx)
                {
                    if(other_tensor_idx == tensor_idx) continue;
                    m_conn[tensor_idx][subspace_idx].push_back(idx_pair(other_tensor_idx,subspace_idx));
                }
            }
        }
    }
    else
    {
        throw bad_parameter(g_ns,k_clazz,"connectivity(...)",__FILE__, __LINE__,
            "Unsupported node type");
    }
}

} // namespace expr

} // namespace libtensor
