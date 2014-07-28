#include "connectivity.h"

using namespace std;

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

    expr_tree::node_id_t n_op_id = tree.get_edges_out(root_id)[1];
    expr_tree::edge_list_t op_children = tree.get_edges_out(n_op_id);

    const node& n_op = tree.get_vertex(n_op_id);
    size_t NC = root.get_n();
    if(n_op.check_type<node_add>())
    {
        m_conn.resize(1+op_children.size(),std::vector<idx_pair_list>(NC));
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
    else if(n_op.check_type<node_contract>())
    {
        const node_contract& n_contr = dynamic_cast< const node_contract& >(n_op);
        multimap<size_t,size_t> contr_map = n_contr.get_map();

        multimap<size_t,size_t> contr_inv;
        for(multimap<size_t,size_t>::const_iterator it = contr_map.begin(); it != contr_map.end(); ++it)
        {
            contr_inv.insert(idx_pair(it->second,it->first));
        }
        size_t NA = tree.get_vertex(op_children[0]).get_n();
        size_t NB = tree.get_vertex(op_children[1]).get_n();
        m_conn.push_back(std::vector<idx_pair_list>(NC));
        m_conn.push_back(std::vector<idx_pair_list>(NA));
        m_conn.push_back(std::vector<idx_pair_list>(NB));
        size_t c_sub_idx = 0;
        for(size_t input_sub_idx = 0; input_sub_idx < NA+NB; ++input_sub_idx)
        {
            if(input_sub_idx < NA)
            {
                multimap<size_t,size_t>::iterator it = contr_map.find(input_sub_idx);
                if(it == contr_map.end())
                {
                    m_conn[0][c_sub_idx].push_back(idx_pair(1,input_sub_idx));
                    m_conn[1][input_sub_idx].push_back(idx_pair(0,c_sub_idx));
                    ++c_sub_idx;
                }
                else
                {
                    m_conn[1][input_sub_idx].push_back(idx_pair(2,it->second - NA));
                    m_conn[2][it->second - NA].push_back(idx_pair(1,input_sub_idx));
                }
            }
            else if(input_sub_idx >= NA && (contr_inv.find(input_sub_idx) == contr_inv.end()))
            {
                m_conn[0][c_sub_idx].push_back(idx_pair(2,input_sub_idx - NA));
                m_conn[2][input_sub_idx - NA].push_back(idx_pair(0,c_sub_idx));
                ++c_sub_idx;
            }
        }
    }
    else if(n_op.check_type<node_transform_base>())
    {
        const node_transform_base& n_tf = dynamic_cast< const node_transform_base& >(n_op);
        m_conn.resize(1+op_children.size(),std::vector<idx_pair_list>(NC));
        vector<size_t> perm_entries = n_tf.get_perm();
        for(size_t i = 0; i < NC; ++i)
        {
            m_conn[0][i].push_back(idx_pair(1,perm_entries[i]));
            m_conn[1][perm_entries[i]].push_back(idx_pair(0,i));
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
