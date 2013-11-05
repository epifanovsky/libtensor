#include "sparse_block_tree_node.h"
#include "../defs.h"
#include "../exception.h"
#include <algorithm>

namespace libtensor { 

namespace impl {

sparse_block_tree_leaf_node::sparse_block_tree_leaf_node(const std::vector<size_t>& key,const size_t idx)
{
    m_keys.push_back(key[idx]);
    m_values.push_back(0);
}

void sparse_block_tree_leaf_node::search(const std::vector<size_t>& key,std::vector<size_t>& positions,const size_t idx) const
{
    std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),key[idx]);
    if(cur_pos_it == m_keys.end() || *cur_pos_it != key[idx])
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","search(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    size_t cur_pos = distance(m_keys.begin(),cur_pos_it);
    positions[idx] = cur_pos;
}

const block_list& sparse_block_tree_leaf_node::get_sub_key_block_list(const std::vector<size_t>& sub_key,size_t cur_idx) const
{ 
    return m_keys; 
}

void sparse_block_tree_leaf_node::push_back(const std::vector<size_t>& key,size_t cur_idx)
{
    m_keys.push_back(key[cur_idx]);
    m_values.push_back(0);
}

void sparse_block_tree_branch_node::search(const std::vector<size_t>& key,std::vector<size_t>& positions,const size_t idx) const
{
    std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),key[idx]);
    if(cur_pos_it == m_keys.end() || *cur_pos_it != key[idx])
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","search(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    size_t cur_pos = distance(m_keys.begin(),cur_pos_it);
    positions[idx] = cur_pos;
    m_children[cur_pos]->search(key,positions,idx+1);
}


sparse_block_tree_branch_node::sparse_block_tree_branch_node(const std::vector<size_t>& key,size_t cur_idx,size_t order) : m_order(order)
{
    m_keys.push_back(key[cur_idx]);
    if(m_order == 2)
    {
        m_children.push_back(new sparse_block_tree_leaf_node(key,cur_idx+1));
    }
    else
    {
        m_children.push_back(new sparse_block_tree_branch_node(key,cur_idx+1,order-1));
    }
}


sparse_block_tree_branch_node::sparse_block_tree_branch_node(const sparse_block_tree_branch_node& rhs) : m_children(rhs.m_children.size()), m_order(rhs.m_order)
{
    for(size_t i = 0; i < rhs.m_keys.size(); ++i)
    {
        m_keys.push_back(rhs.m_keys[i]);
        m_children[i] = rhs.m_children[i]->clone(); 
    }
}

sparse_block_tree_branch_node& sparse_block_tree_branch_node::operator=(const sparse_block_tree_branch_node& rhs)
{
    m_children.resize(rhs.m_children.size());
    m_keys.resize(rhs.m_keys.size());
    m_order = rhs.m_order;

    for(size_t i = 0; i < rhs.m_keys.size(); ++i)
    {
        m_keys[i] = rhs.m_keys[i];
        m_children[i] = rhs.m_children[i]->clone();
    }
}

const block_list& sparse_block_tree_branch_node::get_sub_key_block_list(const std::vector<size_t>& sub_key,size_t cur_idx) const
{
    //Find the position of the current key in this node's list of keys
    size_t cur_val = sub_key[cur_idx];
    block_list::const_iterator cur_pos = std::lower_bound(m_keys.begin(),m_keys.end(),cur_val);
    if(cur_pos == m_keys.end() || *cur_pos != cur_val)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","get_sub_key_block_list(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    else
    {
        //Done?
        if(cur_idx == (sub_key.size() - 1))
        {
            return m_children[distance(m_keys.begin(),cur_pos)]->m_keys;
        }
        return m_children[distance(m_keys.begin(),cur_pos)]->get_sub_key_block_list(sub_key,cur_idx+1);
    }
}

//Must be templated so that it can use the sequence length of the parent
void sparse_block_tree_branch_node::push_back(const std::vector<size_t>& key,size_t cur_idx)
{
    size_t cur_key = key[cur_idx];
    if(m_keys.back() != cur_key)
    {
        m_keys.push_back(cur_key);
        if(m_order == 2)
        {
            m_children.push_back(new sparse_block_tree_leaf_node(key,cur_idx+1));
        }
        else
        {
            m_children.push_back(new sparse_block_tree_branch_node(key,cur_idx+1,m_order - 1));
        }
    }
    else
    {
        m_children.back()->push_back(key,cur_idx+1);
    }
}

sparse_block_tree_branch_node::~sparse_block_tree_branch_node()
{
    for(size_t i = 0; i < m_children.size(); ++i)
    {
        delete m_children[i];
    }
}

} // namespace impl

} // namespace libtensor
