#include "sparse_block_tree_node.h"
#include "sparse_block_tree_iterator.h"
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

sparse_block_tree_iterator<true> sparse_block_tree_leaf_node::end() const
{
    return const_iterator(NULL,1);
}

size_t sparse_block_tree_leaf_node::search(const std::vector<size_t>& key,const size_t idx) const
{
    std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),key[idx]);
    if(cur_pos_it == m_keys.end() || *cur_pos_it != key[idx])
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","search(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    size_t cur_pos = distance(m_keys.begin(),cur_pos_it);
    return m_values[cur_pos];
}

const block_list& sparse_block_tree_leaf_node::get_sub_key_block_list(const std::vector<size_t>& sub_key,size_t cur_idx) const
{ 
    return m_keys; 
}

sparse_block_tree_iterator<true> sparse_block_tree_leaf_node::get_sub_key_begin_iterator_internal(const std::vector<size_t>& sub_key,const sparse_block_tree_node* root,std::vector<size_t>& displacement,const size_t cur_idx) const
{
    //If the sub key has already been found in full, we just start from the first key in this node
    if(cur_idx > (sub_key.size() - 1))
    {
        displacement[cur_idx] = 0;
    }
    else
    {
        std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),sub_key[cur_idx]);

        //If not found, return end()
        if(cur_pos_it == m_keys.end() || *cur_pos_it != sub_key[cur_idx])
        {
            return const_iterator(NULL,displacement.size(),displacement);
        }
        displacement[cur_idx] = std::distance(m_keys.begin(),cur_pos_it);
    }
    return const_iterator(root,displacement.size(),displacement);
}

sparse_block_tree_iterator<true> sparse_block_tree_leaf_node::get_sub_key_begin_iterator(const std::vector<size_t>& sub_key) const
{
    std::vector<size_t> displacement(1); 
    return get_sub_key_begin_iterator_internal(sub_key,this,displacement,0);
}

sparse_block_tree_iterator<true> sparse_block_tree_leaf_node::get_sub_key_end_iterator(const std::vector<size_t>& sub_key) const
{
    const_iterator sub_key_iter = get_sub_key_begin_iterator(sub_key);
    if(sub_key_iter == end())
    {
        throw bad_parameter(g_ns,"sparse_block_tree_leaf_node","get_sub_key_end_displacement(...)",
            __FILE__,__LINE__,"key not found"); 
    } 
    else
    {
        //Go to the next key
        std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),sub_key.back());
        ++cur_pos_it;
        //If not found, return end()
        if(cur_pos_it == m_keys.end())
        {
            return const_iterator(NULL,1);
        }
        else
        {
            return ++sub_key_iter;
        }
    }
}

void sparse_block_tree_leaf_node::push_back(const std::vector<size_t>& key,size_t cur_idx)
{
    m_keys.push_back(key[cur_idx]);
    m_values.push_back(0);
}

size_t sparse_block_tree_branch_node::search(const std::vector<size_t>& key,const size_t idx) const
{
    std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),key[idx]);
    if(cur_pos_it == m_keys.end() || *cur_pos_it != key[idx])
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node<N>","search(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    size_t cur_pos = distance(m_keys.begin(),cur_pos_it);
    return m_children[cur_pos]->search(key,idx+1);
}

sparse_block_tree_iterator<true> sparse_block_tree_branch_node::end() const
{
    return const_iterator(NULL,m_order);
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
        if(sub_key.size() == 0)
        {
            return m_keys;
        }
        if(cur_idx == (sub_key.size() - 1))
        {
            return m_children[distance(m_keys.begin(),cur_pos)]->m_keys;
        }
        return m_children[distance(m_keys.begin(),cur_pos)]->get_sub_key_block_list(sub_key,cur_idx+1);
    }
}

sparse_block_tree_iterator<true> sparse_block_tree_branch_node::get_sub_key_begin_iterator_internal(const std::vector<size_t>& sub_key,const sparse_block_tree_node* root,std::vector<size_t>& displacement,const size_t cur_idx) const
{
    //If the sub key has already been found in full, we just start from the first key in this node
    size_t cur_pos = 0;
    if(cur_idx < sub_key.size())
    {
        std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(m_keys.begin(),m_keys.end(),sub_key[cur_idx]);
        if(cur_pos_it == m_keys.end() || *cur_pos_it != sub_key[cur_idx])
        {
            return end();
        }
        cur_pos  = std::distance(m_keys.begin(),cur_pos_it);
    }
    displacement[cur_idx] = cur_pos;
    return m_children[cur_pos]->get_sub_key_begin_iterator_internal(sub_key,root,displacement,cur_idx+1);
}

sparse_block_tree_iterator<true> sparse_block_tree_branch_node::get_sub_key_begin_iterator(const std::vector<size_t>& sub_key) const
{
    std::vector<size_t> displacement(m_order); 
    return get_sub_key_begin_iterator_internal(sub_key,this,displacement,0);
}

sparse_block_tree_iterator<true> sparse_block_tree_branch_node::get_sub_key_end_iterator(const std::vector<size_t>& sub_key) const
{
    const_iterator sub_key_iter = get_sub_key_begin_iterator(sub_key);
    if(sub_key_iter == end())
    {
        throw bad_parameter(g_ns,"sparse_block_tree_branch_node","get_sub_key_end_displacement(...)",
            __FILE__,__LINE__,"key not found"); 
    }
    else
    {
        //Get the next possible sub key and return the begin iterator to THAT, or end() 
        std::vector<size_t> partial_sub_key(sub_key.begin(),sub_key.end());
        size_t steps_back;
        for(steps_back = 1; steps_back < sub_key.size()+1; ++steps_back)
        {
            size_t prev_value = partial_sub_key.back();
            partial_sub_key.pop_back();
            const block_list& other_last_idx_values = get_sub_key_block_list(partial_sub_key,0);
            //If we are really done, return end()
            std::vector<size_t>::const_iterator cur_pos_it = std::lower_bound(other_last_idx_values.begin(),other_last_idx_values.end(),prev_value);
            ++cur_pos_it;

            //Found a valid sub key to increment
            if(cur_pos_it != m_keys.end())
            {
                partial_sub_key.push_back(*cur_pos_it);
                break;
            }
            else
            {
                //Otherwise,tep back one more, or return end() if we are done
                //Didn't find any possible dimension to increment
                if(steps_back == sub_key.size())
                {
                    return end();
                }
            }
        }

        //Now that we have advanced a dimension, fill in the rest of the sub key with zeros
        for(size_t steps_forward = 0; steps_forward < steps_back - 1; ++steps_forward)
        {
            partial_sub_key.push_back(0);
        }
        return get_sub_key_begin_iterator(partial_sub_key);
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
