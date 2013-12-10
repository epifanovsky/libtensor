#include "sparse_block_tree_any_order_new.h"
#include "sparse_block_tree_iterator_new.h"

//TODO REMOVE
#include <iostream>

namespace libtensor {

namespace impl {

sparse_block_tree_any_order_new::sparse_block_tree_any_order_new(const sparse_block_tree_any_order_new& rhs)
{
    m_order = rhs.m_order;
    m_keys.resize(rhs.m_keys.size());
    m_children.resize(rhs.m_children.size());
    m_values.resize(rhs.m_values.size());
    for(size_t i = 0; i < m_keys.size(); ++i)
    {
        m_keys[i] = rhs.m_keys[i];
    }

    if(m_order > 1)
    {
        for(size_t i = 0; i < m_children.size(); ++i)
        {
            m_children[i] = new sparse_block_tree_any_order_new(*rhs.m_children[i]);
        }
    } 
    else
    {
        for(size_t i = 0; i < m_values.size(); ++i)
        {
            m_values[i] = rhs.m_values[i];
        }
    }
}

sparse_block_tree_any_order_new::sparse_block_tree_any_order_new(const std::vector< std::vector<key_t> >& sig_blocks,size_t order)
{
    m_order = order;
    for(size_t i = 0; i < sig_blocks.size(); ++i)
    {
        push_back(sig_blocks[i],m_order);
    }
}

struct sparse_block_tree_any_order_new::kv_pair_compare {

    bool operator()(const std::pair< std::vector<key_t>,value_t>& p1,const std::pair< std::vector<key_t>,value_t>& p2)
    {
        if(p1.first.size() != p2.first.size())
        {
            return false;
        }

        for(size_t i = 0; i < p1.first.size(); ++i)
        {
            if(p1.first[i] < p2.first[i])
            {
                return true;
            }
            else if(p1.first[i] > p2.first[i])
            {
                return false;
            }
        }
        return false;
    }
};

sparse_block_tree_any_order_new sparse_block_tree_any_order_new::permute(const runtime_permutation& perm) const
{
    std::vector< std::pair< std::vector<key_t>, value_t > > kv_pairs;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        std::vector<key_t> new_key = it.key(); 
        perm.apply(new_key);
        kv_pairs.push_back(std::make_pair(new_key,value_t(*it)));
    }

    std::sort(kv_pairs.begin(),kv_pairs.end(),kv_pair_compare());

    std::vector< std::vector<key_t> > all_keys;
    std::vector< value_t > all_vals;
    for(size_t i = 0; i < kv_pairs.size(); ++i)
    {
        all_keys.push_back(kv_pairs[i].first);
        all_vals.push_back(kv_pairs[i].second);
    }

    sparse_block_tree_any_order_new sbt(all_keys,m_order);
    size_t m = 0; 
    for(iterator it = sbt.begin(); it != sbt.end(); ++it)
    {
        *it = all_vals[m];
        ++m;
    }
    return sbt;
}

sparse_block_tree_iterator_new<false> sparse_block_tree_any_order_new::begin()
{
    if(m_keys.size() == 0)
    {
        return iterator(NULL);
    }
    else
    {
        return iterator(this);
    }
}

sparse_block_tree_iterator_new<false> sparse_block_tree_any_order_new::end()
{
    return iterator(NULL);
}


//TODO: merge const and non-const functions
sparse_block_tree_iterator_new<true> sparse_block_tree_any_order_new::begin() const
{
    if(m_keys.size() == 0)
    {
        return const_iterator(NULL);
    }
    else
    {
        return const_iterator(this);
    }
}

sparse_block_tree_iterator_new<true> sparse_block_tree_any_order_new::end() const
{
    return const_iterator(NULL);
}

bool sparse_block_tree_any_order_new::operator==(const sparse_block_tree_any_order_new& rhs) const
{
    for(size_t i = 0; i < m_keys.size(); ++i)
    {
        if(m_keys[i] != rhs.m_keys[i])
        {
            return false;
        }
        if(m_order == 1)
        {
            if(m_values[i] != rhs.m_values[i])
            {
                return false;
            }
        }
        else
        {
            if((*m_children[i]) != (*rhs.m_children[i]))
            {
                return false;
            }
        }
    }
    return true;
}

bool sparse_block_tree_any_order_new::operator!=(const sparse_block_tree_any_order_new& rhs) const
{
    return !(*this == rhs);
}

} // namespace impl

} // namespace libtensor
