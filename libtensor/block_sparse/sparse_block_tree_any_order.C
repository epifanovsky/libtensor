#include "sparse_block_tree_any_order.h"
#include "../defs.h"
#include "../exception.h"

namespace libtensor {

//We don't check vector length because assume that is taken care of by template wrappers
void sparse_block_tree_any_order::init(const std::vector< key_t > sig_blocks)
{
    if(m_order > 1) 
    {
        m_node = new impl::sparse_block_tree_branch_node(sig_blocks[0],0,m_order);
    }
    else
    {
        m_node = new impl::sparse_block_tree_leaf_node(sig_blocks[0],0);
    }
    //Ensure that block list is sorted in lexicographic order
    for(size_t i = 1; i < sig_blocks.size(); ++i)
    {
        const key_t& cur = sig_blocks[i];
        const key_t& prev = sig_blocks[i-1];

        bool equal = true;
        for(size_t j = 0; j < m_order; ++j)
        {
            if(cur[j] < prev[j])
            {
                throw bad_parameter(g_ns,"sparse_block_tree<N>","sparse_block_tree(...)",
                    __FILE__,__LINE__,"list is not strictly increasing"); 
            }
            else if(cur[j] > prev[j])
            {
                equal = false;
                break;
            }
        }

        if(equal)
        {
            throw bad_parameter(g_ns,"sparse_block_tree<N>","sparse_block_tree(...)",
                __FILE__,__LINE__,"duplicate keys are not allowed"); 
        }

        m_node->push_back(cur,0);
    }
}

sparse_block_tree_any_order::sparse_block_tree_any_order(const std::vector< key_t > sig_blocks,size_t order) : m_order(order)
{
    init(sig_blocks);
}

size_t sparse_block_tree_any_order::search(const std::vector<size_t>& key) const
{
    if(key.size() != m_order)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_any_order","search(...)",
            __FILE__,__LINE__,"key length does not match depth of tree"); 
    }
    std::vector<size_t> positions(m_order);
    m_node->search(key,positions,0);
    return *(const_iterator(m_node,positions,m_order));
}

struct sparse_block_tree_any_order::kv_pair_compare {
    bool operator()(const std::pair< key_t,size_t>& p1,const std::pair< key_t,size_t>& p2)
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

const block_list& sparse_block_tree_any_order::get_sub_key_block_list(const std::vector<size_t>& sub_key) const
{
    if((sub_key.size() == 0) || (sub_key.size() > (m_order - 1)))
    {
        throw bad_parameter(g_ns,"sparse_block_tree_any_order","get_sub_key_block_list(...)",
            __FILE__,__LINE__,"invalid key size"); 
    }
    return m_node->get_sub_key_block_list(sub_key,0); 
};

sparse_block_tree_any_order sparse_block_tree_any_order::permute(const runtime_permutation& perm) const
{
    std::vector< std::pair< key_t, size_t > > kv_pairs;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        key_t new_key = it.key(); 
        perm.apply(new_key);
        kv_pairs.push_back(std::make_pair(new_key,*it));
    }


    std::sort(kv_pairs.begin(),kv_pairs.end(),kv_pair_compare());

    std::vector< key_t > all_keys;
    std::vector<size_t> all_vals;
    for(size_t i = 0; i < kv_pairs.size(); ++i)
    {
        all_keys.push_back(kv_pairs[i].first);
        all_vals.push_back(kv_pairs[i].second);
    }

    sparse_block_tree_any_order sbt(all_keys,m_order);
    size_t m = 0; 
    for(iterator it = sbt.begin(); it != sbt.end(); ++it)
    {
        *it = all_vals[m];
        ++m;
    }
    return sbt;
}

sparse_block_tree_any_order sparse_block_tree_any_order::contract(size_t contract_idx) const
{
    std::vector< std::pair< key_t, size_t > > kv_pairs;
    size_t out_order = m_order - 1;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        key_t key = it.key(); 
        key_t new_key(out_order);

        size_t new_key_idx = 0;
        for(size_t key_idx = 0; key_idx < m_order; ++key_idx)
        {
            if(key_idx == contract_idx)
            {
                continue;
            }

            new_key[new_key_idx] = key[key_idx];
            ++new_key_idx;
        }
        kv_pairs.push_back(std::make_pair(new_key,*it));
    }

    std::sort(kv_pairs.begin(),kv_pairs.end(),kv_pair_compare());


    std::vector<key_t> all_keys;
    std::vector<size_t> all_vals;

    for(size_t i = 0; i < kv_pairs.size(); ++i)
    {
        //Remove the duplicate keys, if there are any
        if(i != 0)
        {
            bool equal = true;
            for(size_t j = 0; j < out_order; ++j)
            {
                if(kv_pairs[i].first[j] != kv_pairs[i-1].first[j])
                {
                    equal = false;
                    break;
                }
            }
            if(equal)
            {
                continue;
            }
        }
        all_keys.push_back(kv_pairs[i].first);
        all_vals.push_back(kv_pairs[i].second);
    }

    sparse_block_tree_any_order sbt(all_keys,out_order);
    size_t m = 0; 
    for(iterator it = sbt.begin(); it != sbt.end(); ++it)
    {
        *it = all_vals[m];
        ++m;
    }
    return sbt;
}

sparse_block_tree_any_order sparse_block_tree_any_order::fuse(const sparse_block_tree_any_order& rhs) const
{
    size_t out_order = m_order+rhs.m_order-1;
    std::vector<key_t> new_keys;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        key_t base_key = it.key();

        //Our new key consists of everything before the fuse point, then key from rhs that was fused in
        //We do the first segment now
        key_t new_key(out_order);
        for(size_t new_key_idx = 0; new_key_idx < m_order-1; ++new_key_idx)
        {
            new_key[new_key_idx] = base_key[new_key_idx];
        }

        //Determine what keys in the rhs match to the attachment point on the LHS
        key_t rhs_key(rhs.m_order);
        std::vector<size_t>::const_iterator attach_pos_it = std::lower_bound(rhs.m_node->m_keys.begin(),rhs.m_node->m_keys.end(),base_key[m_order-1]);
        
        //Nothing to graft onto this branch tip
        if(attach_pos_it == rhs.m_node->m_keys.end())
        {
            continue;
        }

        size_t attach_pos = std::distance<std::vector<size_t>::const_iterator>(rhs.m_node->m_keys.begin(),attach_pos_it);
        std::vector<size_t> positions(rhs.m_order,0); 
        positions[0] = attach_pos;
        const_iterator rhs_it(rhs.m_node,positions,rhs.m_order);

        //We end at the next child, or rhs.end() if we are the last child
        const impl::sparse_block_tree_node* end_ptr;
        if(attach_pos == (rhs.m_node->m_keys.size() - 1))
        {
            end_ptr = NULL;
        }
        else
        {
            end_ptr = rhs.m_node; 
            positions[0] = attach_pos+1;
        }
        const_iterator rhs_end(end_ptr,positions,rhs.m_order);

        //Attach each relevant sub key to the attachment point
        for(rhs_it; rhs_it != rhs_end; ++rhs_it)
        {
            key_t rhs_key = rhs_it.key();

            //Now fill in the middle (fused) segment of the new key
            for(size_t sub_key_idx = 0; sub_key_idx < rhs.m_order; ++sub_key_idx)
            {
                new_key[m_order - 1 + sub_key_idx] = rhs_key[sub_key_idx];
            }

            //Add the key to the list
            new_keys.push_back(new_key);
        }
    }

    //By virtue of both trees being sorted, the list will be sorted already
    return sparse_block_tree_any_order(new_keys,out_order);
}

bool sparse_block_tree_any_order::operator==(const sparse_block_tree_any_order& rhs) const
{
    if(m_order != rhs.m_order)
    {
        return false;
    }

    const_iterator rhs_it = rhs.begin();
    for(const_iterator lhs_it = begin(); lhs_it != end(); ++lhs_it)
    {
        if(rhs_it == rhs.end())
        {
            return false;
        }

        //Compare keys
        const key_t& lhs_key = lhs_it.key();
        const key_t& rhs_key = rhs_it.key();

        for(size_t i = 0; i < m_order; ++i)
        {
            if(lhs_key[i] != rhs_key[i])
            {
                return false;
            }
        }

        //Compare values
        if(*lhs_it != *rhs_it)
        {
            return false;
        }
        ++rhs_it;
    }
    return true;
}

bool sparse_block_tree_any_order::operator!=(const sparse_block_tree_any_order& rhs) const
{
    return !(*this == rhs);
}

} // namespace libtensor
