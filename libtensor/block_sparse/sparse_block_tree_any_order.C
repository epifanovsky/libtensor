#include "sparse_block_tree_any_order.h"
#include "../defs.h"
#include "../exception.h"

namespace libtensor {

//token

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
    return m_node->search(key,0);
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

sparse_block_tree_any_order sparse_block_tree_any_order::fuse(const sparse_block_tree_any_order& rhs,const std::vector<size_t>& lhs_indices,const std::vector<size_t>& rhs_indices) const
{

    //Sanitize input
    if(lhs_indices.size() != rhs_indices.size())
    {
        throw bad_parameter(g_ns,"sparse_block_tree_any_order","fuse(...)",
            __FILE__,__LINE__,"lhs and rhs number of fused indices dont match"); 
    }
    else if(lhs_indices.size() == 0)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_any_order","fuse(...)",
            __FILE__,__LINE__,"must specify at least one index to fuse"); 
    }

    size_t n_fused_inds = lhs_indices.size();
    size_t out_order = m_order+rhs.m_order-n_fused_inds;
    std::vector<key_t> new_keys;

    for(size_t i = 1; i < lhs_indices.size(); ++i)
    {
        size_t cur_lhs = lhs_indices[i]; 
        size_t prev_lhs = lhs_indices[i-1]; 
        if(cur_lhs < prev_lhs)
        {
            throw bad_parameter(g_ns,"sparse_block_tree_any_order","fuse(...)",
                __FILE__,__LINE__,"lhs not strictly increasing"); 
        }
    }


    //Permute RHS to bring the fused indices to the left-most position...
    size_t rhs_order = rhs.get_order();
    std::vector<size_t> permutation_entries(rhs_order);
    for(size_t i = 0; i < rhs_indices.size(); ++i)
    {
        permutation_entries[i] = rhs_indices[i];
    }

    size_t cur_fused_idx = 0;
    size_t idx_excluding_fused = 0; 
    for(size_t i = rhs_indices.size(); i < rhs_order; ++i)
    {
        //Still fused indices to skip over?
        if(cur_fused_idx < rhs_indices.size())
        {
            //Skip over patches of fused indices
            while(idx_excluding_fused == rhs_indices[cur_fused_idx])
            {
                ++cur_fused_idx;
                ++idx_excluding_fused;
                if(cur_fused_idx == rhs_indices.size())
                {
                    break;
                }
            }
        }
        permutation_entries[i] = idx_excluding_fused;
        ++idx_excluding_fused;
    }

    //Don't permute if unnecessary
    runtime_permutation perm(permutation_entries);
    const sparse_block_tree_any_order& rhs_permuted = (perm == runtime_permutation(rhs_order)) ? rhs : rhs.permute(perm);

    for(const_iterator it = begin(); it != end(); ++it)
    {
        key_t base_key = it.key();

        //The first part of our new key is just the old left hand side key - we copy this part now
        key_t new_key(out_order);
        for(size_t new_key_idx = 0; new_key_idx < m_order; ++new_key_idx)
        {
            new_key[new_key_idx] = base_key[new_key_idx];
        }

        //Extract the fusing portion of the LHS key that determines what RHS keys to include
        key_t sub_key(n_fused_inds);
        for(size_t lhs_idx = 0; lhs_idx < n_fused_inds; ++lhs_idx)
        {
            sub_key[lhs_idx] = base_key[lhs_indices[lhs_idx]];
        }

        const_iterator rhs_it = rhs_permuted.m_node->get_sub_key_begin_iterator(sub_key);
        if(rhs_it == rhs_permuted.end())
        {
            continue;
        }
        const_iterator rhs_end = rhs_permuted.m_node->get_sub_key_end_iterator(sub_key);

        //Attach each relevant sub key to the attachment point
        for(rhs_it; rhs_it != rhs_end; ++rhs_it)
        {
            key_t rhs_key = rhs_it.key();

            //Now fill in the right side of the key, everything after the fused indices 
            for(size_t sub_key_idx = n_fused_inds; sub_key_idx < rhs_permuted.m_order; ++sub_key_idx)
            {
                new_key[m_order -  n_fused_inds + sub_key_idx] = rhs_key[sub_key_idx];
            }

            //Add the key to the list
            new_keys.push_back(new_key);
        }
    }

    //By virtue of both trees being sorted, the list will be sorted already
    return sparse_block_tree_any_order(new_keys,out_order);
}

sparse_block_tree_any_order sparse_block_tree_any_order::fuse(const sparse_block_tree_any_order& rhs) const
{
    std::vector<size_t> lhs_fuse_points(1,m_order - 1);
    std::vector<size_t> rhs_fuse_points(1,0);
    return fuse(rhs,lhs_fuse_points,rhs_fuse_points);
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
