#include "sparsity_data.h"
#include "range.h"
#include <algorithm>

using namespace std;

namespace libtensor {

const char* sparsity_data::k_clazz = "sparsity_data";

sparsity_data::sparsity_data(size_t order,const std::vector<idx_list>& keys) : m_order(order)
{
    if(order == 0)
    {
        throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"sparsity_data order cannot be zero"); 
    }
    for(size_t key_idx = 0; key_idx < keys.size(); ++key_idx)
    {
        const idx_list& prev_key = (key_idx == 0 ? keys[key_idx] : keys[key_idx - 1]);
        const idx_list& key = keys[key_idx];
        if(key.size() != order)
        {
            throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"Invalid key length"); 
        }
        if(key_idx != 0 && key < prev_key)
        {
            throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"Key list is not sorted"); 
        }
        if(key_idx != 0 && key == prev_key)
        {
            throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"Duplicate keys in input"); 
        }
        m_kv_pairs.push_back(make_pair(key,idx_list()));
    }
}

bool sparsity_data::operator==(const sparsity_data& rhs) const
{
    return m_kv_pairs == rhs.m_kv_pairs;
}

bool sparsity_data::operator!=(const sparsity_data& rhs) const
{
    return !(*this == rhs);
}

sparsity_data sparsity_data::permute(const runtime_permutation& perm) const
{
    vector<pair<idx_list,idx_list> > permuted_kv_pairs;
    for(size_t kv_idx = 0; kv_idx < m_kv_pairs.size(); ++kv_idx)
    {
        idx_list new_key = m_kv_pairs[kv_idx].first;
        perm.apply(new_key);
        permuted_kv_pairs.push_back(make_pair(new_key,m_kv_pairs[kv_idx].second));
    }
    sort(permuted_kv_pairs.begin(),permuted_kv_pairs.end());

    return sparsity_data(m_order,permuted_kv_pairs);
}

sparsity_data sparsity_data::contract(size_t contracted_subspace_idx) const
{
    vector<idx_list> contr_keys;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        idx_list new_key(it->first);
        new_key.erase(new_key.begin()+contracted_subspace_idx);
        contr_keys.push_back(new_key);
    }
    sort(contr_keys.begin(),contr_keys.end());
    vector<idx_list>::iterator last = unique(contr_keys.begin(),contr_keys.end());
    contr_keys.erase(last,contr_keys.end());
    return sparsity_data(m_order-1,contr_keys);
}

sparsity_data sparsity_data::fuse(const sparsity_data& rhs,
                                  const idx_list& lhs_indices,
                                  const idx_list& rhs_indices) const
{
    //Sanitize input
    if(lhs_indices.size() != rhs_indices.size())
    {
        throw bad_parameter(g_ns,k_clazz,"fuse(...)",__FILE__,__LINE__,"lhs and rhs number of fused indices dont match"); 
    }
    else if(lhs_indices.size() == 0)
    {
        throw bad_parameter(g_ns,k_clazz,"fuse(...)",__FILE__,__LINE__,"must specify at least one index to fuse"); 
    }
    size_t nfi = lhs_indices.size();
    for(size_t idx = 0; idx < nfi; ++idx)
    {
        if((lhs_indices[idx] >= m_order) || (rhs_indices[idx] >= rhs.m_order))
        {
            throw bad_parameter(g_ns,k_clazz,"fuse(...)",__FILE__,__LINE__,"subspace idx out of bounds"); 
        }
    }

    //Permute RHS to bring the fused indices to the left-most position...
    idx_list permutation_entries;
    idx_list indices_to_erase;
    for(size_t rhs_rel_idx = 0; rhs_rel_idx < rhs_indices.size(); ++rhs_rel_idx)
    {
    	size_t rhs_fused_idx = rhs_indices[rhs_rel_idx];
    	permutation_entries.push_back(rhs_fused_idx);
    	indices_to_erase.push_back(rhs_fused_idx);
    }

    //Erase indices that are fused in reverse order
    sort(indices_to_erase.begin(),indices_to_erase.end());
    idx_list rhs_unfused_inds(range(0,rhs.m_order));
    for(size_t erase_idx_incr = 0; erase_idx_incr < indices_to_erase.size(); ++erase_idx_incr)
    {
    	rhs_unfused_inds.erase(rhs_unfused_inds.begin() + indices_to_erase[indices_to_erase.size() - erase_idx_incr - 1]);
    }

    //Add the remaining unfused indices to the permutation
    for(size_t rhs_unfused_idx = 0; rhs_unfused_idx < rhs_unfused_inds.size(); ++rhs_unfused_idx)
    {
    	permutation_entries.push_back(rhs_unfused_inds[rhs_unfused_idx]);
    }

    //Don't permute if unnecessary
    runtime_permutation perm(permutation_entries);
    const sparsity_data& permuted_rhs = (perm == runtime_permutation(rhs.m_order)) ? rhs : rhs.permute(perm);

    vector<pair<idx_list,idx_list> > kv_pairs;
    for(size_t kv_idx = 0; kv_idx < m_kv_pairs.size(); ++kv_idx)
    {
        idx_list sub_key(nfi);
        for(size_t lhs_idx = 0; lhs_idx < nfi; ++lhs_idx)
        {
            sub_key[lhs_idx] = m_kv_pairs[kv_idx].first[lhs_indices[lhs_idx]];
        }

        idx_list padded_sub_key(sub_key); 
        padded_sub_key.resize(rhs.m_order);
        const_iterator rhs_pos = lower_bound(permuted_rhs.begin(),permuted_rhs.end(),pair<idx_list,idx_list>(padded_sub_key,idx_list()));
        if(rhs_pos != permuted_rhs.end())
        {
            for(size_t rel_fused_idx = 0; rel_fused_idx < nfi; ++rel_fused_idx)
            {
                if(rhs_pos->first[rel_fused_idx] != sub_key[rel_fused_idx]) 
                {
                    rhs_pos = permuted_rhs.end();
                    break;
                }
            }
        }

        //Add keys that share the same fused key base until it changes
        while(rhs_pos != permuted_rhs.end())
        {
            const pair<idx_list,idx_list>& r_kv = *rhs_pos;
            pair<idx_list,idx_list> kv(m_kv_pairs[kv_idx]);
            kv.first.insert(kv.first.end(),r_kv.first.begin()+nfi,r_kv.first.end());
            kv.second.insert(kv.second.end(),r_kv.second.begin(),r_kv.second.end());
            kv_pairs.push_back(kv);

            ++rhs_pos;
            if(rhs_pos == permuted_rhs.end()) break;
            for(size_t rel_fused_idx = 0; rel_fused_idx < nfi; ++rel_fused_idx)
            {
                if(rhs_pos->first[rel_fused_idx] != sub_key[rel_fused_idx])
                {
                    rhs_pos = permuted_rhs.end();
                    break;
                }
            }
        }
    }
    return sparsity_data(this->m_order+rhs.m_order - nfi,kv_pairs);
}

sparsity_data sparsity_data::truncate_subspace(size_t subspace_idx,
                                               const idx_pair& bounds) const
{
    if(subspace_idx >= m_order)  
    {
        throw bad_parameter(g_ns,k_clazz,"truncated_subspace(...)",__FILE__,__LINE__,"subspace idx out of bounds"); 
    }
    vector<pair<idx_list,idx_list> > kv_pairs;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        size_t block_idx = it->first[subspace_idx];
        if((bounds.first <= block_idx) && (block_idx < bounds.second))
        {
            kv_pairs.push_back(*it);
        }
    }
    return sparsity_data(m_order,kv_pairs);
}

sparsity_data sparsity_data::insert_entries(size_t subspace_idx,
                                            const idx_list& entries) const
{
    if(subspace_idx >= m_order)  
    {
        throw bad_parameter(g_ns,k_clazz,"insert_entries(...)",__FILE__,__LINE__,"subspace idx out of bounds"); 
    }
    vector<pair<idx_list,idx_list> > kv_pairs;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        for(size_t entry_idx = 0; entry_idx  < entries.size(); ++entry_idx)
        {
            pair<idx_list,idx_list> kv(*it);
            kv.second.clear();
            kv.first.push_back(entries[entry_idx]);
            kv_pairs.push_back(kv);
        }
    }
    sparsity_data inter(m_order+1,kv_pairs);
    runtime_permutation perm(inter.m_order);
    perm.permute(subspace_idx,inter.m_order-1);
    return inter.permute(perm);
}

} // namespace libtensor
