#include "sparsity_data.h"
#include "range.h"
#include <algorithm>

using namespace std;

namespace libtensor {

const char* sparsity_data::k_clazz = "sparsity_data";

sparsity_data::sparsity_data(size_t order,const idx_list& keys,size_t value_order,const idx_list& values) : m_order(order),m_value_order(value_order),m_n_entries(keys.size()/order),m_keys(keys),m_values(values)
{
    if(order == 0)
    {
        throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"sparsity_data order cannot be zero"); 
    }
    if(keys.size() % order != 0)
    {
        throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"key list is not a multiple of sparsity_data order"); 
    }
    if(values.size() % m_n_entries != 0)
    {
        throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"values array size does not match number of entries"); 
    }

    for(size_t key_idx = 0; key_idx < m_n_entries; ++key_idx)
    {
        if(key_idx != 0)
        {
            bool equal = true; 
            for(size_t j = 0; j < m_order; ++j)
            {
                if(keys[key_idx*m_order+j] > keys[(key_idx-1)*m_order+j])
                {
                    equal = false;
                    break;
                }
                else if(keys[key_idx*m_order+j] < keys[(key_idx-1)*m_order+j])
                {
                    throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"Key list is not sorted"); 
                }
            }
            if(equal)
                throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"Duplicate keys in input"); 
        }
    }
}

void sparsity_data::set_values(size_t value_order,const idx_list& values)
{
    if(values.size() % m_n_entries != 0)
    {
        throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"values array size does not match number of entries"); 
    }

    m_value_order = value_order;
    if(!(values.size() == 0 && m_value_order == 0) && values.size() % m_value_order != 0)
    {
        throw bad_parameter(g_ns,k_clazz,"sparsity_data(...)",__FILE__,__LINE__,"values array size is not a multiple of m_value_order"); 
    }
    m_values = values;
}


bool sparsity_data::operator==(const sparsity_data& rhs) const
{
    return (m_keys == rhs.m_keys) && (m_values == rhs.m_values) 
                                  && (m_order == rhs.m_order) 
                                  && (m_value_order == rhs.m_value_order)
                                  && (m_n_entries == rhs.m_n_entries);
}

bool sparsity_data::operator!=(const sparsity_data& rhs) const
{
    return !(*this == rhs);
}

class key_idx_comparer_lt
{
    private:
        size_t m_order;
        const idx_list& m_keys;
    public:
        key_idx_comparer_lt(size_t order,const idx_list& keys) : m_order(order),m_keys(keys) {}
        bool operator()(size_t idx_0,size_t idx_1)
        {
            for(size_t j = 0; j < m_order; ++j)
            {
                if(m_keys[idx_0*m_order+j] < m_keys[idx_1*m_order+j])
                {
                    return true;
                }
                else if(m_keys[idx_0*m_order+j] > m_keys[idx_1*m_order+j])
                {
                    return false;
                }
            }
            return false;
        }
};

sparsity_data sparsity_data::permute(const runtime_permutation& perm) const
{
    idx_list key_idx_vec(m_n_entries);
    for(size_t i = 0; i < key_idx_vec.size(); ++i)
        key_idx_vec[i] = i;

    idx_list permuted_keys(m_n_entries*m_order);
    for(size_t i = 0; i < m_n_entries; ++i)
    {
        for(size_t j = 0; j < m_order; ++j)
            permuted_keys[i*m_order+j] = m_keys[i*m_order+j];
        perm.apply(&permuted_keys[i*m_order]);
    }


    sort(key_idx_vec.begin(),key_idx_vec.end(),key_idx_comparer_lt(m_order,permuted_keys));
    idx_list sorted_keys(m_n_entries*m_order);
    idx_list sorted_values(m_n_entries*m_value_order);
    for(size_t i = 0; i < m_n_entries; ++i)
    {
        for(size_t j = 0; j < m_order; ++j)
        {
            sorted_keys[i*m_order+j] = permuted_keys[key_idx_vec[i]*m_order+j];
        }
        for(size_t j = 0; j < m_value_order; ++j)
        {
            sorted_values[i*m_value_order+j] = m_values[key_idx_vec[i]*m_value_order+j];
        }
    }

    return sparsity_data(m_order,sorted_keys,m_value_order,sorted_values);
}

sparsity_data sparsity_data::contract(size_t contracted_subspace_idx) const
{
    idx_list key_idx_vec(m_n_entries);
    for(size_t i = 0; i < key_idx_vec.size(); ++i)
        key_idx_vec[i] = i;

    idx_list contr_keys(m_n_entries*(m_order-1));
    for(size_t i = 0; i < m_n_entries; ++i)
    {
        for(size_t j = 0; j < m_order; ++j)
        {
            if(j == contracted_subspace_idx)
                continue;
            contr_keys[i*(m_order-1)+(j > contracted_subspace_idx ? j - 1: j)] = m_keys[i*m_order+j];
        }
    }
    sort(key_idx_vec.begin(),key_idx_vec.end(),key_idx_comparer_lt(m_order-1,contr_keys));
    idx_list unique_key_idx_vec;
    for(size_t i = 0; i < key_idx_vec.size(); ++i)
    {
        bool equal = false;
        if(i != 0)
        {
            equal = true;
            for(size_t j = 0; j < m_order - 1; ++j)
            {
                if(contr_keys[key_idx_vec[i]*(m_order-1)+j] != contr_keys[key_idx_vec[i-1]*(m_order-1)+j])
                {
                    equal = false;
                    break;
                }
            }
        }
        if(!equal)
        {
            unique_key_idx_vec.push_back(key_idx_vec[i]);
        }
    }

    idx_list sorted_keys(unique_key_idx_vec.size()*(m_order-1));
    for(size_t i = 0; i < unique_key_idx_vec.size(); ++i)
    {
        for(size_t j = 0; j < m_order-1; ++j)
        {
            sorted_keys[i*(m_order-1)+j] = contr_keys[unique_key_idx_vec[i]*(m_order-1)+j];
        }
    }

    return sparsity_data(m_order-1,sorted_keys,0,idx_list());
}

class fuse_key_comparer
{
    private:
        size_t m_nfi;
        size_t m_order_0;
        const idx_list& m_keys_0;
        const idx_list& m_fuse_inds_1;
    public:
        fuse_key_comparer(size_t nfi,size_t order_0,const idx_list& keys_0,const idx_list& fuse_inds_1) : m_nfi(nfi),m_order_0(order_0),m_keys_0(keys_0),m_fuse_inds_1(fuse_inds_1) {}
        bool operator()(size_t key_idx_0,const idx_list::const_iterator it_1)
        {
            for(size_t j = 0; j < m_nfi; ++j)
            {
                if(*(m_keys_0.begin()+key_idx_0*m_order_0+j) < *(it_1+m_fuse_inds_1[j]))
                {
                    return true;
                }
                else if(*(m_keys_0.begin()+key_idx_0*m_order_0+j) > *(it_1+m_fuse_inds_1[j]))
                {
                    return false;
                }
            }
            return false;
        }
};

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

    idx_list permuted_rhs_idx_vec;
    for(size_t i = 0; i < permuted_rhs.m_n_entries; ++i)
    {
        permuted_rhs_idx_vec.push_back(i);
    }

    idx_list fused_keys;
    idx_list fused_values;
    for(size_t key_idx = 0; key_idx < m_n_entries; ++key_idx)
    {
        fuse_key_comparer comp(nfi,rhs.m_order,permuted_rhs.m_keys,lhs_indices);
        idx_list::iterator rhs_pos = lower_bound(permuted_rhs_idx_vec.begin(),permuted_rhs_idx_vec.end(),m_keys.begin()+key_idx*m_order,comp);
        size_t rhs_key_idx;
        if(rhs_pos == permuted_rhs_idx_vec.end())
            rhs_key_idx = permuted_rhs_idx_vec.size();
        else 
        {
            rhs_key_idx = *rhs_pos;
            for(size_t j = 0; j < nfi; ++j)
            {
                if(permuted_rhs.m_keys[rhs_key_idx*rhs.m_order+j] != m_keys[key_idx*m_order+lhs_indices[j]])
                {
                    rhs_key_idx = permuted_rhs_idx_vec.size();
                    break;
                }
            }
        }

        //Add keys that share the same fused key base until it changes
        while(rhs_key_idx != permuted_rhs_idx_vec.size())
        {
            for(size_t j = 0; j < m_order; ++j)
                fused_keys.push_back(m_keys[key_idx*m_order+j]);
            for(size_t j = nfi; j < rhs.m_order; ++j)
                fused_keys.push_back(permuted_rhs.m_keys[rhs_key_idx*rhs.m_order+j]);

            for(size_t j = 0; j < m_value_order; ++j)
                fused_values.push_back(m_values[key_idx*m_value_order+j]);
            for(size_t j = 0; j < rhs.m_value_order; ++j)
                fused_values.push_back(permuted_rhs.m_values[rhs_key_idx*rhs.m_value_order+j]);


            ++rhs_key_idx;
            if(rhs_key_idx == permuted_rhs_idx_vec.size()) break;
            for(size_t j = 0; j < nfi; ++j)
            {
                if(permuted_rhs.m_keys[rhs_key_idx*rhs.m_order+j] != m_keys[key_idx*m_order+lhs_indices[j]])
                {
                    rhs_key_idx = permuted_rhs_idx_vec.size();
                    break;
                }
            }
        }
    }

    size_t fused_key_order = m_order+rhs.m_order - nfi;
    size_t fused_value_order = m_value_order+rhs.m_value_order;

    return sparsity_data(fused_key_order,fused_keys,fused_value_order,fused_values);
}

sparsity_data sparsity_data::truncate_subspace(size_t subspace_idx,
                                               const idx_pair& bounds) const
{
    if(subspace_idx >= m_order)  
    {
        throw bad_parameter(g_ns,k_clazz,"truncated_subspace(...)",__FILE__,__LINE__,"subspace idx out of bounds"); 
    }

    idx_list new_keys;
    idx_list new_values;
    for(size_t i = 0; i < m_n_entries; ++i)
    {
        size_t block_idx = m_keys[i*m_order+subspace_idx];
        if((bounds.first <= block_idx) && (block_idx < bounds.second))
        {
            for(size_t j = 0; j < m_order; ++j)
                new_keys.push_back(m_keys[i*m_order+j]);
            for(size_t j = 0; j < m_value_order; ++j)
                new_values.push_back(m_values[i*m_value_order+j]);
        }
    }

    return sparsity_data(m_order,new_keys,m_value_order,new_values);
}

sparsity_data sparsity_data::insert_entries(size_t subspace_idx,
                                            const idx_list& entries) const
{
    if(subspace_idx >= m_order)  
    {
        throw bad_parameter(g_ns,k_clazz,"insert_entries(...)",__FILE__,__LINE__,"subspace idx out of bounds"); 
    }

    idx_list new_keys;
    for(size_t i = 0; i < m_n_entries; ++i)
    {
        for(size_t entry_idx = 0; entry_idx  < entries.size(); ++entry_idx)
        {
            for(size_t j = 0; j < m_order+1; ++j)
            {
                new_keys.push_back(m_keys[i*m_order+j]);
            }
            new_keys.push_back(entries[entry_idx]);
        }
    }
    sparsity_data inter(m_order+1,new_keys,0,idx_list());
    runtime_permutation perm(inter.m_order);
    perm.permute(subspace_idx,inter.m_order-1);
    return inter.permute(perm);
}

sparsity_data sparsity_data::merge(const sparsity_data& other) const
{
    if(other.m_order != m_order)
    {
        throw bad_parameter(g_ns,k_clazz,"merge(...)",__FILE__,__LINE__,"sparsity_data objects must be of the same order"); 
    }
    idx_list merged_keys(m_keys);
    merged_keys.insert(merged_keys.end(),other.m_keys.begin(),other.m_keys.end());

    idx_list key_idx_vec(merged_keys.size()/m_order);
    for(size_t i = 0; i < key_idx_vec.size(); ++i)
        key_idx_vec[i] = i;

    sort(key_idx_vec.begin(),key_idx_vec.end(),key_idx_comparer_lt(m_order,merged_keys));
    idx_list unique_key_idx_vec;
    for(size_t i = 0; i < key_idx_vec.size(); ++i)
    {
        bool equal = false;
        if(i != 0)
        {
            equal = true;
            for(size_t j = 0; j < m_order; ++j)
            {
                if(merged_keys[key_idx_vec[i]*m_order+j] != merged_keys[key_idx_vec[i-1]*m_order+j])
                {
                    equal = false;
                    break;
                }
            }
        }
        if(!equal)
        {
            unique_key_idx_vec.push_back(key_idx_vec[i]);
        }
    }

    idx_list sorted_keys(unique_key_idx_vec.size()*m_order);
    for(size_t i = 0; i < unique_key_idx_vec.size(); ++i)
    {
        for(size_t j = 0; j < m_order; ++j)
        {
            sorted_keys[i*m_order+j] = merged_keys[unique_key_idx_vec[i]*m_order+j];
        }
    }
    return sparsity_data(m_order,sorted_keys);
}

} // namespace libtensor
