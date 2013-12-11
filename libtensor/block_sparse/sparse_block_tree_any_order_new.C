#include "sparse_block_tree_new.h"
#include "sparse_block_tree_iterator_new.h"
#include "range.h"

namespace libtensor {

namespace impl {

//Used to return empty trees by sub_tree
static const sparse_block_tree_any_order_new empty = sparse_block_tree_new<1>(std::vector< sequence<1,size_t> >());

sparse_block_tree_any_order_new& sparse_block_tree_any_order_new::operator=(const sparse_block_tree_any_order_new& rhs)
{
    if(this != &rhs)
    {
        //First delete old children and values
        for(size_t i = 0; i < m_children.size(); ++i)
        {
            delete m_children[i];
        }

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
    return *this;
}

sparse_block_tree_any_order_new::sparse_block_tree_any_order_new(const sparse_block_tree_any_order_new& rhs) : m_keys(rhs.m_keys),
                                                                                                               m_values(rhs.m_values)
{
    m_order = rhs.m_order;

    if(m_order > 1)
    {
        m_children.resize(rhs.m_children.size());
        for(size_t i = 0; i < m_children.size(); ++i)
        {
            m_children[i] = new sparse_block_tree_any_order_new(*rhs.m_children[i]);
        }
    } 
}

sparse_block_tree_any_order_new::~sparse_block_tree_any_order_new() 
{
    for(size_t i = 0; i < m_children.size(); ++i)
    {
        delete m_children[i];
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

const sparse_block_tree_any_order_new& sparse_block_tree_any_order_new::get_sub_tree(const std::vector<key_t>& sub_key) const
{
    static const char *method = "get_sub_tree(const std::vector<key_t>& sub_key) const";

    if((sub_key.size() >= m_order) || (sub_key.size() == 0))
    {
        throw bad_parameter(g_ns,k_clazz,method,__FILE__,__LINE__,"key is wrong size"); 
    }

    //Find the position of the current key in this node's list of keys
    const sparse_block_tree_any_order_new* cur_node = this;
    for(size_t i = 0; i < sub_key.size(); ++i)
    {
        std::vector<key_t>::const_iterator cur_pos = std::lower_bound(cur_node->m_keys.begin(),cur_node->m_keys.end(),sub_key[i]);
        if(cur_pos == cur_node->m_keys.end() || *cur_pos != sub_key[i])
        {
            return empty;
        }
        cur_node = cur_node->m_children[distance(cur_node->m_keys.begin(),cur_pos)];
    }
    return *cur_node;
}

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

sparse_block_tree_any_order_new sparse_block_tree_any_order_new::fuse(const sparse_block_tree_any_order_new& rhs,const std::vector<size_t>& lhs_indices,const std::vector<size_t>& rhs_indices) const
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
    std::vector< std::vector<key_t> > new_keys;
    std::vector< value_t > new_values;

    //Permute RHS to bring the fused indices to the left-most position...
    size_t rhs_order = rhs.m_order;
    std::vector<size_t> permutation_entries;
    std::vector<size_t> indices_to_erase;
    for(size_t rhs_fused_idx_incr = 0; rhs_fused_idx_incr < rhs_indices.size(); ++rhs_fused_idx_incr)
    {
    	size_t rhs_fused_idx = rhs_indices[rhs_fused_idx_incr];
    	permutation_entries.push_back(rhs_fused_idx);
    	indices_to_erase.push_back(rhs_fused_idx);
    }

    //Erase indices that are fused in reverse order
    sort(indices_to_erase.begin(),indices_to_erase.end());
    std::vector<size_t> rhs_unfused_inds(range(0,rhs_order));
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
    const sparse_block_tree_any_order_new& rhs_permuted = (perm == runtime_permutation(rhs_order)) ? rhs : rhs.permute(perm);

    for(const_iterator it = begin(); it != end(); ++it)
    {
        std::vector<key_t> new_key(out_order);
        value_t new_value(*it);
        size_t lhs_val_size = new_value.size();

        //The first part of our new key is just the old left hand side key - we copy this part now
        std::vector<key_t> base_key = it.key();
        for(size_t new_key_idx = 0; new_key_idx < m_order; ++new_key_idx)
        {
            new_key[new_key_idx] = base_key[new_key_idx];
        }

        //Extract the fusing portion of the LHS key that determines what RHS keys to include
        std::vector<key_t> sub_key(n_fused_inds);
        for(size_t lhs_idx = 0; lhs_idx < n_fused_inds; ++lhs_idx)
        {
            sub_key[lhs_idx] = base_key[lhs_indices[lhs_idx]];
        }

        const sparse_block_tree_any_order_new& st = rhs_permuted.get_sub_tree(sub_key);
        //Attach each relevant sub key to the attachment point
        for(sparse_block_tree_any_order_new::const_iterator rhs_it = st.begin(); rhs_it != st.end(); ++rhs_it)
        {
            std::vector<key_t> rhs_key = rhs_it.key();

            //Now fill in the right side of the key, everything after the fused indices 
            for(size_t rhs_idx = 0; rhs_idx < rhs_key.size(); ++rhs_idx)
            {
                new_key[m_order + rhs_idx] = rhs_key[rhs_idx];
            }

            //We merge the data stored in both trees because this is most commonly what is
            //needed for storing tensor offsets
            const value_t& rhs_val = *rhs_it;
            size_t val_size = lhs_val_size + rhs_val.size();
            if(new_value.size() != val_size)
            {
                new_value.resize(val_size);
            }
            for(size_t rhs_val_idx = 0; rhs_val_idx < rhs_val.size(); ++rhs_val_idx)
            {
                new_value[lhs_val_size + rhs_val_idx] = rhs_val[rhs_val_idx];
            }

            //Add the key and value to the list
            new_keys.push_back(new_key);
            new_values.push_back(new_value);
        }
    }

    //By virtue of both trees being sorted, the list will be sorted already
    sparse_block_tree_any_order_new new_tree(new_keys,out_order);
    size_t m = 0; 
    for(sparse_block_tree_any_order_new::iterator it = new_tree.begin(); it != new_tree.end(); ++it)
    {
        *it = new_values[m];
        ++m;
    }
    return new_tree;
}

sparse_block_tree_any_order_new sparse_block_tree_any_order_new::fuse(const sparse_block_tree_any_order_new& rhs) const
{
    std::vector<size_t> lhs_fuse_points(1,m_order - 1);
    std::vector<size_t> rhs_fuse_points(1,0);
    return fuse(rhs,lhs_fuse_points,rhs_fuse_points);
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

const char* sparse_block_tree_any_order_new::k_clazz = "sparse_block_tree_any_order";

} // namespace impl

} // namespace libtensor
