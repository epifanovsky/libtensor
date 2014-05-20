#include "sparse_block_tree.h"
#include "sparse_block_tree_iterator.h"
#include "range.h"
#include "sparse_bispace.h"

using namespace std;

namespace libtensor {

//TODO: DEBUG REMOVE
/*template<typename T>*/
/*T read_timer();*/


//Used to return empty trees by sub_tree
static const sparse_block_tree_any_order empty = sparse_block_tree<1>(vector< sequence<1,size_t> >(),vector< sparse_bispace<1> >(1,sparse_bispace<1>(1)));

void sparse_block_tree_any_order::set_offsets_sizes_nnz(const vector< sparse_bispace<1> >& subspaces)
{
    if(subspaces.size() != m_order)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_any_order","set_offsets_sizes_nnz",
                            __FILE__,__LINE__,"Wrong number of subspaces specified!");
    }

    size_t offset = 0;
    for(iterator it = begin(); it != end(); ++it)
    {
        vector<key_t> key = it.key();
        size_t size = 1;
        for(size_t i = 0; i < m_order; ++i)
        {
            size *= subspaces[i].get_block_size(key[i]);
        }
        *it = value_t(1,make_pair(offset,size));
        offset += size;
    }
    
    m_nnz = offset;
}

//Assignment operator
sparse_block_tree_any_order& sparse_block_tree_any_order::operator=(const sparse_block_tree_any_order& rhs)
{
    if(this != &rhs)
    {
        //First delete old children and values
        for(size_t i = 0; i < m_children.size(); ++i)
        {
            delete m_children[i];
        }

        m_order = rhs.m_order;
        m_nnz = rhs.m_nnz;
        m_n_entries = rhs.m_n_entries;
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
                m_children[i] = new sparse_block_tree_any_order(*rhs.m_children[i]);
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

//Copy constructor
sparse_block_tree_any_order::sparse_block_tree_any_order(const sparse_block_tree_any_order& rhs) : m_keys(rhs.m_keys),
                                                                                                               m_values(rhs.m_values)
{
    m_order = rhs.m_order;
    m_n_entries = rhs.m_n_entries;
    m_nnz = rhs.m_nnz;

    if(m_order > 1)
    {
        m_children.resize(rhs.m_children.size());
        for(size_t i = 0; i < m_children.size(); ++i)
        {
            m_children[i] = new sparse_block_tree_any_order(*rhs.m_children[i]);
        }
    } 
}

//Destructor
sparse_block_tree_any_order::~sparse_block_tree_any_order() 
{
    for(size_t i = 0; i < m_children.size(); ++i)
    {
        delete m_children[i];
    }
}


sparse_block_tree_any_order::sparse_block_tree_any_order(const vector< vector<key_t> >& sig_blocks,size_t order)
{
    m_order = order;
    m_n_entries = 0;
    for(size_t i = 0; i < sig_blocks.size(); ++i)
    {
        push_back(sig_blocks[i],m_order);
    }
}

struct sparse_block_tree_any_order::kv_pair_compare {

    bool operator()(const pair< vector<key_t>,value_t>& p1,const pair< vector<key_t>,value_t>& p2)
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

const sparse_block_tree_any_order& sparse_block_tree_any_order::get_sub_tree(const vector<key_t>& sub_key) const
{
    static const char *method = "get_sub_tree(const vector<key_t>& sub_key) const";

    if((sub_key.size() >= m_order) || (sub_key.size() == 0))
    {
        throw bad_parameter(g_ns,k_clazz,method,__FILE__,__LINE__,"key is wrong size"); 
    }

    //Find the position of the current key in this node's list of keys
    const sparse_block_tree_any_order* cur_node = this;
    for(size_t i = 0; i < sub_key.size(); ++i)
    {
        vector<key_t>::const_iterator cur_pos = lower_bound(cur_node->m_keys.begin(),cur_node->m_keys.end(),sub_key[i]);
        if(cur_pos == cur_node->m_keys.end() || *cur_pos != sub_key[i])
        {
            return empty;
        }
        cur_node = cur_node->m_children[distance(cur_node->m_keys.begin(),cur_pos)];
    }
    return *cur_node;
}

sparse_block_tree_any_order sparse_block_tree_any_order::permute(const runtime_permutation& perm) const
{
    vector< pair< vector<key_t>, value_t > > kv_pairs;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        vector<key_t> new_key = it.key(); 
        perm.apply(new_key);
        kv_pairs.push_back(make_pair(new_key,value_t(*it)));
    }

    sort(kv_pairs.begin(),kv_pairs.end(),kv_pair_compare());

    vector< vector<key_t> > all_keys;
    vector< value_t > all_vals;
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

sparse_block_tree_any_order sparse_block_tree_any_order::contract(size_t contract_idx,const vector< sparse_bispace<1> >& subspaces) const
{
    if(subspaces.size() != m_order - 1)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_any_order","contract(...)",
            __FILE__,__LINE__,"not enough subspaces specified"); 
    }

    vector< key_vec > contracted_keys;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        key_vec new_key(it.key());
        new_key.erase(new_key.begin()+contract_idx);
        contracted_keys.push_back(new_key);
    }
    sort(contracted_keys.begin(),contracted_keys.end());


    //Filter the unique keys only
    vector<key_vec> unique_keys;
    for(size_t i = 0; i < contracted_keys.size(); ++i)
    {
        if(i > 0)
        {
            if(contracted_keys[i] == contracted_keys[i-1])
            {
                continue;
            }
        }
        unique_keys.push_back(contracted_keys[i]);
    }


    sparse_block_tree_any_order sbt(unique_keys,m_order - 1);
    sbt.set_offsets_sizes_nnz(subspaces);
    return sbt;
}

sparse_block_tree_any_order sparse_block_tree_any_order::fuse(const sparse_block_tree_any_order& rhs,const vector<size_t>& lhs_indices,const vector<size_t>& rhs_indices) const
{
    /*double seconds = read_timer<double>();*/

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
    vector< vector<key_t> > new_keys;
    vector< value_t > new_values;

    //Permute RHS to bring the fused indices to the left-most position...
    size_t rhs_order = rhs.m_order;
    vector<size_t> permutation_entries;
    vector<size_t> indices_to_erase;
    for(size_t rhs_fused_idx_incr = 0; rhs_fused_idx_incr < rhs_indices.size(); ++rhs_fused_idx_incr)
    {
    	size_t rhs_fused_idx = rhs_indices[rhs_fused_idx_incr];
    	permutation_entries.push_back(rhs_fused_idx);
    	indices_to_erase.push_back(rhs_fused_idx);
    }

    //Erase indices that are fused in reverse order
    sort(indices_to_erase.begin(),indices_to_erase.end());
    vector<size_t> rhs_unfused_inds(range(0,rhs_order));
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
    const sparse_block_tree_any_order& rhs_permuted = (perm == runtime_permutation(rhs_order)) ? rhs : rhs.permute(perm);

    for(const_iterator it = begin(); it != end(); ++it)
    {
        vector<key_t> new_key(out_order);
        value_t new_value(*it);
        size_t lhs_val_size = new_value.size();

        //The first part of our new key is just the old left hand side key - we copy this part now
        vector<key_t> base_key = it.key();
        for(size_t new_key_idx = 0; new_key_idx < m_order; ++new_key_idx)
        {
            new_key[new_key_idx] = base_key[new_key_idx];
        }

        //Extract the fusing portion of the LHS key that determines what RHS keys to include
        vector<key_t> sub_key(n_fused_inds);
        for(size_t lhs_idx = 0; lhs_idx < n_fused_inds; ++lhs_idx)
        {
            sub_key[lhs_idx] = base_key[lhs_indices[lhs_idx]];
        }

        //There is no sub-tree to find if we are fusing ALL of the RHS indices
        //We only care about the existence of the key
        const_iterator start_it(this);
        const_iterator end_it(this);
        if(rhs_indices.size() == rhs.m_order)
        {
            start_it = rhs.search(sub_key);
            if(start_it != rhs.end())
            {
                end_it = ++const_iterator(start_it);
            }
            else
            {
                end_it = rhs.end();
            }
        }
        else
        {
            const sparse_block_tree_any_order& st = rhs_permuted.get_sub_tree(sub_key);
            start_it = st.begin(); 
            end_it = st.end();
        }
        //Attach each relevant sub key to the attachment point
        while(start_it != end_it)
        {
            if(rhs_indices.size() != rhs.m_order)
            {
                vector<key_t> rhs_key = start_it.key();

                //Now fill in the right side of the key, everything after the fused indices 
                for(size_t rhs_idx = 0; rhs_idx < rhs_key.size(); ++rhs_idx)
                {
                    new_key[m_order + rhs_idx] = rhs_key[rhs_idx];
                }
            }
            new_keys.push_back(new_key);

            //We merge the data stored in both trees because this is most commonly what is
            //needed for storing tensor offsets
            const value_t& rhs_val = *start_it;
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
            new_values.push_back(new_value);
            ++start_it;
        }
    }

    //By virtue of both trees being sorted, the list will be sorted already
    sparse_block_tree_any_order new_tree(new_keys,out_order);
    size_t m = 0; 
    for(sparse_block_tree_any_order::iterator it = new_tree.begin(); it != new_tree.end(); ++it)
    {
        *it = new_values[m];
        ++m;
    }
    /*std::cout << "Time inside fuse: " << read_timer<double>() - seconds << "\n";*/
    return new_tree;
}

sparse_block_tree_any_order sparse_block_tree_any_order::fuse(const sparse_block_tree_any_order& rhs) const
{
    vector<size_t> lhs_fuse_points(1,m_order - 1);
    vector<size_t> rhs_fuse_points(1,0);
    return fuse(rhs,lhs_fuse_points,rhs_fuse_points);
}


sparse_block_tree_any_order::const_iterator sparse_block_tree_any_order::search(const vector<size_t>& key) const
{
    if(key.size() != m_order)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_any_order","search(...)",
            __FILE__,__LINE__,"key length does not match depth of tree"); 
    }

    const sparse_block_tree_any_order* cur_node = this;
    vector<size_t> displacement(m_order);
    for(size_t i = 0; i < key.size(); ++i)
    {
        vector<key_t>::const_iterator cur_pos = lower_bound(cur_node->m_keys.begin(),cur_node->m_keys.end(),key[i]);
        if(cur_pos == cur_node->m_keys.end() || *cur_pos != key[i])
        {
            return end();
        }
        displacement[i] = distance(cur_node->m_keys.begin(),cur_pos);
        if(cur_node->m_order > 1)
        {
            cur_node = cur_node->m_children[displacement[i]];
        }
        else
        {
            break;
        }
    }

    /*cout << "displacement: ";*/
    /*for(size_t j = 0; j < displacement.size(); ++j)*/
    /*{*/
        /*cout << displacement[j] << ",";*/
    /*}*/
    /*cout << "\n";*/
    const_iterator it(this,displacement);
    /*cout << "key: ";*/
    /*for(size_t j = 0; j < it.key().size(); ++j)*/
    /*{*/
        /*cout << it.key()[j] << ",";*/
    /*}*/
    /*cout << "\n";*/
    /*cout << "val: " << (*it)[0] << "\n";*/
    /*cout << "m_keys: "; */
    /*for(size_t j = 0; j < m_keys.size(); ++j)*/
    /*{*/
        /*cout << m_keys[j] << ",";*/
    /*}*/

    return it;
}

sparse_block_tree_iterator<false> sparse_block_tree_any_order::begin()
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

sparse_block_tree_iterator<false> sparse_block_tree_any_order::end()
{
    return iterator(NULL);
}


//TODO: merge const and non-const functions
sparse_block_tree_iterator<true> sparse_block_tree_any_order::begin() const
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

sparse_block_tree_iterator<true> sparse_block_tree_any_order::end() const
{
    return const_iterator(NULL);
}

bool sparse_block_tree_any_order::operator==(const sparse_block_tree_any_order& rhs) const
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

bool sparse_block_tree_any_order::operator!=(const sparse_block_tree_any_order& rhs) const
{
    return !(*this == rhs);
}


sparse_block_tree_any_order sparse_block_tree_any_order::truncate_subspace(size_t subspace_idx,const idx_pair& subspace_bounds) const
{
    vector< vector<key_t> > new_keys; 
    vector<value_t> new_values; 

    size_t min = subspace_bounds.first;
    size_t max = subspace_bounds.second;
    for(const_iterator it = begin(); it != end(); ++it)
    {
        vector<key_t> cur_key = it.key();
        if((min <= cur_key[subspace_idx]) && (cur_key[subspace_idx] < max))
        {
            new_keys.push_back(cur_key);
            new_values.push_back(*it);
        }
    }
    
    //Preserve the original values in the tree
    sparse_block_tree_any_order sbt(new_keys,m_order);
    size_t m = 0;
    for(iterator it = sbt.begin(); it != sbt.end(); ++it)
    {
        *it = new_values[m];
        ++m;
    }
    return sbt;
}

sparse_block_tree_any_order sparse_block_tree_any_order::insert_subspace(size_t subspace_idx,const sparse_bispace<1>& subspace) const
{
    if(subspace_idx > m_order)
    {
        throw out_of_bounds(g_ns,"sparse_block_tree_any_order","insert_subspace(...)",__FILE__,__LINE__,
                "subspace idx out of bounds"); 
    }

    vector< vector<size_t> > all_keys;
    vector< vector<size_t> > batch_keys;
    const_iterator it = begin();
    vector<size_t> outer_inds(subspace_idx);
    while(it != end())
    {
        vector<size_t> key = it.key();
        //Did we come to a new batch?
        bool new_batch = false;
        if(batch_keys.size() == 0)
        {
            new_batch = true;
        }
        else
        {
            for(size_t i = 0; i < subspace_idx; ++i)
            { 
                if(key[i] != outer_inds[i])
                {
                    new_batch = true;
                    break;
                }
            }
        }

        if(new_batch)
        {
            //Process the old batch, if this is not the first batch
            if(batch_keys.size() > 0)
            {
                for(size_t block_idx = 0; block_idx < subspace.get_n_blocks(); ++block_idx)
                {
                    for(size_t key_idx = 0; key_idx < batch_keys.size(); ++key_idx)
                    {
                        vector<size_t> base_key = batch_keys[key_idx];
                        base_key.insert(base_key.begin()+subspace_idx,block_idx);
                        all_keys.push_back(base_key);
                    }
                }
            }

            //Start the new batch
            for(size_t i = 0; i < subspace_idx; ++i)
            { 
                outer_inds[i] = key[i];
            }
            batch_keys = std::vector< vector<size_t> >(1,key);
        }
        else
        {
            batch_keys.push_back(key);
        }

        ++it;
    }

    //Handle the last batch
    for(size_t block_idx = 0; block_idx < subspace.get_n_blocks(); ++block_idx)
    {
        for(size_t key_idx = 0; key_idx < batch_keys.size(); ++key_idx)
        {
            vector<size_t> base_key = batch_keys[key_idx];
            base_key.insert(base_key.begin()+subspace_idx,block_idx);
            all_keys.push_back(base_key);
        }
    }

    sparse_block_tree_any_order sbt(all_keys,m_order+1);
    return sbt;
}

const char* sparse_block_tree_any_order::k_clazz = "sparse_block_tree_any_order";

} // namespace libtensor
