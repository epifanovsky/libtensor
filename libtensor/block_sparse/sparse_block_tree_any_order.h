#ifndef SPARSE_BLOCK_TREE_ANY_ORDER_H
#define SPARSE_BLOCK_TREE_ANY_ORDER_H

#include <vector>
#include "../core/sequence.h"
#include "runtime_permutation.h" 

namespace libtensor { 

class loop_list_sparsity_data;

template<size_t N>
class sparse_bispace;

template<bool is_const>
class sparse_block_tree_iterator;

class sparse_block_tree_any_order
{
public:
    typedef size_t key_t;
    typedef std::vector<key_t> key_vec;
    typedef std::vector< std::pair<size_t,size_t> > value_t;
    typedef sparse_block_tree_iterator<false> iterator;
    typedef sparse_block_tree_iterator<true> const_iterator;

    //Copy constructor
    sparse_block_tree_any_order(const sparse_block_tree_any_order& rhs);

    //Assignment operator
    sparse_block_tree_any_order& operator=(const sparse_block_tree_any_order& rhs);

    //Destructor
    virtual ~sparse_block_tree_any_order();

    //Returns the child/granchild/etc corresponding to the specified sub-key
    const sparse_block_tree_any_order& get_sub_tree(const std::vector<key_t>& sub_key) const;

    //Can't use permutation<N> class because permutation degree may need to be determined at runtime
    sparse_block_tree_any_order permute(const runtime_permutation& perm) const;

    //Removes one of the levels of the tree and aggregates the remaining sub-keys to form a new tree 
    //The offsets and sizes of the blocks in the new tree are generated from the provided subspaces
    sparse_block_tree_any_order contract(size_t contract_idx,const std::vector< sparse_bispace<1> >& subspaces) const;

    //Fuses one sparse tree onto this one at position fuse_pos
    //By default, fuses to the branches of the tree
    sparse_block_tree_any_order fuse(const sparse_block_tree_any_order& rhs,const std::vector<size_t>& lhs_indices,
                                                                                    const std::vector<size_t>& rhs_indices) const;
    //Convenience wrapper for the most common (end to end) case
    sparse_block_tree_any_order fuse(const sparse_block_tree_any_order& rhs) const;

    //Searching for a specific key
    //Use a vector because sometimes key lengths must be determined at runtime
    const_iterator search(const std::vector<size_t>& key) const;

    //Used to initialize the values of the tree to the offsets and sizes of the blocks, also sets nnz
    void set_offsets_sizes_nnz(const std::vector< sparse_bispace<1> >& subspaces);

    //Return the number of nonzero tensor elements corresponding to this tree
    size_t get_nnz() const { return m_nnz; }
    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;

    size_t get_n_entries() const { return m_n_entries; }
    size_t get_order() const { return m_order; }

    bool operator==(const sparse_block_tree_any_order& rhs) const;
    bool operator!=(const sparse_block_tree_any_order& rhs) const;

    template<bool is_const>
    friend class sparse_block_tree_iterator;

    friend class libtensor::loop_list_sparsity_data;

protected:

    //We don't want these to be directly instantiable - force people to use the templated interface instead
    //This is called by the order-templated subclass
    template<size_t N>
    sparse_block_tree_any_order(const std::vector< sequence<N,key_t> >& sig_blocks,const std::vector< sparse_bispace<1> >& subspaces);
private:
    //Utility struct used to implement permute
    struct kv_pair_compare;

    size_t m_n_entries;
    size_t m_order;
    std::vector<key_t> m_keys;
    //For branch nodes only
    std::vector< sparse_block_tree_any_order* > m_children;
    //For leaf nodes only
    std::vector<value_t> m_values;
    size_t m_nnz;

    //Used by primary constructor to add new entries 
    //Templated on container type for use with both vectors and sequences
    template<typename container>
    void push_back(const container& key,size_t key_order);

    //Used by push_back to create new branch nodes below the root
    template<typename container>
    sparse_block_tree_any_order(const container& key,size_t key_order,size_t order);

    //Used by permute/fuse/contract to create new instances
    //Does not do the same input validation as primary constructor 
    sparse_block_tree_any_order(const std::vector< std::vector<key_t> >& sig_blocks,size_t order);


    static const char *k_clazz; //!< Class name
};

template<typename container>
sparse_block_tree_any_order::sparse_block_tree_any_order(const container& key,size_t key_order,size_t node_order)
{
    m_order = node_order;
    m_n_entries = 0;
    push_back(key,key_order);
}

template<typename container>
void sparse_block_tree_any_order::push_back(const container& key,size_t key_order)
{
    //Add this key to this node if we don't already have it
    //Guaranteed sorted by constructor so only need to compare to the last key
    key_t key_entry = key[key_order - m_order];
    if((m_keys.size() == 0) || (m_keys.back() != key_entry))
    {
        m_keys.push_back(key_entry);
        if(m_order > 1)
        {
            m_children.push_back(new sparse_block_tree_any_order(key,key_order,m_order - 1));
        }
        else
        {
            m_values.push_back(value_t());
        }
    }
    else
    {
        m_children.back()->push_back(key,key_order);
    }
    ++m_n_entries;
}

template<size_t N>
sparse_block_tree_any_order::sparse_block_tree_any_order(const std::vector< sequence<N,key_t> >& sig_blocks,const std::vector< sparse_bispace<1> >& subspaces)
{
    if(N == 0)
    {
        throw bad_parameter(g_ns,"sparse_block_tree_node","sparse_block_tree_node(...)",
                            __FILE__,__LINE__,"cannot pass 0 dimensional sequence as argument");
    }
    m_order = N;
    m_n_entries = 0;

    //Ensure that block list is sorted in lexicographic order
    size_t offset = 0;
    for(size_t i = 0; i < sig_blocks.size(); ++i)
    {
        const sequence<N,key_t>& cur = sig_blocks[i];
        if(i > 0)
        {
            const sequence<N,key_t>& prev = sig_blocks[i-1];

            bool equal = true;
            for(size_t j = 0; j < m_order; ++j)
            {
                if(cur[j] < prev[j])
                {
                    throw bad_parameter(g_ns,"sparse_block_tree_node","sparse_block_tree_node(...)",
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
        }

        push_back(cur,m_order);
    }

    set_offsets_sizes_nnz(subspaces);
}

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_ANY_ORDER_H */
