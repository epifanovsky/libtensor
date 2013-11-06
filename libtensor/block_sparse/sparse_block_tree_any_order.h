#ifndef SPARSE_BLOCK_TREE_ANY_ORDER_H
#define SPARSE_BLOCK_TREE_ANY_ORDER_H

#include "sparse_block_tree_node.h"
#include "sparse_block_tree_iterator.h"
#include "runtime_permutation.h"

namespace libtensor { 

typedef std::vector<size_t> block_list;

//Forward declaration for set_offsets
template<size_t N>
class sparse_bispace;

//Forward declaration for friendship
template<size_t N>
class sparse_block_tree;

class sparse_block_tree_any_order
{
public:
    //typedefs needed for public and private methods - more public/private methods later
    typedef impl::sparse_block_tree_iterator<false> iterator;
    typedef impl::sparse_block_tree_iterator<true> const_iterator;
private:
    typedef std::vector<size_t> key_t;
    struct kv_pair_compare;
    impl::sparse_block_tree_node* m_node;
    size_t m_order;

    //Used by this class and derived class constructors 
    void init(const std::vector< key_t > sig_blocks);

    //Constructor
    //Used by internal methods to return appropriate instances
    sparse_block_tree_any_order(const std::vector< key_t > sig_blocks,size_t order); 

    //Needed for templated wrapper constructor - init is called later instead
    sparse_block_tree_any_order() { m_node = NULL; }

public:
    //Copy constructor
    sparse_block_tree_any_order(const sparse_block_tree_any_order& rhs) { m_node = rhs.m_node->clone(); m_order = rhs.m_order; }

    //Assignment operator
    sparse_block_tree_any_order& operator=(const sparse_block_tree_any_order& rhs) { delete m_node; m_node = rhs.m_node->clone(); m_order = rhs.m_order; }

    iterator begin() { return iterator(m_node,m_order); };
    const_iterator begin() const { return const_iterator(m_node,m_order); }
    iterator end() { return iterator(NULL,m_order); };
    const_iterator end() const { return const_iterator(NULL,m_order); }

    //Searching for a specific key
    //Use a vector because sometimes key lengths must be determined at runtime
    size_t search(const std::vector<size_t>& key) const;

    //Return a list of the blocks associated with a given sub-key
    const block_list& get_sub_key_block_list(const std::vector<size_t>& sub_key) const;

    //Can't use permutation<N> class because permutation degree may need to be determined at runtime
    sparse_block_tree_any_order permute(const runtime_permutation& perm) const;

    //Removes one of the levels of the tree and aggregates the remaining sub-keys to form a new tree 
    //Necessary to represent the tree resulting from the contraction of a sparse quantity
    sparse_block_tree_any_order contract(size_t contract_idx) const;

    //Fuses one sparse tree onto this one at position fuse_pos
    //By default, fuses to the branches of the tree
    sparse_block_tree_any_order fuse(const sparse_block_tree_any_order& rhs,const std::vector<size_t>& lhs_indices,
                                                                            const std::vector<size_t>& rhs_indices) const;
    //Convenience wrapper for the most common (end to end) case
    sparse_block_tree_any_order fuse(const sparse_block_tree_any_order& rhs) const;

    //Used to initialize the values of the tree to represent the offsets of the blocks in a bispace
    //Implemented in sparse_bispace.h to avoid incomplete type errors
    ///Returns the sum of the sizes of all blocks in the tree;
    //Must use vectors etc instead of compile time types like sequence because may not know length at compile time
    size_t set_offsets(const std::vector< sparse_bispace<1> >& subspaces,const std::vector<size_t>& positions); 


    size_t get_order() const { return m_order; } 
    bool operator==(const sparse_block_tree_any_order& rhs) const;
    bool operator!=(const sparse_block_tree_any_order& rhs) const;

    virtual ~sparse_block_tree_any_order() { delete m_node; }

    //Friend templated wrapper
    template<size_t N>
    friend class sparse_block_tree;
};

} // namespace libtensor


#endif /* SPARSE_BLOCK_TREE_ANY_ORDER_H */
