#ifndef SPARSE_BISPACE_IMPL_H
#define SPARSE_BISPACE_IMPL_H

#include "subspace.h"
#include "sparse_block_tree.h"
#include "runtime_permutation.h"

namespace libtensor {

class sparse_bispace_impl
{
private:
    std::vector<subspace> m_subspaces;
    std::vector<sparse_block_tree> m_trees;
    std::vector<size_t> m_tree_offsets;
public:
    static const char* k_clazz; //!< Class name

    //1D special case
    sparse_bispace_impl(const subspace& subspace_0) : m_subspaces(1,subspace_0) {};

    //Constructor called by '%' operator of sparse_bispace 
    sparse_bispace_impl(const std::vector<subspace>& subspaces,
                        const sparse_block_tree& tree);

    //Constructor called by '|' operator of sparse_bispace
    sparse_bispace_impl(const sparse_bispace_impl& lhs,
                        const sparse_bispace_impl& rhs);
                        
    /** \brief Returns an appropriately permuted copy of this bispace 
     **/
    sparse_bispace_impl permute(const runtime_permutation& perm) const;

    /** \brief Returns whether this object is equal to another of the same dimension. 
     *         Two N-D spaces are equal if:
     *              1. Their vectors of subspaces are equal
     *              2. Their sparsity metadata is equal
     **/
    bool operator==(const sparse_bispace_impl& rhs) const;
    bool operator!=(const sparse_bispace_impl& rhs) const { return !(*this == rhs); }
};

} // namespace libtensor


#endif /* SPARSE_BISPACE_IMPL_H */
