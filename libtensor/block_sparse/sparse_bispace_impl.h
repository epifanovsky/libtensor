#ifndef SPARSE_BISPACE_IMPL_H
#define SPARSE_BISPACE_IMPL_H

#include "subspace.h"
#include "sparse_block_tree.h"

namespace libtensor {

class sparse_bispace_impl
{
private:
    std::vector<subspace> m_subspaces;
public:
    static const char* k_clazz; //!< Class name
    std::vector<sparse_block_tree> m_trees;

    //Constructor called to create a fully dense composite bispace
    sparse_bispace_impl(const std::vector<subspace>& subspaces);

    //Constructor called to create a single sparse subspace group
    sparse_bispace_impl(const std::vector<subspace>& subspaces,
                        const sparse_block_tree& tree);
                        

    /** \brief Returns whether this object is equal to another of the same dimension. 
     *         Two N-D spaces are equal if:
     *              1. Their vectors of subspaces are equal
     *              2. Their sparsity metadata is equal
     **/
    bool operator==(const sparse_bispace_impl& rhs) const;

};

} // namespace libtensor


#endif /* SPARSE_BISPACE_IMPL_H */
