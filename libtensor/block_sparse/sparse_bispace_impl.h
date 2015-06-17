#ifndef SPARSE_BISPACE_IMPL_H
#define SPARSE_BISPACE_IMPL_H

#include "subspace.h"

namespace libtensor {

class sparse_bispace_impl
{
private:
    std::vector<subspace> m_subspaces;
public:
    static const char* k_clazz; //!< Class name

    sparse_bispace_impl(const std::vector<subspace>& subspaces) : m_subspaces(subspaces) { }

    /** \brief Returns whether this object is equal to another of the same dimension. 
     *         Two N-D spaces are equal if:
     *              1. Their vectors of subspaces are equal
     *              2. Their sparsity metadata is equal
     **/
    bool operator==(const sparse_bispace_impl& rhs) const;

};

} // namespace libtensor


#endif /* SPARSE_BISPACE_IMPL_H */
