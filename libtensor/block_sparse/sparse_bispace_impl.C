#include "sparse_bispace_impl.h"

namespace libtensor {
    
const char* sparse_bispace_impl::k_clazz = "sparse_bispace_impl";

bool sparse_bispace_impl::operator==(const sparse_bispace_impl& rhs) const
{
    return (m_subspaces == rhs.m_subspaces) && (m_trees == rhs.m_trees);
}

sparse_bispace_impl::sparse_bispace_impl(const std::vector<subspace>& subspaces) : m_subspaces(subspaces)
{
}

    //Constructor called to create a single sparse subspace group
sparse_bispace_impl::sparse_bispace_impl(const std::vector<subspace>& subspaces,
                    const sparse_block_tree& tree) : m_subspaces(subspaces),m_trees(1,tree)
{
}

} // namespace libtensor
