#include "sparse_bispace_impl.h"

using namespace std;

namespace libtensor {
    
const char* sparse_bispace_impl::k_clazz = "sparse_bispace_impl";

bool sparse_bispace_impl::operator==(const sparse_bispace_impl& rhs) const
{
    return (m_subspaces == rhs.m_subspaces) && (m_trees == rhs.m_trees);
}

sparse_bispace_impl::sparse_bispace_impl(const sparse_bispace_impl& lhs,
                                         const sparse_bispace_impl& rhs)
{
    m_subspaces.insert(m_subspaces.end(),lhs.m_subspaces.begin(),lhs.m_subspaces.end());
    m_subspaces.insert(m_subspaces.end(),rhs.m_subspaces.begin(),rhs.m_subspaces.end());
    m_trees.insert(m_trees.end(),lhs.m_trees.begin(),lhs.m_trees.end());
    m_trees.insert(m_trees.end(),rhs.m_trees.begin(),rhs.m_trees.end());
}

sparse_bispace_impl::sparse_bispace_impl(const vector<subspace>& subspaces,
                                         const sparse_block_tree& tree) : m_subspaces(subspaces),m_trees(1,tree)
{
}

} // namespace libtensor
