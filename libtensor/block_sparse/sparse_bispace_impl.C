#include "sparse_bispace_impl.h"

namespace libtensor {
    
const char* sparse_bispace_impl::k_clazz = "sparse_bispace_impl";

bool sparse_bispace_impl::operator==(const sparse_bispace_impl& rhs) const
{
    return m_subspaces == rhs.m_subspaces;
}

} // namespace libtensor
