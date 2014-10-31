#ifndef SPARSE_BISPACE_IMPL_H
#define SPARSE_BISPACE_IMPL_H

#include "subspace.h"
#include "sparsity_data.h"
#include "runtime_permutation.h"

namespace libtensor {

class sparse_bispace_impl
{
private:
    std::vector<subspace> m_subspaces;
    std::vector<sparsity_data> m_group_sd;
    idx_list m_group_offsets;
public:
    static const char* k_clazz; //!< Class name

    sparse_bispace_impl(const std::vector<subspace>& subspaces) : m_subspaces(subspaces) {};

    sparse_bispace_impl(const std::vector<subspace>& subspaces,
                        const std::vector<sparsity_data>& group_sd,
                        const idx_list& group_offsets);
                        
    sparse_bispace_impl permute(const runtime_permutation& perm) const;

    bool operator==(const sparse_bispace_impl& rhs) const;
    bool operator!=(const sparse_bispace_impl& rhs) const { return !(*this == rhs); }
};

} // namespace libtensor


#endif /* SPARSE_BISPACE_IMPL_H */
