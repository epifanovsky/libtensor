#ifndef SPARSE_BISPACE_IMPL_H
#define SPARSE_BISPACE_IMPL_H

#include "subspace.h"
#include "sparsity_data.h"
#include "runtime_permutation.h"

namespace libtensor {

class sparse_bispace_impl
{
private:
    void init_ig();
protected:
    static const char* k_clazz; //!< Class name
    std::vector<subspace> m_subspaces;
    std::vector<sparsity_data> m_group_sd;
    idx_list m_group_offsets;
    idx_list m_ig_offsets;
    idx_list m_ig_dims;
public:

    sparse_bispace_impl(const std::vector<subspace>& subspaces) : m_subspaces(subspaces) 
    { 
        init_ig();
    }

    //Called by permute, contract
    sparse_bispace_impl(const std::vector<subspace>& subspaces,
                        const std::vector<sparsity_data>& group_sd,
                        const idx_list& group_offsets);

    //Called by symbolic operator|
    sparse_bispace_impl(const sparse_bispace_impl& lhs,
                        const sparse_bispace_impl& rhs);

    size_t get_order() const { return m_subspaces.size(); }
                        
    sparse_bispace_impl permute(const runtime_permutation& perm) const;
    sparse_bispace_impl contract(size_t contract_idx) const;

    size_t get_n_ig() const { return m_ig_dims.size(); }
    size_t get_ig_offset(size_t grp_idx) const { return m_ig_offsets[grp_idx]; }
    size_t get_ig_dim(size_t grp_idx) const { return m_ig_dims[grp_idx]; }
    size_t get_ig_order(size_t grp_idx) const;
    size_t get_ig_containing_subspace(size_t subspace_idx) const;

    bool operator==(const sparse_bispace_impl& rhs) const;
    bool operator!=(const sparse_bispace_impl& rhs) const { return !(*this == rhs); }
    subspace operator[](size_t idx) const { return m_subspaces[idx]; }

    template<size_t P,size_t Q>
    friend class sparsity_expr;
};

} // namespace libtensor


#endif /* SPARSE_BISPACE_IMPL_H */
