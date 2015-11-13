#ifndef SPARSITY_DATA_H
#define SPARSITY_DATA_H

#include "sparse_defs.h"
#include "runtime_permutation.h"

namespace libtensor {

class sparsity_data
{
private:
    size_t m_order;
    size_t m_n_entries;
    size_t m_value_order;
    idx_list m_keys;
    idx_list m_values;
public:

    size_t get_order() const { return m_order; }
    size_t get_n_entries() const { return m_n_entries; }
    size_t get_value_order() const { return m_value_order; }
    const idx_list& get_keys() const { return m_keys; }
    const idx_list& get_values() const { return m_values; }
    void set_values(size_t value_order,const idx_list& values);

    static const char *k_clazz; //!< Class name

    sparsity_data(size_t order,const idx_list& keys,size_t value_order=0,const idx_list& values=idx_list());


    sparsity_data permute(const runtime_permutation& perm) const;
    sparsity_data contract(size_t contracted_subspace_idx) const;
    sparsity_data fuse(const sparsity_data& rhs,
                       const idx_list& lhs_indices,
                       const idx_list& rhs_indices) const;
    sparsity_data truncate_subspace(size_t subspace_idx,
                                    const idx_pair& bounds) const;
    sparsity_data insert_entries(size_t subspace_idx,
                                 const idx_list& entries) const;
    sparsity_data merge(const sparsity_data& other) const;

    bool operator==(const sparsity_data& rhs) const;
    bool operator!=(const sparsity_data& rhs) const;
};

} // namespace libtensor

#endif /* SPARSITY_DATA_H */
