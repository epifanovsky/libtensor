#ifndef SPARSITY_DATA_H
#define SPARSITY_DATA_H

#include "sparse_defs.h"
#include "runtime_permutation.h"

namespace libtensor {

class sparsity_data
{
private:
    std::vector< std::pair<idx_list,idx_list> > m_kv_pairs;
    size_t m_order;

    //Internal constructor used by permute and fuse
    sparsity_data(size_t order,
                  const std::vector< std::pair<idx_list,idx_list> >& kv_pairs) : m_order(order),
                                                                                 m_kv_pairs(kv_pairs) {}

public:



    static const char *k_clazz; //!< Class name
    sparsity_data(size_t order,const std::vector<idx_list>& keys);

    sparsity_data permute(const runtime_permutation& perm) const;
    sparsity_data contract(size_t contracted_subspace_idx) const;
    sparsity_data fuse(const sparsity_data& rhs,
                       const idx_list& lhs_indices,
                       const idx_list& rhs_indices) const;
    sparsity_data truncate_subspace(size_t subspace_idx,
                                    const idx_pair& bounds) const;
    sparsity_data insert_entries(size_t subspace_idx,
                                 const idx_list& entries) const;

    typedef std::vector< std::pair<idx_list,idx_list> >::iterator iterator;
    typedef std::vector< std::pair<idx_list,idx_list> >::const_iterator const_iterator;

    iterator begin() { return m_kv_pairs.begin(); }
    iterator end() { return m_kv_pairs.end(); }

    const_iterator begin() const { return m_kv_pairs.begin(); }
    const_iterator end() const { return m_kv_pairs.end(); }

    bool operator==(const sparsity_data& rhs) const;
    bool operator!=(const sparsity_data& rhs) const;
};

} // namespace libtensor

#endif /* SPARSITY_DATA_H */
