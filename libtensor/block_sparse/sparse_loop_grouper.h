#ifndef SPARSE_LOOP_GROUPER_H
#define SPARSE_LOOP_GROUPER_H

#include "sparsity_fuser.h"

namespace libtensor {

class sparse_loop_grouper
{
private:
    static const char* k_clazz; //!< Class name
    std::vector< std::vector<off_dim_pair_list> > m_offsets_and_sizes;
    std::vector<idx_pair_list> m_bispaces_and_index_groups;
public:
    sparse_loop_grouper(const sparsity_fuser& sf);
    size_t get_n_groups() const { return m_bispaces_and_index_groups.size(); }
    std::vector<idx_pair_list> get_bispaces_and_index_groups() const;
    std::vector<std::vector<off_dim_pair_list> > get_offsets_and_sizes() const;
};

} // namespace libtensor

#endif /* SPARSE_LOOP_GROUPER_H */
