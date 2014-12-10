#ifndef BLOCK_LOOP_H
#define BLOCK_LOOP_H

#include "subspace.h"

namespace libtensor
{

class block_loop
{
private:
    static const char* k_clazz; //!< Class name

    size_t m_start_idx;
    size_t m_cur_idx;
    idx_list m_block_dims;
    idx_list m_block_inds;
    std::vector<idx_list> m_block_offs;
    idx_pair_list m_t_igs;
public:
    //Dense constructor
	block_loop(const subspace& subspace,
               const idx_pair_list& t_igs);

    void apply(std::vector<idx_list>& ig_offs,
               std::vector<idx_list>& block_szs) const;

    block_loop& operator++();

    bool done() const;
    void reset();
};

} /* namespace libtensor */

#endif /* BLOCK_LOOP_H */
