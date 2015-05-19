#ifndef BLOCK_LOOP_H
#define BLOCK_LOOP_H

#include "subspace.h"
#include "sparsity_data.h"

namespace libtensor
{

class block_loop
{
private:
    static const char* k_clazz; //!< Class name

    bool m_sparse;
    bool m_done;
    size_t m_start_idx;
    size_t m_cur_idx;
    idx_list m_block_dims;
    idx_list m_block_inds;
    std::vector<idx_list> m_block_offs;
    idx_pair_list m_t_igs;
    idx_pair_list m_t_s;
    std::vector<bool> m_set_ig_off;
    block_loop* m_controlling_loop;
    block_loop* m_dependent_loop;
public:
    //Dense constructor
	block_loop(const subspace& subspace,
               const idx_pair_list& t_igs,
               const idx_pair_list& t_s);

    //Sparse constructor
	block_loop(const subspace& subspace,
               const idx_pair_list& t_igs,
               const idx_pair_list& t_s,
               const std::vector<bool>& set_ig_off,
               const sparsity_data& sd,
               size_t sd_sub,
               const idx_pair_list& sd_off_map);

    void apply_offsets(std::vector<idx_list>& ig_offs) const;
                       
    void apply_dims(std::vector<idx_list>& block_dims) const;

    void set_dependent_loop(block_loop& dep_loop);
                    
    block_loop& operator++();

    bool done() const;
    void reset();
};

} /* namespace libtensor */

#endif /* BLOCK_LOOP_H */
