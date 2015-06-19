#ifndef SPARSE_LOOP_LIST_H
#define SPARSE_LOOP_LIST_H

#include "sparse_bispace.h"
#include "block_kernel_i.h"
#include "block_loop.h"

namespace libtensor
{


class sparse_loop_list
{
private:
    static const char* k_clazz; //!< Class name
    std::vector<block_loop> m_loops;
    std::vector<idx_list> m_ig_offs;
    std::vector<idx_list> m_block_dims;
public:
    sparse_loop_list(const std::vector<sparse_bispace_impl>& bispaces,
                     const std::vector<idx_pair_list>& ts_groups);

    template<typename kern_t,typename T>
    void run(kern_t& kernel,
             const std::vector<T*> ptrs);
             
};

template<typename kern_t,typename T>
void sparse_loop_list::run(kern_t& kernel,
                           const std::vector<T*> ptrs)
{
    std::vector<T*> block_ptrs(ptrs.size());

    std::vector< std::vector<idx_list> > ig_off_grps(m_loops.size());

    size_t c_loop_idx = 0;
    while(!(m_loops[0].done() && c_loop_idx == 0))
    {
        block_loop& c_loop = m_loops[c_loop_idx];
        if(c_loop_idx == 0)
            ig_off_grps[0] = m_ig_offs;
        else
            ig_off_grps[c_loop_idx] = ig_off_grps[c_loop_idx-1];

        if(!c_loop.done())
        {
            c_loop.apply_offsets(ig_off_grps[c_loop_idx]);
            c_loop.apply_dims(m_block_dims);

            if(c_loop_idx == m_loops.size() - 1)
            {
                for(size_t t_idx = 0; t_idx < ptrs.size(); ++t_idx)
                {
                    size_t offset = 0;
                    for(size_t ig = 0; ig < m_ig_offs[t_idx].size(); ++ig)
                    {
                        offset += ig_off_grps[c_loop_idx][t_idx][ig];
                    }
                    block_ptrs[t_idx] = ptrs[t_idx] + offset;
                }
                kernel(block_ptrs,m_block_dims);
                ++c_loop;
            }
            else
                m_loops[++c_loop_idx].reset();
        }
        else
            ++m_loops[--c_loop_idx];
    }
}

} /* namespace libtensor */


#endif /* SPARSE_LOOP_LIST_H */
