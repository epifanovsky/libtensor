#include "block_loop.h"
#include "range.h"

using namespace std;

namespace libtensor
{

const char* block_loop::k_clazz = "block_loop";

block_loop::block_loop(const subspace& subspace,
                       const idx_pair_list& t_igs) : 
                   m_t_igs(t_igs),
                   m_block_inds(range(0,subspace.get_n_blocks())),
                   m_start_idx(0),
                   m_cur_idx(0)
{
    for(size_t i = 0; i < m_block_inds.size(); ++i)
    {
        m_block_szs.push_back(subspace.get_block_size(i));
        m_block_offs.push_back(idx_list());
        for(size_t t = 0; t < m_t_igs.size(); ++t)
        {
            m_block_offs.back().push_back(subspace.get_block_abs_index(i));
        }
    }
}
void block_loop::apply(vector<idx_list>& ig_offs,
                       vector<idx_list>& block_szs) const
{
    for(size_t i = 0; i < m_t_igs.size(); ++i)
    {
        size_t t_idx = m_t_igs[i].first;
        size_t ig_idx = m_t_igs[i].second;
        block_szs[t_idx][ig_idx] = m_block_szs[m_cur_idx];
        ig_offs[t_idx][ig_idx] *= m_block_offs[m_cur_idx][i];
        for(size_t f_ig_idx = ig_idx+1; f_ig_idx < ig_offs[t_idx].size(); ++f_ig_idx)
        {
            ig_offs[t_idx][f_ig_idx] *= m_block_szs[m_cur_idx];
        }
    }
}

block_loop& block_loop::operator++()
{
   ++m_cur_idx;
   return *this;
}

bool block_loop::done() const
{
    return (m_cur_idx == m_block_inds.size());
}

} /* namespace libtensor */

