#include "sparse_loop_list.h"

using namespace std;

namespace libtensor
{

const char* sparse_loop_list::k_clazz = "sparse_loop_list";

sparse_loop_list::sparse_loop_list(const vector<sparse_bispace_impl>& bispaces,
                                   const vector<idx_pair_list>& ts_groups) : m_block_dims(bispaces.size())
{
    vector<bool> set_ig_off(ts_groups.size(),false);
    for(size_t g_idx = 0; g_idx < ts_groups.size(); ++g_idx)
    {
        const idx_pair_list& ts_grp = ts_groups[g_idx];
        const subspace& sub = bispaces[ts_grp[0].first][ts_grp[0].second];
        idx_pair_list ig_grp(ts_grp.size());
        for(size_t i = 0; i < ts_grp.size(); ++i)
        {
            size_t t = ts_grp[i].first;
            size_t s = ts_grp[i].second;
            size_t ig = bispaces[t].get_ig_containing_subspace(s);
            ig_grp[i] = idx_pair(t,ig);
            if(s == bispaces[t].get_ig_offset(ig)+bispaces[t].get_ig_order(ig) - 1)
            {
                set_ig_off[g_idx] = true;
            }
        }
        m_loops.push_back(block_loop(sub,ig_grp,ts_grp));
    }
    
    for(size_t t_idx = 0; t_idx < bispaces.size(); ++t_idx)
    {
        size_t n_ig = bispaces[t_idx].get_n_ig();
        idx_list ig_off_grp(n_ig);
        size_t inner_sz = 1;
        for(size_t ig_rev = 0; ig_rev < n_ig; ++ig_rev)
        {
            size_t ig = bispaces[t_idx].get_n_ig() - ig_rev - 1;
            size_t ig_dim = bispaces[t_idx].get_ig_dim(ig);
            ig_off_grp[ig] = inner_sz;
            inner_sz *= ig_dim;
        }
        m_ig_offs.push_back(ig_off_grp);
        m_block_dims[t_idx].resize(bispaces[t_idx].get_order());
    }
}

} /* namespace libtensor */


