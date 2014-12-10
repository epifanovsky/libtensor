#include "sparse_loop_list.h"

using namespace std;

namespace libtensor
{

const char* sparse_loop_list::k_clazz = "sparse_loop_list";

sparse_loop_list::sparse_loop_list(const vector<sparse_bispace_impl>& bispaces,
                                   const vector<idx_pair_list>& ts_groups)
{
    for(size_t g_idx = 0; g_idx < ts_groups.size(); ++g_idx)
    {
        const idx_pair_list& grp = ts_groups[g_idx];
        const subspace& sub = bispaces[grp[0].first][grp[0].second];
        m_loops.push_back(block_loop(sub,grp));
    }
}

} /* namespace libtensor */


