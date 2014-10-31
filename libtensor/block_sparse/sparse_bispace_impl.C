#include "sparse_bispace_impl.h"
#include "range.h"

using namespace std;

namespace libtensor {
    
const char* sparse_bispace_impl::k_clazz = "sparse_bispace_impl";

sparse_bispace_impl::sparse_bispace_impl(const vector<subspace>& subspaces,
                                         const vector<sparsity_data>& group_sd,
                                         const idx_list& group_offsets) : m_subspaces(subspaces),
                                                                          m_group_sd(group_sd),
                                                                          m_group_offsets(group_offsets)
{
}

bool sparse_bispace_impl::operator==(const sparse_bispace_impl& rhs) const
{
    return (m_subspaces == rhs.m_subspaces) && 
           (m_group_sd == rhs.m_group_sd) && 
           (m_group_offsets == rhs.m_group_offsets);
}

sparse_bispace_impl sparse_bispace_impl::permute(const runtime_permutation& perm) const
{
    vector<subspace> p_subspaces(m_subspaces);
    for(size_t i = 0; i < m_subspaces.size(); ++i)
    {
        p_subspaces[i] = m_subspaces[perm[i]];
    }
   
    idx_list inv_p_ent(m_subspaces.size());
    for(size_t i = 0; i < m_subspaces.size(); ++i) inv_p_ent[perm[i]] = i;
    runtime_permutation inv_p(inv_p_ent);

    vector<sparsity_data> p_group_sd;
    idx_list p_group_offsets;
    for(size_t i = 0; i < m_group_sd.size(); ++i)
    {
        idx_list o_subs;
        idx_list n_subs;
        idx_list rel_n_subs;
        size_t off = m_group_offsets[i];
        size_t order = m_group_sd[i].get_order();
        for(size_t j = off; j < off+order; ++j) o_subs.push_back(j);
        for(size_t j = 0; j < order; ++j) n_subs.push_back(inv_p[o_subs[j]]);

        idx_list abs_to_rel(n_subs);
        sort(abs_to_rel.begin(),abs_to_rel.end());
        for(size_t j = 0; j < order; ++j)
        {
            idx_list::iterator pos = find(abs_to_rel.begin(),abs_to_rel.end(),n_subs[j]);
            size_t rel = distance(abs_to_rel.begin(),pos);
            rel_n_subs.push_back(rel);
        }

        idx_list rel_p_ent(order);
        for(size_t j = 0; j < order; ++j) rel_p_ent[rel_n_subs[j]] = j;

        p_group_sd.push_back(m_group_sd[i].permute(runtime_permutation(rel_p_ent)));

        for(size_t j = 0; j < order; ++j)
        {
            size_t start = abs_to_rel[j]+1;
            size_t end = (j == order - 1) ? abs_to_rel[j]+1 : abs_to_rel[j+1];
            for(size_t k = start; k < end; ++k)
            {
                idx_list ents = range(0,p_subspaces[k].get_n_blocks());
                p_group_sd.back() = p_group_sd.back().insert_entries(rel_n_subs[j+k-start],ents);
            }
        }
        p_group_offsets.push_back(abs_to_rel[0]);
    }

    return sparse_bispace_impl(p_subspaces,p_group_sd,p_group_offsets);
}

} // namespace libtensor
