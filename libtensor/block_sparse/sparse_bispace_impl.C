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
        p_subspaces[i] = m_subspaces[perm[i]];

    idx_list inv_p_ent(m_subspaces.size());
    for(size_t i = 0; i < m_subspaces.size(); ++i) inv_p_ent[perm[i]] = i;
    runtime_permutation inv_p(inv_p_ent);

    //Sort the sds by their lowest subspace position in the final bispace 
    size_t n_sd = m_group_sd.size();
    vector<idx_list> o_subs(n_sd); //old sd subspaces
    vector<idx_list> n_subs(n_sd); //new sd subspaces
    vector<idx_list> a_to_r(n_sd); //absolute to relative
    for(size_t i = 0; i < n_sd; ++i)
    {
        size_t off = m_group_offsets[i];
        size_t order = m_group_sd[i].get_order();
        for(size_t j = off; j < off+order; ++j) o_subs[i].push_back(j);
        for(size_t j = 0; j < order; ++j) n_subs[i].push_back(inv_p[o_subs[i][j]]);
        for(size_t j = 0; j < order; ++j) a_to_r[i].push_back(n_subs[i][j]);
        
        sort(a_to_r[i].begin(),a_to_r[i].end());
    }

    //Permute all of our arrays to match sd order in result
    vector<sparsity_data> p_group_sd(m_group_sd);
    idx_pair_list min_subs;
    for(size_t i = 0; i < n_sd; ++i)
        min_subs.push_back(make_pair(a_to_r[i][0],i));
    sort(min_subs.begin(),min_subs.end());
    idx_list p_ent;
    for(size_t i = 0; i < n_sd; ++i) p_ent.push_back(min_subs[i].second);
    runtime_permutation sd_p(p_ent); 
    sd_p.apply(p_group_sd);
    sd_p.apply(o_subs);
    sd_p.apply(n_subs);
    sd_p.apply(a_to_r);

    for(size_t i = 0; i < n_sd; ++i)
        for(size_t j = i+1; j < n_sd; ++j)
            for(size_t k = 0; k < a_to_r[j].size(); ++k)
                if((a_to_r[i][0] < a_to_r[j][k]) && (a_to_r[j][k] < a_to_r[i].back()))
                    throw bad_parameter(g_ns,k_clazz,"permute(...)",__FILE__,__LINE__,"Permutation interleaves trees"); 

    //Now with our repositioned sds in proper order, can proceed
    idx_list p_group_offsets;
    vector<idx_list> r_n_subs(n_sd); //sd-relative new subspaces
    for(size_t i = 0; i < n_sd; ++i)
    {
        size_t order = p_group_sd[i].get_order();
        for(size_t j = 0; j < order; ++j)
        {
            idx_list::iterator it;
            it = find(a_to_r[i].begin(),a_to_r[i].end(),n_subs[i][j]);
            size_t rel = distance(a_to_r[i].begin(),it);
            r_n_subs[i].push_back(rel);
        }

        idx_list rel_p_ent(order);
        for(size_t j = 0; j < order; ++j) rel_p_ent[r_n_subs[i][j]] = j;

        p_group_sd[i] = p_group_sd[i].permute(runtime_permutation(rel_p_ent));

        for(size_t j = 0; j < order; ++j)
        {
            size_t start = a_to_r[i][j]+1;
            size_t end = (j == order - 1) ? a_to_r[i][j]+1 : a_to_r[i][j+1];
            for(size_t k = start; k < end; ++k)
            {
                idx_list ents = range(0,p_subspaces[k].get_n_blocks());
                size_t sd_sub = r_n_subs[i][j+k-start];
                p_group_sd[i] = p_group_sd[i].insert_entries(sd_sub,ents);
            }
        }
        p_group_offsets.push_back(a_to_r[i][0]);
    }

    return sparse_bispace_impl(p_subspaces,p_group_sd,p_group_offsets);
}

} // namespace libtensor
