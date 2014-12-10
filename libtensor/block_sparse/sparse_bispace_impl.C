#include "sparse_bispace_impl.h"
#include "range.h"

using namespace std;

namespace libtensor {
    
const char* sparse_bispace_impl::k_clazz = "sparse_bispace_impl";

void sparse_bispace_impl::init_ig()
{
    size_t subspace_idx = 0; 
    size_t sd_idx = 0;
    while(subspace_idx < m_subspaces.size())
    {
        m_ig_offsets.push_back(subspace_idx);

        //Anything sparse in this bispace?
        bool treat_as_sparse = false;
        if(sd_idx < m_group_offsets.size())
        {
            //Are we in a sparse group?
            if(subspace_idx == m_group_offsets[sd_idx])
            {
                treat_as_sparse = true;
            }
        }

        if(treat_as_sparse)
        {
            //We are in a sparse group, use the total group size
            size_t grp_nnz = 0;
            const sparsity_data& sd = m_group_sd[sd_idx];
            for(sparsity_data::const_iterator it = sd.begin(); it != sd.end(); ++it)
            {
                size_t sd_off = m_group_offsets[sd_idx];
                size_t ent_nnz = 1;
                for(size_t i = sd_off; i < sd_off + sd.get_order(); ++i)
                    ent_nnz *= m_subspaces[i].get_block_size(it->first[i-sd_off]);
                grp_nnz += ent_nnz;
            }
            m_ig_dims.push_back(grp_nnz);
            subspace_idx += m_group_sd[sd_idx].get_order();
            ++sd_idx;
        }
        else
        {
            m_ig_dims.push_back(m_subspaces[subspace_idx].get_dim());
            ++subspace_idx;
        }
    }
}

sparse_bispace_impl::sparse_bispace_impl(const vector<subspace>& subspaces,
                                         const vector<sparsity_data>& group_sd,
                                         const idx_list& group_offsets) : m_subspaces(subspaces),
                                                                          m_group_sd(group_sd),
                                                                          m_group_offsets(group_offsets)
{
    init_ig();
}

sparse_bispace_impl::sparse_bispace_impl(const sparse_bispace_impl& lhs,
                                         const sparse_bispace_impl& rhs)
{
    m_subspaces.insert(m_subspaces.end(),lhs.m_subspaces.begin(),lhs.m_subspaces.end());
    m_subspaces.insert(m_subspaces.end(),rhs.m_subspaces.begin(),rhs.m_subspaces.end());
    m_group_sd.insert(m_group_sd.end(),lhs.m_group_sd.begin(),lhs.m_group_sd.end());
    m_group_sd.insert(m_group_sd.end(),rhs.m_group_sd.begin(),rhs.m_group_sd.end());
    m_group_offsets.insert(m_group_offsets.end(),lhs.m_group_offsets.begin(),lhs.m_group_offsets.end());
    for(size_t i = 0; i < rhs.m_group_offsets.size(); ++i)
        m_group_offsets.push_back(rhs.m_group_offsets[i] + lhs.m_subspaces.size()); 

    init_ig();
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

sparse_bispace_impl sparse_bispace_impl::contract(size_t contract_idx) const
{
    if(m_subspaces.size() == 1)
        throw bad_parameter(g_ns,k_clazz,"contract(...)",__FILE__,__LINE__,"cannot contract 1d bispace"); 
    if(contract_idx >= m_subspaces.size())
        throw bad_parameter(g_ns,k_clazz,"contract(...)",__FILE__,__LINE__,"contract_idx out of bounds"); 

    vector<subspace> subspaces(m_subspaces);
    subspaces.erase(subspaces.begin()+contract_idx);

    vector<sparsity_data> group_sd;
    idx_list group_offsets;
    for(size_t sd_idx = 0; sd_idx < m_group_sd.size(); ++sd_idx)
    {
        size_t order = m_group_sd[sd_idx].get_order();
        size_t off = m_group_offsets[sd_idx];
        size_t new_off = off > contract_idx ? off-1 : off;
        if((off <= contract_idx) && (contract_idx < off+order))
        {
            if(order == 2) continue; //sparsity destroyed totally
            else
            {
                size_t rel_idx = contract_idx - off;
                group_sd.push_back(m_group_sd[sd_idx].contract(rel_idx));
            }
        }
        else 
            group_sd.push_back(m_group_sd[sd_idx]);
        group_offsets.push_back(new_off);
    }
    return sparse_bispace_impl(subspaces,group_sd,group_offsets);
}

size_t sparse_bispace_impl::get_ig_order(size_t grp_idx) const
{
    if(grp_idx == m_ig_offsets.size() - 1)
        return m_subspaces.size() - m_ig_offsets.back();
    else
        return m_ig_offsets[grp_idx+1] - m_ig_offsets[grp_idx];
}

size_t sparse_bispace_impl::get_ig_containing_subspace(size_t subspace_idx) const
{
    if(subspace_idx >= m_subspaces.size())
    {
        throw bad_parameter(g_ns,k_clazz,"get_ig_containing_subspace()",__FILE__,__LINE__,"subspace idx too large"); 
    } 

    size_t ig;
    for(size_t i = 0; i < m_ig_offsets.size(); ++i)
    {
        if(m_ig_offsets[i] <= subspace_idx)
        {
            if(i == m_ig_offsets.size() - 1)
            {
                if(subspace_idx < m_subspaces.size())
                {
                    ig = i;
                    break;
                }
            }
            else if(m_ig_offsets[i+1] > subspace_idx)
            {
                ig = i;
                break;
            }
        }
    }
    return ig;
}

} // namespace libtensor
