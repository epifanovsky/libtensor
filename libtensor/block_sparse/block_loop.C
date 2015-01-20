#include "block_loop.h"
#include "range.h"

using namespace std;

namespace libtensor
{

const char* block_loop::k_clazz = "block_loop";

block_loop::block_loop(const subspace& subspace,
                       const idx_pair_list& t_igs,
                       const idx_pair_list& t_s) : 
                                                   m_t_igs(t_igs),
                                                   m_t_s(t_s),
                                                   m_block_inds(range(0,subspace.get_n_blocks())),
                                                   m_start_idx(0),
                                                   m_cur_idx(0),
                                                   m_done(false),
                                                   m_sparse(false),
                                                   m_controlling_loop(NULL),
                                                   m_dependent_loop(NULL)
{
    for(size_t i = 0; i < m_block_inds.size(); ++i)
    {
        m_block_dims.push_back(subspace.get_block_size(i));
        m_block_offs.push_back(idx_list());
        for(size_t t = 0; t < m_t_igs.size(); ++t)
        {
            m_block_offs.back().push_back(subspace.get_block_abs_index(i));
        }
    }
}

//Sparse constructor
block_loop::block_loop(const subspace& subspace,
                       const idx_pair_list& t_igs,
                       const idx_pair_list& t_s,
                       const sparsity_data& sd,
                       size_t sd_sub,
                       const idx_pair_list& sd_off_map) :
                                                            m_t_igs(t_igs),
                                                            m_t_s(t_s),
                                                            m_start_idx(0),
                                                            m_cur_idx(0),
                                                            m_done(false),
                                                            m_sparse(true),
                                                            m_controlling_loop(NULL),
                                                            m_dependent_loop(NULL)

{
    for(sparsity_data::const_iterator it = sd.begin(); it != sd.end(); ++it)
    {
        size_t block_idx = it->first[sd_sub];
        m_block_inds.push_back(block_idx);
        m_block_offs.push_back(idx_list());
        for(size_t t_ig_idx = 0; t_ig_idx < t_igs.size(); ++t_ig_idx)
        {
            bool ig_is_sparse = false;
            for(size_t s_ig_idx = 0; s_ig_idx < sd_off_map.size(); ++s_ig_idx)
            {
                size_t src = sd_off_map[s_ig_idx].first;
                size_t dest = sd_off_map[s_ig_idx].second;
                if(dest == t_ig_idx)
                {
                    m_block_offs.back().push_back(it->second[src]);
                    ig_is_sparse = true;
                    break;
                }
            }
            if(!ig_is_sparse)
                m_block_offs.back().push_back(subspace.get_block_abs_index(block_idx));
        }
        m_block_dims.push_back(subspace.get_block_size(block_idx));
    }
}

void block_loop::apply_offsets(vector<idx_list>& ig_offs) const
{
    for(size_t i = 0; i < m_t_igs.size(); ++i)
    {
        size_t t_idx = m_t_igs[i].first;
        size_t ig_idx = m_t_igs[i].second;
        ig_offs[t_idx][ig_idx] *= m_block_offs[m_cur_idx][i];
        for(size_t f_ig_idx = ig_idx+1; f_ig_idx < ig_offs[t_idx].size(); ++f_ig_idx)
        {
            ig_offs[t_idx][f_ig_idx] *= m_block_dims[m_cur_idx];
        }
    }
}

void block_loop::apply_dims(vector<idx_list>& block_dims) const
{
    for(size_t i = 0; i < m_t_s.size(); ++i)
    {
        size_t t_idx = m_t_s[i].first;
        size_t s_idx = m_t_s[i].second;
        block_dims[t_idx][s_idx] = m_block_dims[m_cur_idx];
    }
}

void block_loop::set_dependent_loop(block_loop& dep_loop)
{
    dep_loop.m_controlling_loop = this;
    this->m_dependent_loop = &dep_loop;
}

block_loop& block_loop::operator++()
{
   size_t old_block_idx = m_block_inds[m_cur_idx];
   if(m_dependent_loop != NULL)
   {
       if(m_dependent_loop->done())
       {
           m_start_idx = m_dependent_loop->m_cur_idx;
           m_cur_idx = m_start_idx;
           m_dependent_loop->m_start_idx = m_start_idx;
           m_dependent_loop->reset();
       }
       else
           throw bad_parameter(g_ns,k_clazz,"operator++",__FILE__,__LINE__,"incrementing outer loop before inner loop finished"); 
   }
   else
   {
       ++m_cur_idx;
   }
   if(m_cur_idx == m_block_inds.size())
       m_done = true;
   else if(m_block_inds[m_cur_idx] <= old_block_idx)
       m_done = true;
   return *this;
}

bool block_loop::done() const
{
    return m_done;
}

void block_loop::reset()
{
    m_cur_idx = m_start_idx;
    if(m_cur_idx != m_block_inds.size()) 
        m_done = false;
}

} /* namespace libtensor */

