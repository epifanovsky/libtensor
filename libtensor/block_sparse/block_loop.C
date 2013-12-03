/*
 * block_loop_new.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#include "block_loop.h"

namespace libtensor
{

size_t flops = 0;
bool count_flops = false;

const char* block_loop::k_clazz = "block_loop";

block_loop::block_loop(const std::vector< sparse_bispace_any_order >& bispaces) : m_bispaces(bispaces)
    
{
	if(bispaces.size() == 0)
	{
		throw bad_parameter(g_ns, k_clazz,"block_loop_new(...)",
				__FILE__, __LINE__, "loop must access at least one bispace");
	}
    size_t n_bispaces = m_bispaces.size();
    m_subspaces_looped.resize(n_bispaces);
    m_bispaces_ignored.resize(n_bispaces);
    for(size_t bispace_idx = 0; bispace_idx  < n_bispaces; ++bispace_idx)
    {
        m_bispaces_ignored[bispace_idx] = true;
    }
}

void block_loop::set_subspace_looped(size_t bispace_idx, size_t subspace_idx)
{
	//Is bispace_idx/subspace_idx valid?
	if(bispace_idx >= m_bispaces.size())
	{
		throw out_of_bounds(g_ns, k_clazz,"set_subspace_looped(...)",
				__FILE__, __LINE__, "bispace_idx is out of bounds");
	}
	if(subspace_idx >= m_bispaces[bispace_idx].get_order())
	{
		throw out_of_bounds(g_ns, k_clazz,"set_subspace_looped(...)",
				__FILE__, __LINE__, "subspace_idx is out of bounds");
	}

	//Verify that all of the subspaces accessed by this loop are identical
    const sparse_bispace_any_order& cur_subspace = m_bispaces[bispace_idx][subspace_idx];
    for(size_t ref_bispace_idx = 0; ref_bispace_idx < m_bispaces.size(); ++ref_bispace_idx)
	{
        if(!m_bispaces_ignored[ref_bispace_idx] && ref_bispace_idx != bispace_idx)
        {
            size_t ref_subspace_idx = m_subspaces_looped[ref_bispace_idx];
            const sparse_bispace_any_order& ref_subspace = m_bispaces[ref_bispace_idx][ref_subspace_idx];
            if(ref_subspace != cur_subspace)
            {
                throw bad_parameter(g_ns, k_clazz,"set_subspace_looped(...)",
                                    __FILE__, __LINE__, "subspaces specified are incompatible");
            }
        }
	}

	//Finally, record that the loop touches the specified subspace of the given bispace
	m_subspaces_looped[bispace_idx] = subspace_idx;
    m_bispaces_ignored[bispace_idx] = false;
}

size_t block_loop::get_subspace_looped(size_t bispace_idx) const
{
#ifdef LIBTENSOR_DEBUG
	if(bispace_idx >= m_bispaces.size())
	{
		throw out_of_bounds(g_ns, k_clazz,"get_subspace_looped(...)",
				__FILE__, __LINE__, "bispace_idx is out of bounds");
	}
	else if(m_bispaces_ignored[bispace_idx] == true)
	{
		throw bad_parameter(g_ns, k_clazz,"get_subspace_looped(...)",
				__FILE__, __LINE__, "bispace_idx is not looped over");
	}
#endif

	return m_subspaces_looped[bispace_idx];
}

bool block_loop::is_bispace_ignored(size_t bispace_idx) const
{
#ifdef LIBTENSOR_DEBUG
	if(bispace_idx >= m_bispaces.size())
	{
		throw out_of_bounds(g_ns, k_clazz,"is_bispace_ignored(...)",
				__FILE__, __LINE__, "bispace_idx is out of bounds");
	}
#endif

	return m_bispaces_ignored[bispace_idx];
}

} /* namespace libtensor */

