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
/*double contract_seconds = 0;*/

const char* block_loop::k_clazz = "block_loop";

block_loop::block_loop(const std::vector< sparse_bispace_any_order >& bispaces)
    
{
	if(bispaces.size() == 0)
	{
		throw bad_parameter(g_ns, k_clazz,"block_loop_new(...)",
				__FILE__, __LINE__, "loop must access at least one bispace");
	}
    size_t n_bispaces = bispaces.size();
    m_subspaces_looped.resize(n_bispaces);
    m_bispaces_ignored.resize(n_bispaces);
    for(size_t bispace_idx = 0; bispace_idx  < n_bispaces; ++bispace_idx)
    {
        m_bispaces_ignored[bispace_idx] = true;
        m_bispace_orders.push_back(bispaces[bispace_idx].get_order());
    }
}

void block_loop::set_subspace_looped(size_t bispace_idx, size_t subspace_idx)
{
#ifdef LIBTENSOR_DEBUG
	//Is bispace_idx/subspace_idx valid?
	if(bispace_idx >= m_bispaces_ignored.size())
	{
		throw out_of_bounds(g_ns, k_clazz,"set_subspace_looped(...)",
				__FILE__, __LINE__, "bispace_idx is out of bounds");
	}
	if(subspace_idx >= m_bispace_orders[bispace_idx])
	{
		throw out_of_bounds(g_ns, k_clazz,"set_subspace_looped(...)",__FILE__, __LINE__,
                "subspace_idx is out of bounds");
	}
#endif

	//Finally, record that the loop touches the specified subspace of the given bispace
	m_subspaces_looped[bispace_idx] = subspace_idx;
    m_bispaces_ignored[bispace_idx] = false;
}

size_t block_loop::get_subspace_looped(size_t bispace_idx) const
{
#ifdef LIBTENSOR_DEBUG
	if(bispace_idx >= m_bispaces_ignored.size())
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
	if(bispace_idx >= m_bispaces_ignored.size())
	{
		throw out_of_bounds(g_ns, k_clazz,"is_bispace_ignored(...)",
				__FILE__, __LINE__, "bispace_idx is out of bounds");
	}
#endif

	return m_bispaces_ignored[bispace_idx];
}

} /* namespace libtensor */

