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
	if(m_subspace_map.size() > 0)
	{
		std::map<size_t,size_t>::iterator sp_it = m_subspace_map.begin();
		const sparse_bispace_any_order& ref_bispace = m_bispaces[sp_it->first][sp_it->second];
		const sparse_bispace_any_order& cur_bispace = m_bispaces[bispace_idx][subspace_idx];
		if(ref_bispace != cur_bispace)
		{
			throw bad_parameter(g_ns, k_clazz,"set_subspace_looped(...)",
					__FILE__, __LINE__, "subspaces specified are incompatible");
		}
	}

	//Finally, record that the loop touches the specified subspace of the given bispace
	m_subspace_map[bispace_idx] = subspace_idx;
}

size_t block_loop::get_subspace_looped(size_t bispace_idx) const
{
#ifdef LIBTENSOR_DEBUG
	if(bispace_idx >= m_bispaces.size())
	{
		throw out_of_bounds(g_ns, k_clazz,"get_subspace_looped(...)",
				__FILE__, __LINE__, "bispace_idx is out of bounds");
	}
	else if(m_subspace_map.find(bispace_idx) == m_subspace_map.end())
	{
		throw bad_parameter(g_ns, k_clazz,"get_subspace_looped(...)",
				__FILE__, __LINE__, "bispace_idx is not looped over");
	}
#endif

	return m_subspace_map.at(bispace_idx);
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

	return m_subspace_map.find(bispace_idx) == m_subspace_map.end();
}

} /* namespace libtensor */

