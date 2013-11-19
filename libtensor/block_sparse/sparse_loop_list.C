/*
 * sparse_loop_list.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#include "sparse_loop_list.h"

namespace libtensor
{

const char* sparse_loop_list::k_clazz = "sparse_loop_list";

sparse_loop_list::sparse_loop_list(const std::vector< sparse_bispace_any_order >& bispaces) : m_bispaces(bispaces)
{
}

void sparse_loop_list::add_loop(const block_loop& loop)
{
	const std::vector< sparse_bispace_any_order >& cur_bispaces = loop.get_bispaces();
	if(m_loops.size() > 0)
	{
		//Check that bispaces are compatible with existing loops
		if(m_bispaces.size() != cur_bispaces.size())
		{
			throw bad_parameter(g_ns, k_clazz,"add_loop(...)",
					__FILE__, __LINE__, "wrong number of bispaces in loop to be added");
		}

		for(size_t i = 0; i < m_bispaces.size(); ++i)
		{
			if(m_bispaces[i] != cur_bispaces[i])
			{
				throw bad_parameter(g_ns, k_clazz,"add_loop(...)",
						__FILE__, __LINE__, "bispaces of loop do not match those already in loop list");
			}
		}

		//Check that no two loops access the same subspace
		for(size_t cur_loop_idx = 0; cur_loop_idx < m_loops.size(); ++cur_loop_idx)
		{
			const block_loop& cur_loop = m_loops[cur_loop_idx];
			for(size_t bis_idx = 0; bis_idx < cur_bispaces.size(); ++bis_idx)
			{
				if(!cur_loop.is_bispace_ignored(bis_idx) && !loop.is_bispace_ignored(bis_idx))
				{
					if(cur_loop.get_subspace_looped(bis_idx) == loop.get_subspace_looped(bis_idx))
					{
						throw bad_parameter(g_ns, k_clazz,"add_loop(...)",
								__FILE__, __LINE__, "Two loops cannot access the same subspace of the same bispace");
					}
				}
			}
		}
	}

	//Loops that don't touch any subspaces aren't allowed
	bool all_ignored = true;
	for(size_t i = 0 ; i < cur_bispaces.size(); ++i)
	{
		if(!loop.is_bispace_ignored(i))
		{
			all_ignored = false;
			break;
		}
	}

	if(all_ignored)
	{
		throw bad_parameter(g_ns, k_clazz,"add_loop(...)",
				__FILE__, __LINE__, "a loop may not ignore all bispaces");
	}
	m_loops.push_back(loop);
}

std::vector<size_t> sparse_loop_list::get_loops_that_access_bispace(
		size_t bispace_idx) const
{
	std::vector<size_t> loops_that_access_bispace;
	if(bispace_idx >= m_bispaces.size())
	{
		throw out_of_bounds(g_ns, k_clazz,"get_loops_that_access_bispace(...)",
				__FILE__, __LINE__, "bispace index out of bounds");
	}
	for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
	{
		if(!m_loops[loop_idx].is_bispace_ignored(bispace_idx))
		{
			loops_that_access_bispace.push_back(loop_idx);
		}
	}
	return loops_that_access_bispace;
}

std::vector<size_t> sparse_loop_list::get_loops_that_access_group(
		size_t bispace_idx, size_t group_idx) const
{
	std::vector<size_t> loops_that_access_bispace = get_loops_that_access_bispace(bispace_idx);
	const sparse_bispace_any_order& cur_bispace = m_bispaces[bispace_idx];
	if(group_idx >= cur_bispace.get_n_sparse_groups())
	{
		throw out_of_bounds(g_ns, k_clazz,"get_loops_that_access_group(...)",
				__FILE__, __LINE__, "group index out of bounds");
	}

	//Filter loops that only access this group
	std::vector<size_t> loops_that_access_group;
	for(size_t bispace_loop_idx = 0; bispace_loop_idx < loops_that_access_bispace.size(); ++bispace_loop_idx)
	{
		size_t loop_idx = loops_that_access_bispace[bispace_loop_idx];
		size_t cur_subspace_idx = m_loops[loop_idx].get_subspace_looped(bispace_idx);
		size_t group_offset = cur_bispace.get_sparse_group_offset(group_idx);
		size_t group_order = cur_bispace.get_sparse_group_tree(group_idx).get_order();
		if((group_offset <= cur_subspace_idx) && (cur_subspace_idx < group_offset+group_order))
		{
			loops_that_access_group.push_back(loop_idx);
		}
	}
	return loops_that_access_group;
}

} /* namespace libtensor */


