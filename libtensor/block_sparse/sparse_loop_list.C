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

sparse_loop_list::sparse_loop_list()
{
	// TODO Auto-generated constructor stub

}

void sparse_loop_list::add_loop(const block_loop_new& loop)
{
	const std::vector< sparse_bispace_any_order >& cur_bispaces = loop.get_bispaces();
	if(m_loops.size() > 0)
	{
		//Check that bispaces are compatible with existing loops
		const std::vector< sparse_bispace_any_order >& ref_bispaces = m_loops[0].get_bispaces();
		if(ref_bispaces.size() != cur_bispaces.size())
		{
			throw bad_parameter(g_ns, k_clazz,"add_loop(...)",
					__FILE__, __LINE__, "wrong number of bispaces in loop to be added");
		}

		for(size_t i = 0; i < ref_bispaces.size(); ++i)
		{
			if(ref_bispaces[i] != cur_bispaces[i])
			{
				throw bad_parameter(g_ns, k_clazz,"add_loop(...)",
						__FILE__, __LINE__, "bispaces of loop do not match those already in loop list");
			}
		}

		//Check that no two loops access the same subspace
		for(size_t cur_loop_idx = 0; cur_loop_idx < m_loops.size(); ++cur_loop_idx)
		{
			const block_loop_new& cur_loop = m_loops[cur_loop_idx];
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

} /* namespace libtensor */
