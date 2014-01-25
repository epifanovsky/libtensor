/*
 * sparse_loop_list.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#include "sparse_loop_list.h"
#include "sparsity_fuser.h"
#include "sparse_loop_grouper.h"

using namespace std;

namespace libtensor
{

const char* sparse_loop_list::k_clazz = "sparse_loop_list";

sparse_loop_list::sparse_loop_list(const vector<block_loop>& loops,const idx_list& direct_tensors) : m_loops(loops)
{
    if(m_loops.size() == 0)
    {
        throw bad_parameter(g_ns, k_clazz,"sparse_loop_list(...)",__FILE__, __LINE__,
            "Cannot have an empty loop list");
    }
    m_bispaces = m_loops[0].get_bispaces();

    //Check that all loops have compatible bispaces
    for(size_t loop_idx = 1; loop_idx < m_loops.size(); ++loop_idx)
    {
        if(m_loops[loop_idx].get_bispaces() != m_bispaces)
        {
            throw bad_parameter(g_ns, k_clazz,"sparse_loop_list(...)",
                    __FILE__, __LINE__, "bispaces of loops do not match");
        }
    }

    for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
    {
        const block_loop& loop = m_loops[loop_idx];

        //Check that no loop is an empty loop - one that touches no bispaces/subspaces
        bool all_ignored = true;
        for(size_t bispace_idx = 0 ; bispace_idx < m_bispaces.size(); ++bispace_idx)
        {
            if(!loop.is_bispace_ignored(bispace_idx))
            {
                all_ignored = false;
                break;
            }
        }
        if(all_ignored)
        {
            throw bad_parameter(g_ns, k_clazz,"sparse_loop_list(...)",__FILE__, __LINE__,
                "a loop may not ignore all bispaces");
        }

		//Check that no two loops access the same subspace
		for(size_t other_loop_idx = loop_idx + 1; other_loop_idx < m_loops.size(); ++other_loop_idx)
		{
			const block_loop& other_loop = m_loops[other_loop_idx];
            for(size_t bispace_idx = 0 ; bispace_idx < m_bispaces.size(); ++bispace_idx)
			{
				if(!other_loop.is_bispace_ignored(bispace_idx) && !loop.is_bispace_ignored(bispace_idx))
				{
					if(other_loop.get_subspace_looped(bispace_idx) == loop.get_subspace_looped(bispace_idx))
					{
						throw bad_parameter(g_ns, k_clazz,"sparse_loop_list(...)",__FILE__, __LINE__,
                            "Two loops cannot access the same subspace of the same bispace");
					}
				}
			}
		}
    }

    //Fuse all coupled sparse trees
    sparsity_fuser sf(m_loops,m_bispaces);
    for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
    {
        idx_list tree_indices = sf.get_trees_for_loop(loop_idx);

        //Fuse in reverse order so that all tree indices remain valid throughout process
        for(size_t tree_rel_idx = 1; tree_rel_idx < tree_indices.size(); ++tree_rel_idx)
        {
            //Find common loops between both trees
            size_t rhs_tree_idx = tree_indices[tree_indices.size() - tree_rel_idx];
            idx_list lhs_loops = sf.get_loops_for_tree(tree_indices[0]);
            idx_list rhs_loops = sf.get_loops_for_tree(rhs_tree_idx);
            idx_list common_loops;
            for(size_t lhs_loop_rel_idx = 0; lhs_loop_rel_idx < lhs_loops.size(); ++lhs_loop_rel_idx)
            {
                size_t common_loop_idx =  lhs_loops[lhs_loop_rel_idx];
                if(find(rhs_loops.begin(),rhs_loops.end(),common_loop_idx) != rhs_loops.end())
                {
                    common_loops.push_back(common_loop_idx);
                }
            }
            sf.fuse(tree_indices[0],rhs_tree_idx,common_loops);
        }
    }

    //Create block offset and size information for each loop group
    sparse_loop_grouper slg(sf);
    m_bispaces_and_index_groups = slg.get_bispaces_and_index_groups();
    m_bispaces_and_subspaces = slg.get_bispaces_and_subspaces();
    m_block_dims = slg.get_block_dims();
    m_offsets_and_sizes = slg.get_offsets_and_sizes();
    m_loops_for_groups = slg.get_loops_for_groups();
    m_loop_bounds.resize(m_offsets_and_sizes.size());
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


