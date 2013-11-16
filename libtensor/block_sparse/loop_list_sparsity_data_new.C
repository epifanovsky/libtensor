/*
 * loop_list_sparsity_data_new.cpp
 *
 *  Created on: Nov 14, 2013
 *      Author: smanzer
 */
#include "loop_list_sparsity_data_new.h"
#include <algorithm>

namespace libtensor
{

loop_list_sparsity_data_new::loop_list_sparsity_data_new(
		const sparse_loop_list& loop_list)
{
	//First, we determine the full list of blocks of each subspace that is looped over by any loop
	const std::vector< block_loop_new >& loops = loop_list.get_loops();
	for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
	{
		const block_loop_new& cur_loop = loops[loop_idx];
		const std::vector< sparse_bispace_any_order >& cur_bispaces = cur_loop.get_bispaces();
		for(size_t bispace_idx = 0; bispace_idx < cur_bispaces.size(); ++bispace_idx)
		{
			if(!cur_loop.is_bispace_ignored(bispace_idx))
			{
				const sparse_bispace_any_order& cur_bispace = cur_bispaces[bispace_idx];
				size_t subspace_idx = cur_loop.get_subspace_looped(bispace_idx);
				m_subspace_block_lists.push_back(range(0,cur_bispace[subspace_idx].get_n_blocks()));
				break;
			}
		}
	}

	//Now, we identify which loops access each sparse index group within each bispace
	const std::vector< sparse_bispace_any_order >& bispaces = loop_list.get_bispaces();
	for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
	{
		const sparse_bispace_any_order& cur_bispace = bispaces[bispace_idx];
		//Get indices of loops that access this bispace
		const std::vector<size_t> bispace_loop_indices = loop_list.get_loops_that_access_bispace(bispace_idx);

		for(size_t group_idx = 0; group_idx < cur_bispace.get_n_sparse_groups(); ++group_idx)
		{
			sparse_block_tree_any_order cur_tree = cur_bispace.get_sparse_group_tree(group_idx);
			size_t group_offset = cur_bispace.get_sparse_group_offset(group_idx);
			size_t lhs_ini_order = cur_tree.get_order();

			//Filter loops that only access this group
			std::vector<size_t> group_loop_indices;
			std::vector<size_t> group_loop_subspaces;
			for(size_t bispace_loop_idx = 0; bispace_loop_idx < bispace_loop_indices.size(); ++bispace_loop_idx)
			{
				size_t loop_idx = bispace_loop_indices[bispace_loop_idx];
				size_t cur_subspace_idx = loops[loop_idx].get_subspace_looped(bispace_idx);
				if((group_offset <= cur_subspace_idx) && (cur_subspace_idx < group_offset+lhs_ini_order))
				{
					group_loop_indices.push_back(loop_idx);
					group_loop_subspaces.push_back(cur_subspace_idx - group_offset);
				}
			}

			// Determine all trees that are coupled to this one by common loops
			std::vector<size_t> trees_to_fuse;
			for(size_t group_loop_idx = 0; group_loop_idx < group_loop_indices.size(); ++group_loop_idx)
			{
				size_t loop_idx = group_loop_indices[group_loop_idx];
				std::map<size_t, std::pair<size_t,size_t> >::const_iterator  ltt_it = m_loops_to_tree_subspaces.find(loop_idx);
				if(ltt_it != m_loops_to_tree_subspaces.end())
				{
					size_t tree_idx = ltt_it->second.first;
					//Two loops that touch this tree may also both touch another tree - don't add it twice
					if(std::find(trees_to_fuse.begin(),trees_to_fuse.end(),tree_idx) == trees_to_fuse.end())
					{
						trees_to_fuse.push_back(tree_idx);
					}
				}
			}

			//Fuse all trees that couple to this one into one tree
			for(size_t tree_to_fuse_idx = 0; tree_to_fuse_idx < trees_to_fuse.size(); ++tree_to_fuse_idx)
			{
				size_t lhs_order = cur_tree.get_order();
				size_t rhs_tree_idx = trees_to_fuse[tree_to_fuse_idx];
				const sparse_block_tree_any_order& rhs_tree = m_trees[rhs_tree_idx];

				//Determine common indices between lhs and rhs trees
				std::vector<size_t> lhs_fuse_inds;
				std::vector<size_t> rhs_fuse_inds;
				for(size_t lhs_subspace_idx = 0; lhs_subspace_idx < cur_tree.get_order(); ++lhs_subspace_idx)
				{
					size_t loop_idx = group_loop_indices[lhs_subspace_idx];
					std::map<size_t, std::pair<size_t,size_t> >::const_iterator  ltt_it = m_loops_to_tree_subspaces.find(loop_idx);
					if(ltt_it != m_loops_to_tree_subspaces.end() && ltt_it->second.first == rhs_tree_idx)
					{
						lhs_fuse_inds.push_back(group_loop_subspaces[lhs_subspace_idx]);
						rhs_fuse_inds.push_back(ltt_it->second.second);
					}
				}

				//Finally, actually fuse the trees
				cur_tree = cur_tree.fuse(rhs_tree,lhs_fuse_inds,rhs_fuse_inds);

				//All loop indices that touch the RHS tree and are not fused become part of the current tree
				for(std::map<size_t, std::pair<size_t,size_t> >::iterator it = m_loops_to_tree_subspaces.begin(); it != m_loops_to_tree_subspaces.end(); ++it)
				{
					size_t loop_idx = distance(m_loops_to_tree_subspaces.begin(),it);
					size_t subspace_idx = it->second.second;
					bool not_fused = (std::find(rhs_fuse_inds.begin(),rhs_fuse_inds.end(),subspace_idx) == rhs_fuse_inds.end());
					if(it->second.first == rhs_tree_idx && not_fused)
					{
						group_loop_indices.push_back(loop_idx);
						//Shift subspace idx to account for new position in the tree
						size_t new_subspace_idx = subspace_idx + lhs_order;
						if(new_subspace_idx > rhs_fuse_inds.back() + lhs_order)
						{
							new_subspace_idx -= rhs_fuse_inds.size();
						}
						group_loop_subspaces.push_back(new_subspace_idx);
					}
				}
			}

			//Delete all of the trees that were fused from the list, in reverse order for efficiency
			for(size_t tree_to_fuse_idx = 0; tree_to_fuse_idx < trees_to_fuse.size(); ++tree_to_fuse_idx)
			{
				size_t tree_idx = trees_to_fuse[trees_to_fuse.size() - 1 - tree_to_fuse_idx];
				m_trees.erase(m_trees.begin() + tree_idx);
			}

			//Permute the tree and associated index arrays to match loop ordering
			std::vector< std::pair<size_t,size_t> > perm_kv;
			for(size_t group_loop_idx = 0; group_loop_idx < group_loop_indices.size(); ++group_loop_idx)
			{
				perm_kv.push_back(std::make_pair(group_loop_indices[group_loop_idx],group_loop_idx));
			}
			sort(perm_kv.begin(),perm_kv.end());

			std::vector<size_t> perm_entries(perm_kv.size());
			for(size_t perm_idx = 0; perm_idx < perm_entries.size(); ++perm_idx)
			{
				perm_entries[perm_idx] = perm_kv[perm_idx].second;
			}
			runtime_permutation perm(perm_entries);
			perm.apply(group_loop_indices);
			perm.apply(group_loop_subspaces);

			//Now we need to permuted the tree such that its subspaces are in the order that
			//they are accessed by the loops
			runtime_permutation tree_perm(group_loop_subspaces);
			m_trees.push_back(cur_tree.permute(tree_perm));

			//Now that the subspaces of our newly formed tree are loop-ordered, we can sort them as such
			sort(group_loop_subspaces.begin(),group_loop_subspaces.end());

			//Record that each loop touches the appropriate subspace of this tree
			for(size_t group_loop_idx = 0; group_loop_idx < group_loop_indices.size(); ++group_loop_idx)
			{
				size_t tree_idx = m_trees.size()-1;
				size_t tree_subspace_idx = group_loop_subspaces[group_loop_idx];
				size_t loop_idx = group_loop_indices[group_loop_idx];
				m_loops_to_tree_subspaces[loop_idx] = std::pair<size_t,size_t>(tree_idx,tree_subspace_idx);
			}

			//TODO: DEBUG REMOVE
//			std::cout << "\n-----------------------------\n";
//			std::cout << "m_loops_to_tree_subspaces:\n";
//			for(std::map<size_t, std::pair<size_t,size_t> >::iterator it = m_loops_to_tree_subspaces.begin(); it != m_loops_to_tree_subspaces.end(); ++it)
//			{
//				std::cout << it->first << ": " << "(" << it->second.first << "," << it->second.second << ")\n";
//			}
		}
	}
}

block_list loop_list_sparsity_data_new::get_sig_block_list(
	const block_list& sub_key,size_t loop_idx) const
{
	std::map<size_t, std::pair<size_t,size_t> >::const_iterator  ltt_it = m_loops_to_tree_subspaces.find(loop_idx);
	if(ltt_it == m_loops_to_tree_subspaces.end())
	{
		return m_subspace_block_lists[loop_idx];
	}
	else
	{
		size_t tree_idx = ltt_it->second.first;
		return m_trees[tree_idx].get_sub_key_block_list(sub_key);
	}
}

} /* namespace libtensor */

