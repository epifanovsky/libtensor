/*
 * sparse_loop_list.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef SPARSE_LOOP_LIST_H_
#define SPARSE_LOOP_LIST_H_

#include <vector>
#include "block_loop.h"
#include "sparse_bispace.h"
#include "block_kernel_i.h"
#include "loop_list_sparsity_data.h"

namespace libtensor
{

typedef std::map<size_t, std::vector<size_t> > fixed_block_map;
typedef std::map<size_t,size_t> loop_fixed_block_map;

class sparse_loop_list
{
private:
    static const char* k_clazz; //!< Class name
    std::vector<block_loop> m_loops;
    std::vector<sparse_bispace_any_order> m_bispaces;

	template<typename T>
    void _run_internal(block_kernel_i<T>& kernel,
    				   std::vector<T*>& ptrs,
    				   loop_list_sparsity_data& llsd,
    				   std::vector<dim_list>& bispace_dim_lists,
    				   std::vector<block_list>& bispace_block_lists,
    				   block_list& loop_indices,
    				   size_t loop_idx=0);
    std::vector<std::vector<off_dim_pair_list> > m_group_offsets_and_sizes; //Each entry contains offsets and sizes of the blocks for each a given loop group
    std::vector<idx_pair_list> m_bispaces_and_index_groups; //Each entry contains the bispaces and index groups touched by a given loop group
    std::vector<dim_list> m_block_sizes; //The block sizes of each tensor 
    std::vector<offset_list> m_block_offsets; //The block 
public:
	sparse_loop_list(const std::vector<block_loop>& loops);

	template<typename T>
	void run(block_kernel_i<T>& kernel,std::vector<T*>& ptrs);

	const std::vector< sparse_bispace_any_order >& get_bispaces() const { return m_bispaces; }
	const std::vector< block_loop >& get_loops() const { return m_loops; }

	//Returns the indices of the loops that access any subspace of the specified bispace
	std::vector<size_t> get_loops_that_access_bispace(size_t bispace_idx) const;

	//Returns the indices of the loops that access any subspace of the specified sparse group
	//within the specified bispace
	std::vector<size_t> get_loops_that_access_group(size_t bispace_idx,size_t group_idx) const;
};

template<typename T>
void sparse_loop_list::run(block_kernel_i<T>& kernel,std::vector<T*>& ptrs)
{
	if(m_loops.size() == 0)
	{
		throw bad_parameter(g_ns, k_clazz,"run(...)",
				__FILE__, __LINE__, "no loops in loop list");
	}

	//Set up vectors for keeping track of current block indices and dimensions
	const std::vector<sparse_bispace_any_order>& cur_bispaces = m_loops[0].get_bispaces();
	size_t n_bispaces = cur_bispaces.size();
	std::vector<dim_list> bispace_dim_lists(n_bispaces);
	std::vector<block_list> bispace_block_lists(n_bispaces);
	for(size_t bispace_idx = 0; bispace_idx < n_bispaces; ++bispace_idx)
	{
		bispace_dim_lists[bispace_idx].resize(cur_bispaces[bispace_idx].get_order());
		bispace_block_lists[bispace_idx].resize(cur_bispaces[bispace_idx].get_order());
	}

	//Aggregate the sparsity information from all of the loops
	loop_list_sparsity_data llsd(*this);

	//Fix appropriate loop indices and block indices/dimensions based on fixed_blocks data
	block_list loop_indices(m_loops.size());
#if 0
	loop_fixed_block_map lfbm;
	for(fixed_block_map::const_iterator it = fixed_blocks.begin(); it != fixed_blocks.end(); ++it)
	{
		size_t bispace_idx = it->first;
		if(bispace_idx >= m_bispaces.size())
		{
			throw out_of_bounds(g_ns, k_clazz,"run(...)",
					__FILE__, __LINE__, "bispace_idx is out of bounds");
		}

		size_t cur_order = m_bispaces[bispace_idx].get_order();
		const std::vector<size_t>& block_indices = it->second;
		if(block_indices.size() != cur_order)
		{
			throw bad_parameter(g_ns, k_clazz,"run(...)",
					__FILE__, __LINE__, "invalid number of block indices for fixed block");
		}
		//TODO: Add validation of whether block indices are actually traversable given sparsity (will need to search trees)

		std::vector<size_t> cur_dims(block_indices.size());
		const sparse_bispace_any_order& cur_bispace = m_bispaces[bispace_idx];
		for(size_t i = 0; i < cur_dims.size(); ++i)
		{
			cur_dims[i] = cur_bispace[i].get_block_size(block_indices[i]);
		}

		//TODO: check for inconsistent indices in duplicate loops!!!!

		//Determine which loops are fixed as a result of fixing the indices associated with this bispace
		//When we run those loops, we will automatically only traverse the single fixed block value
		std::vector<size_t> fixed_loop_indices = get_loops_that_access_bispace(bispace_idx);
		for(size_t i = 0; i < fixed_loop_indices.size(); ++i)
		{
			size_t fixed_loop_idx = fixed_loop_indices[i];
			const block_loop& cur_loop = m_loops[fixed_loop_idx];
			size_t block_idx = block_indices[cur_loop.get_subspace_looped(bispace_idx)];
			lfbm.insert(std::make_pair(fixed_loop_idx,block_idx));
		}
	}
#endif
	_run_internal(kernel,ptrs,llsd,bispace_dim_lists,bispace_block_lists,loop_indices,0);
}

template<typename T>
void sparse_loop_list::_run_internal(block_kernel_i<T>& kernel,
				   std::vector<T*>& ptrs,
				   loop_list_sparsity_data& llsd,
				   std::vector<dim_list>& bispace_dim_lists,
				   std::vector<block_list>& bispace_block_lists,
				   block_list& loop_indices,
				   size_t loop_idx)
{
	//Get the subspace that we are looping over
    const block_loop& cur_loop = m_loops[loop_idx];
    size_t first_bispace_idx, first_subspace_idx;
    for(size_t i = 0; i < m_bispaces.size(); ++i)
    {
    	if(!cur_loop.is_bispace_ignored(i))
    	{
    		first_bispace_idx = i;
    		first_subspace_idx = cur_loop.get_subspace_looped(first_bispace_idx);
    		break;
    	}
    }
    const sparse_bispace<1>& cur_subspace = m_bispaces[first_bispace_idx][first_subspace_idx];

    block_list block_indices;
    //Could the range of this loop be restricted by fixing the blocks of a particular bispace?
    //loop_fixed_block_map::const_iterator lfbm_it = lfbm.find(loop_idx);
    //if(lfbm_it != lfbm.end())
    //{
        //block_indices.push_back(lfbm_it->second);
    //}
    //else
    //{
		//Get the list of blocks in that subspace that are significant
    	block_indices = llsd.get_sig_block_list(loop_indices,loop_idx);
    //}

    for(size_t cur_block_idx = 0; cur_block_idx < block_indices.size(); ++cur_block_idx)
    {
        size_t cur_block = block_indices[cur_block_idx];
        size_t block_size = cur_subspace.get_block_size(cur_block);

        loop_indices[loop_idx] = cur_block;

        for(size_t bispace_idx = 0; bispace_idx < m_bispaces.size(); ++bispace_idx)
        {
            if(!cur_loop.is_bispace_ignored(bispace_idx))
            {
				size_t cur_subspace_idx = cur_loop.get_subspace_looped(bispace_idx);
				bispace_dim_lists[bispace_idx][cur_subspace_idx] = block_size;
				bispace_block_lists[bispace_idx][cur_subspace_idx] = cur_block;
            }
        }

        //Base case - use kernel to process the block
        if(loop_idx == (m_loops.size() - 1))
        {
            //Locate the appropriate blocks
        	std::vector<T*> bispace_block_ptrs(ptrs);
        	for(size_t bispace_idx = 0; bispace_idx < m_bispaces.size(); ++bispace_idx)
        	{
        		//If blocks for a bispace are fixed, we assume that the pointer points directly to the desired block
                //if(fbm.find(bispace_idx) == fbm.end())
                //{
					bispace_block_ptrs[bispace_idx] += m_bispaces[bispace_idx].get_block_offset(bispace_block_lists[bispace_idx]);
                //}
        	}
            kernel(bispace_block_ptrs,bispace_dim_lists);
        }
        else
        {
            _run_internal(kernel,ptrs,llsd,bispace_dim_lists,bispace_block_lists,loop_indices,loop_idx+1);
        }
    }
}

} /* namespace libtensor */

#endif /* SPARSE_LOOP_LIST_H_ */
