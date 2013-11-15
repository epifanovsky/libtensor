/*
 * sparse_loop_list.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef SPARSE_LOOP_LIST_H_
#define SPARSE_LOOP_LIST_H_

#include <vector>
#include "block_loop_new.h"
#include "sparse_bispace.h"
#include "block_kernels.h"

namespace libtensor
{

class sparse_loop_list
{
private:
    static const char* k_clazz; //!< Class name
    std::vector< block_loop_new > m_loops;
    std::vector< sparse_bispace_any_order > m_bispaces;

	template<typename T>
    void _run_internal(block_kernel_i_new<T>& kernel,
    				   std::vector<T*>& ptrs,
    				   std::vector<dim_list>& bispace_dim_lists,
    				   std::vector<block_list>& bispace_block_lists,
    				   block_list& loop_indices,
    				   size_t loop_idx=0);
public:
	sparse_loop_list(const std::vector< sparse_bispace_any_order >& bispaces);

	void add_loop(const block_loop_new& loop);

	template<typename T>
	void run(block_kernel_i_new<T>& kernel,std::vector<T*>& ptrs);

	const std::vector< sparse_bispace_any_order >& get_bispaces() const { return m_bispaces; }
	const std::vector< block_loop_new >& get_loops() const { return m_loops; }

	//Returns the indices of the loops that access any subspace of the specified bispace
	std::vector<size_t> get_loops_that_access_bispace(size_t bispace_idx) const;
};

template<typename T>
void sparse_loop_list::run(block_kernel_i_new<T>& kernel,std::vector<T*>& ptrs)
{
	if(m_loops.size() == 0)
	{
		throw bad_parameter(g_ns, k_clazz,"run(...)",
				__FILE__, __LINE__, "no loops in loop list");
	}

	const std::vector<sparse_bispace_any_order>& cur_bispaces = m_loops[0].get_bispaces();
	size_t n_bispaces = cur_bispaces.size();
	std::vector<dim_list> bispace_dim_lists(n_bispaces);
	std::vector<block_list> bispace_block_lists(n_bispaces);
	for(size_t bispace_idx = 0; bispace_idx < n_bispaces; ++bispace_idx)
	{
		bispace_dim_lists[bispace_idx].resize(cur_bispaces[bispace_idx].get_order());
		bispace_block_lists[bispace_idx].resize(cur_bispaces[bispace_idx].get_order());
	}
	block_list loop_indices(m_loops.size());
	_run_internal(kernel,ptrs,bispace_dim_lists,bispace_block_lists,loop_indices,0);
}

template<typename T>
void sparse_loop_list::_run_internal(block_kernel_i_new<T>& kernel,
				   std::vector<T*>& ptrs,
				   std::vector<dim_list>& bispace_dim_lists,
				   std::vector<block_list>& bispace_block_lists,
				   block_list& loop_indices,
				   size_t loop_idx)
{
    const block_loop_new& cur_loop = m_loops[loop_idx];
    const std::vector<sparse_bispace_any_order>& cur_bispaces = cur_loop.get_bispaces();
    size_t first_bispace_idx, first_subspace_idx;
    for(size_t i = 0; i < cur_bispaces.size(); ++i)
    {
    	if(!cur_loop.is_bispace_ignored(i))
    	{
    		first_bispace_idx = i;
    		first_subspace_idx = cur_loop.get_subspace_looped(first_bispace_idx);
    		break;
    	}
    }
    const sparse_bispace<1>& cur_subspace = cur_bispaces[first_bispace_idx][first_subspace_idx];

    //TODO: Sparsity happens here - set up as such using llsd
    block_list block_indices(cur_subspace.get_n_blocks());
    for(size_t i = 0; i < block_indices.size(); ++i)
    {
    	block_indices[i] = i;
    }

    for(size_t cur_block_idx = 0; cur_block_idx < block_indices.size(); ++cur_block_idx)
    {
        size_t cur_block = block_indices[cur_block_idx];
        size_t block_size = cur_subspace.get_block_size(cur_block);

        loop_indices[loop_idx] = cur_block;

        for(size_t bispace_idx = 0; bispace_idx < cur_bispaces.size(); ++bispace_idx)
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
        	for(size_t bispace_idx = 0; bispace_idx < cur_bispaces.size(); ++bispace_idx)
        	{
        		bispace_block_ptrs[bispace_idx] += cur_bispaces[bispace_idx].get_block_offset(bispace_block_lists[bispace_idx]);
        	}
            kernel(bispace_block_ptrs,bispace_dim_lists);
        }
        else
        {
            _run_internal(kernel,ptrs,bispace_dim_lists,bispace_block_lists,loop_indices,loop_idx+1);
        }
    }
}

} /* namespace libtensor */

#endif /* SPARSE_LOOP_LIST_H_ */
