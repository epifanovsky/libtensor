/*
 * block_contract2_kernel_new.h
 *
 *  Created on: Nov 16, 2013
 *      Author: smanzer
 */

#ifndef BLOCK_CONTRACT2_KERNEL_H_
#define BLOCK_CONTRACT2_KERNEL_H_

#include "block_kernel_i.h"
#include "sparse_loop_list.h"
#include "../linalg/linalg.h"

//TODO REMOVE
#include <iostream>

namespace libtensor
{

extern size_t flops;
extern bool count_flops;

template<typename T>
class block_contract2_kernel: public libtensor::block_kernel_i<T>
{
private:
    static const char* k_clazz; //!< Class name
    std::vector<size_t> m_block_orders; //!< Orders of each of the blocks

    void _contract_internal(std::vector<T*> ptrs,
    						const std::vector<dim_list>& dim_lists,
    						size_t m,size_t n,size_t k,
    						size_t lda,size_t ldb,size_t ldc,
    						size_t loop_idx = 0);
    const std::vector< block_loop >& m_loops;
    size_t m_n_contracted_inds;
    bool m_A_trans;
    bool m_B_trans;
    std::vector<size_t> m_strides; //Must be pre-allocated for speed
    void (*m_dgemm_fn)(void*,size_t,size_t,size_t,const T*,size_t,const T*,size_t,T*,size_t,T); //DGEMM function to call to process deepest level
public:
    block_contract2_kernel(const sparse_loop_list& loop_list);
	void operator()(const std::vector<T*>& ptrs, const std::vector< dim_list >& dim_lists);
};

template<typename T>
const char* block_contract2_kernel<T>::k_clazz = "block_contract2_kernel<T>";

} /* namespace libtensor */

template<typename T>
inline void libtensor::block_contract2_kernel<T>::_contract_internal(
		std::vector<T*> ptrs, const std::vector<dim_list>& dim_lists, size_t m,
		size_t n, size_t k, size_t lda, size_t ldb, size_t ldc,
		size_t loop_idx)
{
    //Base case: call matmul kernel
    if(loop_idx == m_loops.size() - m_n_contracted_inds - 2)
    {
        (*m_dgemm_fn)(NULL,m,n,k,ptrs[1],lda,ptrs[2],ldb,ptrs[0],ldc,1.0);
        if(count_flops)
        {
        	flops += 2*m*n*k;
        }
    }
    else
    {
    	const block_loop& cur_loop = m_loops[loop_idx];

        //Compute the stride for each block
    	for(size_t bispace_idx = 0; bispace_idx < dim_lists.size(); ++bispace_idx)
    	{
    		const dim_list& cur_dims = dim_lists[bispace_idx];
    		if(!cur_loop.is_bispace_ignored(bispace_idx))
    		{
				size_t cur_subspace = cur_loop.get_subspace_looped(bispace_idx);
                m_strides[bispace_idx] = 1;
				for(size_t stride_idx = cur_subspace+1; stride_idx < cur_dims.size(); ++stride_idx)
				{
					m_strides[bispace_idx] *= cur_dims[stride_idx];
				}
    		}
    	}

    	//We are guaranteed not to ignore output subspace because all contracted indices handled by kernel
    	size_t cur_output_subspace = cur_loop.get_subspace_looped(0);
    	for(size_t i = 0; i < dim_lists[0][cur_output_subspace]; ++i)
    	{
			_contract_internal(ptrs,dim_lists,m,n,k,lda,ldb,ldc,loop_idx+1);
    		for(size_t bispace_idx = 0; bispace_idx < dim_lists.size(); ++bispace_idx)
    		{
				//Can do this bcs pass by value
    			if(!cur_loop.is_bispace_ignored(bispace_idx))
    			{
					ptrs[bispace_idx] += m_strides[bispace_idx];
    			}
    		}
    	}
    }
}

template<typename T>
libtensor::block_contract2_kernel<T>::block_contract2_kernel(
		const sparse_loop_list& sll) : m_loops(sll.get_loops()),m_A_trans(false),m_B_trans(false),m_strides(3,1)
{
	//Simplest contraction is matrix multiply and requires 3 loops
	if(m_loops.size() < 3)
	{
		throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
				__FILE__, __LINE__, "sparse_loop_list must contain at least 3 loops");
	}

	//Ensure that we have a valid number of bispaces
	const std::vector< sparse_bispace_any_order >& bispaces = sll.get_bispaces();
	if(bispaces.size() != 3)
	{
		throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
				__FILE__, __LINE__, "sparse_loop_list must contain at least 3 bispaces");
	}

	//Record the orders of the blocks that this contract kernel takes as input when run
	for(size_t block_idx = 0; block_idx < bispaces.size(); ++block_idx)
	{
		m_block_orders.push_back(bispaces[block_idx].get_order());
	}


	//Figure out what RHS subspace indices are contracted, based on whether
	//the loops that traverse them don't touch the output tensor
	std::vector<size_t> A_contracted_indices;
	std::vector<size_t> B_contracted_indices;
	size_t C_last_A_idx;
	size_t C_last_B_idx;
	for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
	{
		//Contracted? It should be if it doesn't appear in the output tensor
		const block_loop& cur_loop = m_loops[loop_idx];
		if(cur_loop.is_bispace_ignored(0))
		{
			//If contracted idx doesn't appear in both input tensors, it shouldn't be contracted
			if((cur_loop.is_bispace_ignored(1)) || (cur_loop.is_bispace_ignored(2)))
			{
				throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
						__FILE__, __LINE__, "an idx not present in both A and B should appear in C");
			}
			A_contracted_indices.push_back(cur_loop.get_subspace_looped(1));
			B_contracted_indices.push_back(cur_loop.get_subspace_looped(2));
		}
		else
		{
			//Uncontracted? Then it needs to be present in A or B
			if(cur_loop.is_bispace_ignored(1) && cur_loop.is_bispace_ignored(2))
			{
				throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
						__FILE__, __LINE__, "index present in C but not in A or B");
			}

			//Record these for determining if contraction has valid matmul-isomorphic order later
			if(!cur_loop.is_bispace_ignored(1))
			{
				C_last_A_idx = cur_loop.get_subspace_looped(1);
			}
			if(!cur_loop.is_bispace_ignored(2))
			{
				C_last_B_idx = cur_loop.get_subspace_looped(2);
			}
		}
	}

	if(A_contracted_indices.size() == 0)
	{
		throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
				__FILE__, __LINE__, "must have at least one contracted index");
	}

	//Check for strided output
	//Output is strided if the maximum (innermost) subspace index is not traversed last
	for(size_t loop_idx_rev = 0; loop_idx_rev < m_loops.size(); ++loop_idx_rev)
	{
		size_t loop_idx = m_loops.size() - loop_idx_rev - 1;
		const block_loop& cur_loop = m_loops[loop_idx];
		if(!cur_loop.is_bispace_ignored(0))
		{
			if(cur_loop.get_subspace_looped(0) != bispaces[0].get_order() - 1)
			{
				throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
						__FILE__, __LINE__, "strided output is not supported");
			}
			else
			{
				break;
			}
		}
	}

	//Determine the type of matmul required to execute the contraction:
	//	A.B     [ C_ij = A_ik B_kj ]
	//  A.B^T   [ C_ij = A_ik B_jk ]
	//	A^T.B   [ C_ij = A_ki B_kj ]
	//  A^T.B^T [ C_ij = A_ki B_jk ]

	//Is A transposed? If so, the contracted indices appear in order at the beginning.
	//Otherwise, they will appear in order at the end
	size_t A_order = bispaces[1].get_order();
	size_t A_first_contr_subspace;
	size_t C_last_A_idx_correct;
	if(A_contracted_indices[0] == 0)
	{
		//Yes
		m_A_trans = true;
		A_first_contr_subspace = 0;
		C_last_A_idx_correct = A_order - 1;
	}
	else
	{
		//No
		A_first_contr_subspace = A_order - A_contracted_indices.size();
		C_last_A_idx_correct = A_first_contr_subspace - 1;
	}

	//Check contracted index position and order
	for(size_t A_contr_idx = 0; A_contr_idx < A_contracted_indices.size(); ++ A_contr_idx)
	{
		if(A_contracted_indices[A_contr_idx] != A_first_contr_subspace + A_contr_idx)
		{
			throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
					__FILE__, __LINE__, "index order of A is not matmul isomorphic");
		}
	}

	//Additionally, the LAST uncontracted index present in A and C must immediately preceed/follow the
	//contracted indices
	if(C_last_A_idx != C_last_A_idx_correct)
	{
		throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
				__FILE__, __LINE__, "last uncontracted index common to A and C in wrong position");
	}


	//Is B transposed? If so the contracted indices appear in order at the end
	//Otherwise, they will appear in order at the beginning
	size_t B_order = bispaces[2].get_order();
	size_t B_first_contr_subspace;
	size_t C_last_B_idx_correct;
	if(B_contracted_indices[0] == 0)
	{
		//No
		B_first_contr_subspace = 0;
		C_last_B_idx_correct = B_order - 1;
	}
	else
	{
		//Yes
		m_B_trans = true;
		B_first_contr_subspace = B_order - B_contracted_indices.size();
		C_last_B_idx_correct  = B_first_contr_subspace - 1;
	}

	for(size_t B_contr_idx = 0; B_contr_idx < B_contracted_indices.size(); ++ B_contr_idx)
	{
		if(B_contracted_indices[B_contr_idx] != B_first_contr_subspace + B_contr_idx)
		{
			throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
					__FILE__, __LINE__, "index order of B is not matmul isomorphic");
		}
	}

	//Additionally, the LAST uncontracted index present in A and C must immediately preceed/follow the
	//contracted indices
	if(C_last_B_idx != C_last_B_idx_correct)
	{
		throw bad_parameter(g_ns, k_clazz,"block_contract2_kernel(...)",
				__FILE__, __LINE__, "last uncontracted index common to B and C in wrong position");
	}

	m_n_contracted_inds = A_contracted_indices.size();
	if(m_A_trans)
	{
		if(m_B_trans)
		{
            m_dgemm_fn = &linalg::mul2_ij_pi_jp_x;

		}
		else
		{
            m_dgemm_fn = &linalg::mul2_ij_pi_pj_x;

		}
	}
	else
	{
		if(m_B_trans)
		{
            m_dgemm_fn = &linalg::mul2_ij_ip_jp_x;

		}
		else
		{
            m_dgemm_fn = &linalg::mul2_ij_ip_pj_x;
		}
	}
}

template<typename T>
void libtensor::block_contract2_kernel<T>::operator ()(
		const std::vector<T*>& ptrs, const std::vector<dim_list>& dim_lists)
{
#ifdef LIBTENSOR_DEBUG
	if(dim_lists.size() != 3 || ptrs.size() != 3)
	{
		throw bad_parameter(g_ns, k_clazz,"operator()(...)",
				__FILE__, __LINE__, "must pass exactly 3 dim_lists and 3 ptrs");
	}

	for(size_t block_idx = 0; block_idx < dim_lists.size(); ++block_idx)
	{
		if(m_block_orders[block_idx] != dim_lists[block_idx].size())
		{
			throw bad_parameter(g_ns, k_clazz,"operator()(...)",
					__FILE__, __LINE__, "one or more block dimension lists has wrong size");
		}
	}

	//Check that all dimensions match up appropriately
	for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
	{
		const block_loop& cur_loop = m_loops[loop_idx];
		size_t non_ignored_bispace_idx;
		for(size_t bispace_idx = 0; bispace_idx < dim_lists.size(); ++bispace_idx)
		{
			if(!cur_loop.is_bispace_ignored(bispace_idx))
			{
				non_ignored_bispace_idx = bispace_idx;
				break;
			}
		}

		size_t ref_dim = dim_lists[non_ignored_bispace_idx][cur_loop.get_subspace_looped(non_ignored_bispace_idx)];
		for(size_t bispace_idx = non_ignored_bispace_idx; bispace_idx < dim_lists.size(); ++bispace_idx)
		{
			if(!cur_loop.is_bispace_ignored(bispace_idx))
			{
				if(dim_lists[bispace_idx][cur_loop.get_subspace_looped(bispace_idx)] != ref_dim)
				{
					throw bad_parameter(g_ns, k_clazz,"operator()(...)",
							__FILE__, __LINE__, "incompatible dimensions");
				}
			}
		}
	}
#endif

	size_t m,n,k = 1,lda,ldb,ldc;
	if(m_A_trans)
	{
        m = dim_lists[1].back();
        for(size_t k_idx = 0; k_idx < m_n_contracted_inds; ++k_idx)
        {
			k *= dim_lists[1][k_idx];
        }
		lda = m;
	}
	else
	{
		size_t A_i_idx = m_block_orders[1] - m_n_contracted_inds - 1;
		m = dim_lists[1][A_i_idx];
		for(size_t k_idx = A_i_idx + 1; k_idx < A_i_idx + m_n_contracted_inds + 1; ++k_idx)
		{
			k *= dim_lists[1][k_idx];
		}
		lda = k;
	}

	if(m_B_trans)
	{
		n = dim_lists[2][0];
		ldb = k;
	}
	else
	{
		n = dim_lists[2][m_n_contracted_inds];
		ldb = n;
	}
	ldc = n;
	_contract_internal(ptrs,dim_lists,m,n,k,lda,ldb,ldc);
}

#endif /* BLOCK_CONTRACT2_KERNEL_H_ */
