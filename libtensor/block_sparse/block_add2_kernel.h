/*
 * block_add2_kernel.h
 *
 *  Created on: Nov 20, 2013
 *      Author: smanzer
 */

#ifndef BLOCK_ADD2_KERNEL_H_
#define BLOCK_ADD2_KERNEL_H_

#include "block_kernel_i.h"

namespace libtensor {

template<typename T>
class block_add2_kernel: public libtensor::block_kernel_i<block_add2_kernel<T>,T>
{
private:
    static const char* k_clazz; //!< Class name
    T m_lhs_scalar;
    T m_rhs_scalar;
public:
    block_add2_kernel(T lhs_scalar,T rhs_scalar) : m_lhs_scalar(lhs_scalar),m_rhs_scalar(rhs_scalar) {}
	void operator()(const std::vector<T*>& ptrs, const std::vector< dim_list >& dim_lists);
};

template<typename T>
const char* block_add2_kernel<T>::k_clazz = "block_contract2_kernel<T>";

} /* namespace libtensor */

template<typename T>
void libtensor::block_add2_kernel<T>::operator()(
		const std::vector<T*>& ptrs, const std::vector<dim_list>& dim_lists)
{
#ifdef LIBTENSOR_DEBUG
	if(dim_lists.size() != 3 || ptrs.size() != dim_lists.size())
	{
		throw bad_parameter(g_ns, k_clazz,"operator()(...)",
				__FILE__, __LINE__, "invalid number of pointers or dim_lists");
	}

	//Check that dimensions for all blocks are the same
	size_t first_size = dim_lists[0].size();
	const dim_list& first_dims = dim_lists[0];
	for(size_t i = 1; i < dim_lists.size(); ++i)
	{
		const dim_list& cur_dims = dim_lists[i];
		if(cur_dims.size() != first_size)
		{
			throw bad_parameter(g_ns, k_clazz,"operator()(...)",
					__FILE__, __LINE__, "dim lists are not all the same size");
		}

		for(size_t j = 0; j < first_size; ++j)
		{
			if(cur_dims[j] != first_dims[j])
			{
				throw bad_parameter(g_ns, k_clazz,"operator()(...)",
						__FILE__, __LINE__, "dimensions do not match. Permuted addition is not currently supported.");
			}
		}
	}
#endif

	//Just do the add
	size_t n_elements = 1;
	for(size_t i = 0; i < first_size; ++i)
	{
		n_elements *= first_dims[i];
	}

	for(size_t i = 0; i < n_elements; ++i)
	{
		ptrs[0][i] = m_lhs_scalar*ptrs[1][i] + m_rhs_scalar*ptrs[2][i];
	}
}

#endif /* BLOCK_ADD2_KERNEL_H_ */
