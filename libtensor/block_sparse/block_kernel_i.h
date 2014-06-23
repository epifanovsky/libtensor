/*
 * block_kernel_i_new.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef BLOCK_KERNEL_I_NEW_H_
#define BLOCK_KERNEL_I_NEW_H_

#include <vector>
#include "sparse_defs.h"
#include "../exception.h"

namespace libtensor {

template<typename kern_t,typename T>
class block_kernel_i
{
public:
	void operator()(const std::vector<T*>& ptrs, const std::vector< dim_list >& dim_lists)
    {
        (*static_cast<kern_t*>(this))(ptrs,dim_lists);
    }
	virtual ~block_kernel_i() {}
};
} // namespace libtensor


#endif /* BLOCK_KERNEL_I_NEW_H_ */
