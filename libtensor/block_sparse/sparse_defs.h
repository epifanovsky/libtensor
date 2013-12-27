/*
 * sparse_defs.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef SPARSE_DEFS_H_
#define SPARSE_DEFS_H_

#include <vector>

namespace libtensor {

typedef std::vector<size_t> idx_list;
typedef std::vector<size_t> block_list;
typedef std::vector<size_t> dim_list;
typedef std::vector<size_t> offset_list;
typedef std::pair<size_t,size_t> off_dim_pair;

} // namespace libtensor



#endif /* SPARSE_DEFS_H_ */
