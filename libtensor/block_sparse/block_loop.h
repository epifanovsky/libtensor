/*
 * block_loop_new.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef BLOCK_LOOP_H_
#define BLOCK_LOOP_H_

#include <map>
#include <vector>
#include "sparse_bispace.h"

namespace libtensor
{

class block_loop
{
private:
    static const char* k_clazz; //!< Class name
    std::vector<size_t> m_subspaces_looped; //!< What subspaces of what bispaces does this loop span?
    std::vector<bool> m_bispaces_ignored; //!< What bispaces are completely ignored by this loop?
    dim_list m_bispace_orders;
public:
	block_loop(const std::vector< sparse_bispace_any_order >& bispaces);

	void set_subspace_looped(size_t bispace_idx,size_t subspace_idx);
	size_t get_subspace_looped(size_t bispace_idx) const;

	bool is_bispace_ignored(size_t bispace_idx) const;
};

} /* namespace libtensor */

#endif /* BLOCK_LOOP_H_ */
