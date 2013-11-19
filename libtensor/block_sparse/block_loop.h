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
    std::map<size_t,size_t> m_subspace_map; ///!< What subspaces of what bispaces does this loop span?
    std::vector< sparse_bispace_any_order > m_bispaces; //!< What bispaces does this loop access
public:
	block_loop(const std::vector< sparse_bispace_any_order >& bispaces);

	void set_subspace_looped(size_t bispace_idx,size_t subspace_idx);
	size_t get_subspace_looped(size_t bispace_idx) const;

	bool is_bispace_ignored(size_t bispace_idx) const;

	const std::vector< sparse_bispace_any_order > get_bispaces() const { return m_bispaces; }
};

} /* namespace libtensor */

#endif /* BLOCK_LOOP_H_ */
