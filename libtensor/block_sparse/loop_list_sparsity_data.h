/*
 * loop_list_sparsity_data.h
 *
 *  Created on: Nov 14, 2013
 *      Author: smanzer
 */

#ifndef LOOP_LIST_SPARSITY_DATA_H_
#define LOOP_LIST_SPARSITY_DATA_H_

#include <utility>
#include <vector>
#include <map>
#include "sparse_defs.h"
#include "sparse_bispace.h"

namespace libtensor
{

//Forward declaration for classes referencing each other
class sparse_loop_list;

class loop_list_sparsity_data
{
private:
	std::vector< block_list > m_subspace_block_lists;
	std::vector< impl::sparse_block_tree_any_order > m_trees;
	std::map< size_t, std::pair<size_t,size_t> > m_loops_to_tree_subspaces;
public:
	//We choose to friend loop_list and use its loop vector member var instead of having the vector passed directly
	//because then sparse_loop_list handles validating the loops
	loop_list_sparsity_data(const sparse_loop_list& loop_list);

	block_list get_sig_block_list(const block_list& loop_indices,size_t loop_idx) const;
};

} /* namespace libtensor */

#endif /* LOOP_LIST_SPARSITY_DATA_H_ */
