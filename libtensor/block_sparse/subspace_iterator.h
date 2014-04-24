#ifndef SUBSPACE_ITERATOR_H
#define SUBSPACE_ITERATOR_H

#include "sparse_bispace.h"

namespace libtensor {

class subspace_iterator
{
    size_t m_pos;
    block_list m_blocks;
    dim_list m_slice_sizes; 
public:
    subspace_iterator(const sparse_bispace_any_order& bispace,size_t subspace_idx);
    size_t get_block_index() const; 
    size_t get_slice_size() const; 
    subspace_iterator& operator++();
    bool done() const;
};

} // namespace libtensor



#endif /* SUBSPACE_ITERATOR_H */
