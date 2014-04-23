#ifndef SUBSPACE_ITERATOR_H
#define SUBSPACE_ITERATOR_H

#include "sparse_bispace.h"

namespace libtensor {

class subspace_iterator
{
public:
    subspace_iterator(const sparse_bispace_any_order& bispace,size_t subspace_idx) {}
    size_t get_block_index() const; 
};

} // namespace libtensor



#endif /* SUBSPACE_ITERATOR_H */
