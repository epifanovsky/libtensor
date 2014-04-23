#include "subspace_iterator.h"

using namespace std;

namespace libtensor {

subspace_iterator::subspace_iterator(const sparse_bispace_any_order& bispace,size_t subspace_idx)
{
}

size_t subspace_iterator::get_block_index() const
{
    return 0; 
}

} // namespace libtensor
