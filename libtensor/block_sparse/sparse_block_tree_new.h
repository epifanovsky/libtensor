#ifndef SPARSE_BLOCK_TREE_NEW_H
#define SPARSE_BLOCK_TREE_NEW_H

#include "sparse_block_tree_any_order_new.h"

namespace libtensor {

namespace impl {
    
//Convenience typedef for most common template values used in rest of library
typedef sparse_block_tree_node_new<size_t,std::vector<size_t> > sparse_block_tree_any_order_new;

template<size_t N>
class sparse_block_tree_new : public sparse_block_tree_any_order_new
{
public:
    typedef sparse_block_tree_any_order_new::iterator iterator;

    //Constructor - just exposes the base class constructor
    sparse_block_tree_new(const std::vector< sequence<N,size_t> >& sig_blocks) : sparse_block_tree_any_order_new(sig_blocks) {};
};

} // namespace impl

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_H */
