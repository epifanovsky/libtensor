#ifndef SPARSE_BLOCK_TREE_NEW_H
#define SPARSE_BLOCK_TREE_NEW_H

#include "sparse_block_tree_any_order_new.h"
#include "sparse_block_tree_iterator_new.h"

namespace libtensor {

namespace impl {
    
template<size_t N>
class sparse_block_tree_new : public sparse_block_tree_any_order_new
{
private:
    //Private constructor for implicit conversion to keep things wrapped nicely
    sparse_block_tree_new(const sparse_block_tree_any_order_new& rhs) : sparse_block_tree_any_order_new(rhs) {}
public:
    typedef sparse_block_tree_any_order_new::iterator iterator;

    //Constructor - just exposes the base class constructor
    sparse_block_tree_new(const std::vector< sequence<N,size_t> >& sig_blocks) : sparse_block_tree_any_order_new(sig_blocks) {};

    //Wraps base class method
    sparse_block_tree_new<N> permute(const runtime_permutation& perm) const { return sparse_block_tree_any_order_new::permute(perm); }
};

} // namespace impl

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_H */
