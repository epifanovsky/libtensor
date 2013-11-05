#ifndef SPARSE_BLOCK_TREE_H
#define SPARSE_BLOCK_TREE_H

#include <vector>
#include <algorithm>
#include <utility>
#include "../core/sequence.h"
#include "../core/permutation.h"
#include "sparse_block_tree_node.h"
#include "sparse_block_tree_iterator.h"
#include "sparse_block_tree_any_order.h"
#include "runtime_permutation.h"

//TODO REMOVE
#include <iostream>

namespace libtensor {

template<size_t N>
class sparse_block_tree : public sparse_block_tree_any_order {
    //Private constructor for implicit conversion to keep things wrapped nicely
    sparse_block_tree<N>(const sparse_block_tree_any_order& rhs) : sparse_block_tree_any_order(rhs) {}

    //Don't need this method - hide it
    size_t get_order() const { return 0; };
public:
    //Constructor : exposes auto-length checking interface to the base class constructor
    sparse_block_tree(const std::vector< sequence<N,size_t> >& sig_blocks);


    sparse_block_tree<N> permute(const runtime_permutation& perm) const { return sparse_block_tree_any_order::permute(perm); }

    sparse_block_tree<N-1> contract(size_t contract_idx) const { return sparse_block_tree_any_order::contract(contract_idx); }

    template<size_t M>
    sparse_block_tree<N+M-1> fuse(const sparse_block_tree<M>& rhs) const { return sparse_block_tree_any_order::fuse(rhs); }

    size_t set_offsets(const std::vector< sparse_bispace<1> >& subspaces,const sequence<N,size_t>& positions);

    //All other methods directly call parents without  implict conversion
    //
    //Friend for contract(), fuse()
    template<size_t M>
    friend class sparse_block_tree;
};

//Wraps parent constructor
template<size_t N>
sparse_block_tree<N>::sparse_block_tree(const std::vector< sequence<N,size_t> >& sig_blocks)
{
    this->m_order = N;
    std::vector< std::vector<size_t> > sig_blocks_vecs(sig_blocks.size());
    for(size_t i = 0; i < sig_blocks.size(); ++i)
    {
        sig_blocks_vecs[i].resize(N);
        for(size_t j = 0; j < N; ++j)
        {
            sig_blocks_vecs[i][j] = sig_blocks[i][j];
        }
    }
    init(sig_blocks_vecs);
}

template<size_t N>
size_t sparse_block_tree<N>::set_offsets(const std::vector< sparse_bispace<1> >& subspaces,const sequence<N,size_t>& positions)
{
    return sparse_block_tree_any_order::set_offsets(subspaces,std::vector<size_t>(&positions[0],&positions[0]+N));
}

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_H */
