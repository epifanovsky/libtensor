#ifndef SPARSE_LOOP_ITERATOR_H
#define SPARSE_LOOP_ITERATOR_H

#include <vector>
#include "sparse_bispace.h"

namespace libtensor {

namespace impl {

//TODO: Put this in sparse_defs.h
typedef std::pair<size_t,size_t > size_t_pair;

class sparse_loop_iterator
{
public:
    //Dense constructor
    sparse_loop_iterator(const sparse_bispace<1>& subspace,const std::vector< size_t_pair >& bispaces_and_index_groups);

    //Sparse constructor
    sparse_loop_iterator(const sparse_block_tree_any_order& tree,
                         const std::vector< sparse_bispace<1> >& tree_subspaces,
                         const std::vector< size_t_pair >& bispaces_and_index_groups,
                         const std::vector< size_t_pair >& sparse_bispaces_and_index_groups);

    void set_offsets_and_dims(std::vector< offset_list >& offset_lists,std::vector< dim_list >& dim_lists) const;
    sparse_loop_iterator& operator++();
    bool done() const;

    //Destructor
    ~sparse_loop_iterator();
private:
    std::vector< sparse_bispace<1> > m_subspaces;
    std::vector< size_t_pair > m_bispaces_and_index_groups;

    //Sparse iterator
    sparse_block_tree_any_order::const_iterator* m_sp_it;
    sparse_block_tree_any_order::const_iterator* m_sp_it_begin;
    sparse_block_tree_any_order::const_iterator* m_sp_it_end;
    //Dense iterator
    size_t m_cur_block;
};

} // namespace impl

} // namespace libtensor

#endif /* SPARSE_LOOP_ITERATOR_H */
