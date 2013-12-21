#include "sparse_loop_iterator.h"

using namespace std;

namespace libtensor {

namespace impl {

sparse_loop_iterator::sparse_loop_iterator(const sparse_bispace<1>& subspace,const vector< size_t_pair >& bispaces_and_index_groups) : m_subspaces(1,subspace),
                                                                                                                                       m_bispaces_and_index_groups(bispaces_and_index_groups),
                                                                                                                                       m_sp_it_begin(NULL),m_sp_it_end(NULL),m_cur_block(0)
{
}

sparse_loop_iterator::sparse_loop_iterator(const sparse_block_tree_any_order& tree,
                                           const std::vector< sparse_bispace<1> >& tree_subspaces,
                                           const std::vector< size_t_pair >& bispaces_and_index_groups,
                                           const std::vector< size_t_pair >& sparse_bispaces_and_index_groups) : m_subspaces(tree_subspaces),
                                                                                                                 m_bispaces_and_index_groups(bispaces_and_index_groups)
{
    m_sp_it_begin = new sparse_block_tree_any_order::const_iterator(tree.begin());
    m_sp_it_end = new sparse_block_tree_any_order::const_iterator(tree.end());
    m_sp_it = m_sp_it_begin;
}

void sparse_loop_iterator::set_offsets_and_dims(std::vector< offset_list >& offset_lists,std::vector< dim_list >& dim_lists) const
{
    /*if(m_sparse_bispaces_and_index_groups.size() > 0)*/
    /*{*/
        /*//For each bispace with a sparse index group that is part of this tree, extract the*/
        /*//pertinent parts of the key to get the size*/
        /*const block_list& key = m_sp_it->key();*/
        /*size_t dim = 1;*/
        /*for(bispace_tree_entry_map_t::iterator it = m_bispace_kv_idx_map.begin(); it != m_bispace_kv_idx_map.end(); ++it)*/
        /*{*/
            /*//Figure out the size of the block based on the key*/
            /*size_t dim = 1; */
            /*const offset_list& key_idxs = it->second.first;*/
            /*for(size_t key_idx = 0; key_idx < key_idxs.size(); ++key_idx)*/
            /*{*/
                /*dim *= m_subspaces[key_idx].get_block_size(key[key_idxs[key_idx]]);*/
            /*}*/

            /*//Load the offset from the tree leaf values*/
            /*size_t bispace_idx = it->first;*/
            /*const size_t_pair& b_ig = m_bispaces_and_index_groups[bispace_idx];*/
            /*size_t offset_idx = it->second->second;*/
            /*offset_lists[b_ig.first][b_ig.second] =  (*m_sp_it)[offset_idx];*/
            /*dim_lists[b_ig.first][b_ig.second] = dim;*/
        /*}*/
    /*}*/

    for(size_t i = 0; i < m_bispaces_and_index_groups.size(); ++i)
    {
        const size_t_pair& b_ig = m_bispaces_and_index_groups[i];
        offset_lists[b_ig.first][b_ig.second] =  m_subspaces[0].get_block_abs_index(m_cur_block);
        dim_lists[b_ig.first][b_ig.second] = m_subspaces[0].get_block_size(m_cur_block);
    }
}

sparse_loop_iterator& sparse_loop_iterator::operator++()
{
    ++m_cur_block;
    return *this;
}

bool sparse_loop_iterator::done() const
{
    return (m_cur_block == (m_subspaces[0].get_n_blocks() - 1));
}

sparse_loop_iterator::~sparse_loop_iterator()
{
    if(m_sp_it_begin != NULL)
    {
        delete m_sp_it_begin;
    }
    if(m_sp_it_end != NULL)
    {
        delete m_sp_it_end;
    }
}

} // namespace impl

} // namespace libtensor
