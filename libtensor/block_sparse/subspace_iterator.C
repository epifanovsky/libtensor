#include "subspace_iterator.h"

using namespace std;

namespace libtensor {

subspace_iterator::subspace_iterator(const sparse_bispace_any_order& bispace,size_t subspace_idx) : m_pos(0)
{
    size_t idx_grp = bispace.get_index_group_containing_subspace(subspace_idx);
    size_t idx_grp_order = bispace.get_index_group_order(idx_grp);
    size_t idx_grp_offset = bispace.get_index_group_offset(idx_grp);
    if(idx_grp_order > 1)
    {
        for(size_t sparse_grp_idx = 0; sparse_grp_idx < bispace.get_n_sparse_groups(); ++sparse_grp_idx)
        {
            size_t sparse_grp_offset = bispace.get_sparse_group_offset(sparse_grp_idx);
            if(idx_grp_offset == sparse_grp_offset)
            {
                //Permute tree so that iterated subspace is at position 0
                sparse_block_tree_any_order tree = bispace.get_sparse_group_tree(sparse_grp_idx);
                if(subspace_idx != 0)
                {
                    runtime_permutation perm(tree.get_order());
                    perm.permute(0,subspace_idx - sparse_grp_offset);
                    tree = tree.permute(perm);
                }

                for(sparse_block_tree_any_order::iterator it = tree.begin(); it != tree.end(); ++it)
                {
                    size_t block_idx = it.key()[0];
                    if(m_blocks.size() == 0 || block_idx != m_blocks.back())
                    {
                        m_blocks.push_back(block_idx);
                    }
                }
            }
        }
    }
    else
    {
        const sparse_bispace<1> subspace = bispace[subspace_idx]; 
        for(size_t i = 0; i < subspace.get_n_blocks(); ++i) 
        {
            m_blocks.push_back(i);
        }
    }

}

size_t subspace_iterator::get_block_index() const
{
    return m_blocks[m_pos]; 
}

subspace_iterator& subspace_iterator::operator++()
{
    ++m_pos;
    return *this;
}

} // namespace libtensor
