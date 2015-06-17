#include "subspace_iterator.h"

using namespace std;

namespace libtensor {

subspace_iterator::subspace_iterator(const sparse_bispace_any_order& bispace,size_t subspace_idx) : m_pos(0)
{
    const sparse_bispace<1> subspace = bispace[subspace_idx]; 
    size_t subspace_idx_grp = bispace.get_index_group_containing_subspace(subspace_idx);
    size_t idx_grp_order = bispace.get_index_group_order(subspace_idx_grp);
    size_t idx_grp_offset = bispace.get_index_group_offset(subspace_idx_grp);

    //Figure out the scale factor needed to get the total size of a slice 
    size_t slice_scale_fac = 1;
    for(size_t idx_grp = 0; idx_grp < bispace.get_n_index_groups(); ++idx_grp)
    {
        if(idx_grp == subspace_idx_grp) continue;
        slice_scale_fac *= bispace.get_index_group_dim(idx_grp);
    }

    //Build our list of blocks
    if(idx_grp_order > 1)
    {
        for(size_t sparse_grp_idx = 0; sparse_grp_idx < bispace.get_n_sparse_groups(); ++sparse_grp_idx)
        {
            size_t sparse_grp_offset = bispace.get_sparse_group_offset(sparse_grp_idx);
            if(idx_grp_offset == sparse_grp_offset)
            {
                //Permute tree so that iterated subspace is at position 0
                sparse_block_tree tree = bispace.get_sparse_group_tree(sparse_grp_idx);
                if(subspace_idx != 0)
                {
                    runtime_permutation perm(tree.get_order());
                    perm.permute(0,subspace_idx - sparse_grp_offset);
                    tree = tree.permute(perm);
                }

                size_t block_subtotal = 0;
                bool empty = (tree.begin() == tree.end());
                if(!empty) m_blocks.push_back(tree.begin().key()[0]);
                for(sparse_block_tree::iterator it = tree.begin(); it != tree.end(); ++it)
                {
                    size_t block_idx = it.key()[0];
                    if(block_idx != m_blocks.back())
                    {
                        m_blocks.push_back(block_idx);
                        m_slice_sizes.push_back(block_subtotal);
                        block_subtotal = 0;
                    }
                    block_subtotal += (*it)[0].second*slice_scale_fac; 
                }
                if(!empty) m_slice_sizes.push_back(block_subtotal);
                break;
            }
        }
    }
    else
    {
        for(size_t i = 0; i < subspace.get_n_blocks(); ++i)
        {
            m_blocks.push_back(i);
            m_slice_sizes.push_back(subspace.get_block_size(i)*slice_scale_fac);
        }
    }
}

size_t subspace_iterator::get_block_index() const
{
#ifdef LIBTENSOR_DEBUG
    if(done())
    {
        throw out_of_bounds(g_ns,"subspace_iterator","get_slice_size(...)",
                            __FILE__,__LINE__,"iterator is past end");
    }
#endif
    return m_blocks[m_pos]; 
}

size_t subspace_iterator::get_slice_size() const
{
#ifdef LIBTENSOR_DEBUG
    if(done())
    {
        throw out_of_bounds(g_ns,"subspace_iterator","get_slice_size(...)",
                            __FILE__,__LINE__,"iterator is past end");
    }
#endif
    return m_slice_sizes[m_pos]; 
}

subspace_iterator& subspace_iterator::operator++()
{
#ifdef LIBTENSOR_DEBUG
    if(done())
    {
        throw out_of_bounds(g_ns,"subspace_iterator","operator++(...)",
                            __FILE__,__LINE__,"Incremented past end");
    }
#endif
    ++m_pos;
    return *this;
}

bool subspace_iterator::done() const
{
    return (m_pos == m_blocks.size());
}

} // namespace libtensor
