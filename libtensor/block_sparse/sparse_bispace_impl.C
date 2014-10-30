#include "sparse_bispace_impl.h"

using namespace std;

namespace libtensor {
    
const char* sparse_bispace_impl::k_clazz = "sparse_bispace_impl";

bool sparse_bispace_impl::operator==(const sparse_bispace_impl& rhs) const
{
    return (m_subspaces == rhs.m_subspaces) && 
           (m_trees == rhs.m_trees) && 
           (m_tree_offsets == rhs.m_tree_offsets);
}

sparse_bispace_impl::sparse_bispace_impl(const sparse_bispace_impl& lhs,
                                         const sparse_bispace_impl& rhs)
{
    m_subspaces.insert(m_subspaces.end(),lhs.m_subspaces.begin(),lhs.m_subspaces.end());
    m_subspaces.insert(m_subspaces.end(),rhs.m_subspaces.begin(),rhs.m_subspaces.end());
    m_trees.insert(m_trees.end(),lhs.m_trees.begin(),lhs.m_trees.end());
    m_trees.insert(m_trees.end(),rhs.m_trees.begin(),rhs.m_trees.end());
    m_tree_offsets.insert(m_tree_offsets.end(),lhs.m_tree_offsets.begin(),lhs.m_tree_offsets.end());

    //Must redo offsets stemming from the RHS tree
    m_tree_offsets.insert(m_tree_offsets.end(),rhs.m_tree_offsets.begin(),rhs.m_tree_offsets.end());

    for(size_t i = 0; i < rhs.m_tree_offsets.size(); ++i)
    {
        m_tree_offsets[m_tree_offsets.size()-i-1] += lhs.m_subspaces.size();
    }
}

sparse_bispace_impl::sparse_bispace_impl(const vector<subspace>& subspaces,
                                         const sparse_block_tree& tree) : m_subspaces(subspaces),m_trees(1,tree),m_tree_offsets(1,0)
{
}

//TODO: This is haxx, needs to be formally verified
//TODO: This will break if I relocate an entire tree!!!!
sparse_bispace_impl sparse_bispace_impl::permute(const runtime_permutation& perm) const
{
    //Sparse metadata is rebuilt as we go
    sparse_bispace_impl copy(*this);
    copy.m_tree_offsets.clear();

    //Permute subspaces
    for(size_t i = 0; i < m_subspaces.size(); ++i)
    {
        copy.m_subspaces[i] = m_subspaces[perm[i]];
    }

    //Permute trees
    vector<size_t> cur_perm_entries;
    vector<subspace> cur_tree_subspaces;
    size_t cur_tree_idx;
    size_t cur_order;
    size_t cur_tree_start_dest_sub_idx;
    map<size_t,size_t> cur_dense_subspaces;
    for(size_t dest_sub_idx = 0; dest_sub_idx < m_subspaces.size(); ++dest_sub_idx)
    {
        size_t src_sub_idx = perm[dest_sub_idx];

        //Is this index associated with a sparse group?
        bool sparse = false;
        for(size_t tree_idx = 0; tree_idx < m_tree_offsets.size(); ++tree_idx)
        {
            size_t offset = m_tree_offsets[tree_idx];
            size_t order = m_trees[tree_idx].get_order();
            if((offset <= src_sub_idx) && (src_sub_idx < offset+order))
            {
                //This index comes from this sparse group in the unpermuted bispace
                if(cur_perm_entries.size() == 0)
                {
                    //This is the first index in our tree in the new,permuted,bispace
                    cur_tree_idx = tree_idx;
                    cur_order = order;
                    cur_tree_start_dest_sub_idx = dest_sub_idx;
                }
                else if(tree_idx != cur_tree_idx)
                {
                    //This index comes from a different tree, and we haven't filled in our original tree yet
                    throw bad_parameter(g_ns,"sparse_bispace<N>","permute(...)",
                        __FILE__,__LINE__,"permuting between different sparse groups is not supported"); 
                }

                cur_tree_subspaces.push_back(m_subspaces[src_sub_idx]);
                cur_perm_entries.push_back(src_sub_idx - offset);
                sparse = true;
                break;
            }
        }

        if(cur_perm_entries.size() > 0)
        {
            //Did this iteration fill in our tree in the destination bispace?
            if(cur_perm_entries.size() == cur_order)
            {
                //Insert all dense subspaces caught inside this tree
                for(std::map<size_t,size_t>::iterator it = cur_dense_subspaces.begin(); it != cur_dense_subspaces.end(); ++it)
                {
                    size_t cur_dest_sub_idx = it->first;
                    size_t cur_src_sub_idx = it->second;
                    for(size_t entry_idx = 0; entry_idx < cur_perm_entries.size(); ++entry_idx)
                    {
                        if(cur_perm_entries[entry_idx] >= cur_dest_sub_idx)
                        {
                            ++cur_perm_entries[entry_idx];
                        }
                    }
                    cur_perm_entries.insert(cur_perm_entries.begin()+cur_dest_sub_idx,cur_dest_sub_idx);
                    copy.m_trees[cur_tree_idx] = copy.m_trees[cur_tree_idx].insert_subspace(cur_dest_sub_idx - cur_tree_start_dest_sub_idx,m_subspaces[cur_src_sub_idx]);
                }
                runtime_permutation tree_perm(cur_perm_entries);

                //Don't permute if identity
                if(tree_perm != runtime_permutation(cur_order))
                {
                    copy.m_trees[cur_tree_idx] = copy.m_trees[cur_tree_idx].permute(tree_perm);
                    copy.m_trees[cur_tree_idx].set_offsets_sizes_nnz(cur_tree_subspaces);
                }
                copy.m_tree_offsets.push_back(cur_tree_start_dest_sub_idx);

                cur_dense_subspaces.clear();
                cur_perm_entries.clear();
                cur_tree_subspaces.clear();
            }
            else if(!sparse)
            {
                //We don't want to log irrelevant dense bispaces - only ones caught between sparse bispaces
                cur_dense_subspaces[dest_sub_idx - cur_tree_start_dest_sub_idx] = src_sub_idx;
                cur_tree_subspaces.push_back(m_subspaces[src_sub_idx]);
            }
        }
    }
    return copy;
}

} // namespace libtensor
