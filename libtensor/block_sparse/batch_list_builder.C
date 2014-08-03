#include "batch_list_builder.h"

using namespace std;

namespace libtensor {

using namespace expr;

batch_list_builder::batch_list_builder(const vector< vector<sparse_bispace_any_order> >& bispace_grps,
                                       const vector<idx_list>& batched_subspace_grps)
{
    for(size_t grp_idx = 0; grp_idx  < bispace_grps.size(); ++grp_idx)
    {
        const vector<sparse_bispace_any_order>& bispace_grp = bispace_grps[grp_idx];
        const idx_list& batched_subspace_grp = batched_subspace_grps[grp_idx];
        vector<subspace_iterator> iter_grp;
        for(size_t bispace_idx = 0; bispace_idx < bispace_grp.size(); ++bispace_idx)
        {
            const sparse_bispace_any_order& bispace = bispace_grp[bispace_idx];
            size_t subspace_idx = batched_subspace_grp[bispace_idx];
            iter_grp.push_back(subspace_iterator(bispace,subspace_idx));
            m_end_idx = bispace[subspace_idx].get_n_blocks(); 
        }
        m_iter_grps.push_back(iter_grp);
    }
}

idx_pair_list batch_list_builder::get_batch_list(size_t max_n_elem)
{
    idx_pair_list batch_list;
    dim_list grp_batch_sizes(m_iter_grps.size(),0);
    dim_list grp_cur_subtotals(m_iter_grps.size(),0);
    size_t batch_start_idx = 0;

    //Do this by value so as not to corrupt member var as we incr
    vector< vector<subspace_iterator> > iter_grps(m_iter_grps);
    vector< pair<size_t,idx_pair> > cur_block_inds; 
    while(true)
    {
        //Figure out what index each iterator is at
        bool all_done = true;
        for(size_t grp_idx = 0; grp_idx < m_iter_grps.size(); ++grp_idx)
        {
            vector<subspace_iterator>& iter_grp = iter_grps[grp_idx];
            //Did all of our iterators reach the end?
            for(size_t iter_idx = 0; iter_idx < iter_grp.size(); ++iter_idx)
            {
                const subspace_iterator& it = iter_grp[iter_idx];
                if(!it.done())
                {
                    all_done = false;
                    cur_block_inds.push_back(pair<size_t,idx_pair>(it.get_block_index(),idx_pair(grp_idx,iter_idx)));
                }
            }
        }
        if(all_done) break;

        //We need to look at the smallest index first to see if the batch can hold it
        //If other iterators are further advanced, we ignore them for now
        sort(cur_block_inds.begin(),cur_block_inds.end());
        size_t least_idx = cur_block_inds[0].first;
        for(size_t sig_it_idx = 0; sig_it_idx < cur_block_inds.size(); ++sig_it_idx)
        {
            size_t block_idx = cur_block_inds[sig_it_idx].first;
            if(block_idx != least_idx) break;
            size_t grp_idx = cur_block_inds[sig_it_idx].second.first;
            size_t iter_idx = cur_block_inds[sig_it_idx].second.second;
            size_t this_block_contrib = iter_grps[grp_idx][iter_idx].get_slice_size();
            if(this_block_contrib > max_n_elem)
            {
                throw out_of_memory(g_ns,"batch_list_builder","get_batch_list(...)",__FILE__,__LINE__,
                    "Not enough memory provided to compute valid batch list"); 
            }
            grp_cur_subtotals[grp_idx] += this_block_contrib;
            ++iter_grps[grp_idx][iter_idx];
        }

        //Find grps that have exceeded the memory allowance
        //If any of them have the current smallest index, that marks the end of the batch
        for(size_t sig_it_idx = 0; sig_it_idx < cur_block_inds.size(); ++sig_it_idx)
        {
            size_t grp_idx = cur_block_inds[sig_it_idx].second.first;
            if(cur_block_inds[sig_it_idx].first == least_idx)
            {
                if(grp_batch_sizes[grp_idx] + grp_cur_subtotals[grp_idx] > max_n_elem) 
                {
                    grp_batch_sizes[grp_idx] = grp_cur_subtotals[grp_idx];
                    //To ensure that multiple grps at same index don't duplicate
                    if(least_idx > batch_start_idx)
                    {
                        batch_list.push_back(idx_pair(batch_start_idx,least_idx));
                        batch_start_idx = least_idx;
                    }
                }
                else
                {
                    grp_batch_sizes[grp_idx] += grp_cur_subtotals[grp_idx];
                }
                grp_cur_subtotals[grp_idx] = 0;
            }
        }
        cur_block_inds.resize(0);
    }
    batch_list.push_back(idx_pair(batch_start_idx,m_end_idx));
    return batch_list;
}

} // namespace libtensor
