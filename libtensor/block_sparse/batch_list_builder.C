#include "batch_list_builder.h"

using namespace std;

namespace libtensor {

batch_list_builder::batch_list_builder(const vector< vector<labeled_bispace> >& labeled_bispace_groups,const letter& batched_index)
{
    for(size_t grp_idx = 0; grp_idx  < labeled_bispace_groups.size(); ++grp_idx)
    {
        const vector<labeled_bispace>& lbs = labeled_bispace_groups[grp_idx];
        vector<subspace_iterator> iter_group;
        for(size_t bispace_idx = 0; bispace_idx < lbs.size(); ++bispace_idx)
        {
            sparse_bispace_any_order bispace = lbs[bispace_idx].get_bispace();
            size_t subspace_idx = lbs[bispace_idx].index_of(batched_index);
            iter_group.push_back(subspace_iterator(bispace,subspace_idx));
            m_end_idx = bispace[subspace_idx].get_n_blocks(); 
        }
        m_iter_groups.push_back(iter_group);
    }
}

idx_pair_list batch_list_builder::get_batch_list(size_t max_n_elem)
{
    size_t batch_size = 0;
    size_t batch_start_idx = 0;
    idx_pair_list batch_list;
    for(size_t grp_idx = 0; grp_idx < m_iter_groups.size(); ++grp_idx)
    {
        //Do this by value so as not to corrupt member var as we incr
        vector<subspace_iterator> iter_group = m_iter_groups[grp_idx];
        while(true)
        {
            //Did all of our iterators reach the end?
            bool all_done = true;
            idx_pair_list block_inds_it_pairs;
            for(size_t iter_idx = 0; iter_idx < iter_group.size(); ++iter_idx)
            {
                const subspace_iterator& it = iter_group[iter_idx];
                if(!it.done())
                {
                    all_done = false;
                    block_inds_it_pairs.push_back(idx_pair(it.get_block_index(),iter_idx));
                }
            }

            if(all_done)
            {
                batch_list.push_back(idx_pair(batch_start_idx,m_end_idx));
                break;
            }

            //We need to look at the smallest index first to see if the batch can hold it
            //If other iterators are further advanced, we ignore them for now
            sort(block_inds_it_pairs.begin(),block_inds_it_pairs.end());
            size_t most_recent_subtotal = 0;
            for(size_t sig_it_idx = 0; sig_it_idx < block_inds_it_pairs.size(); ++sig_it_idx)
            {
                size_t block_idx = block_inds_it_pairs[sig_it_idx].first;
                if(block_idx!= block_inds_it_pairs[0].first) break;
                size_t iter_idx = block_inds_it_pairs[sig_it_idx].second;
                subspace_iterator& it = iter_group[iter_idx];
                most_recent_subtotal += it.get_slice_size();
                ++it;
            }
            batch_size += most_recent_subtotal;

            if(batch_size > max_n_elem)
            {
                size_t batch_end_idx = block_inds_it_pairs[0].first;
                batch_list.push_back(idx_pair(batch_start_idx,batch_end_idx));
                batch_start_idx = batch_end_idx;
                batch_size = most_recent_subtotal;
            }
        }
    }
    return batch_list;
}

} // namespace libtensor
