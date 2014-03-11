#ifndef PERMUTE_H
#define PERMUTE_H

#include "block_permute_kernel.h"
#include "batch_provider.h"
#include "sparse_loop_list.h"

namespace libtensor {

template<typename T>
class permute2_batch_provider : public batch_provider<T>
{
private:
    static const char *k_clazz; //!< Class name
    block_permute_kernel<T>* m_bpk_ptr;

    virtual void run_impl(const std::vector<block_loop>& loops,
                          const idx_list& direct_tensors,
                          const std::vector<sparse_bispace_any_order>& truncated_bispaces,
                          const std::vector<T*>& ptrs,
                          const std::map<size_t,idx_pair>& loop_batches)
    {
        sparse_loop_list sll(loops,direct_tensors);
        sll.run(*m_bpk_ptr,ptrs,loop_batches);
    }

    template<size_t N>
    static idx_list init_direct_tensors(const gen_labeled_btensor<N>& lhs,const gen_labeled_btensor<N>& rhs)
    {
        idx_list direct_tensors;
        if(lhs.get_data_ptr() == NULL)
        {
            direct_tensors.push_back(0);
        }
        if(rhs.get_data_ptr() == NULL)
        {
            direct_tensors.push_back(1);
        }
        return direct_tensors;
    }

public:
    template<size_t N>
    permute2_batch_provider(const gen_labeled_btensor<N>& lhs,const gen_labeled_btensor<N>& rhs) : batch_provider<T>(std::vector<block_loop>(),init_direct_tensors(lhs,rhs),std::vector<batch_provider<T>*>(),std::vector<T*>(),0)
    {
        //Determine the permutation of indices between the two tensors
        //We also populate the loops necessary to execute the transformation
        std::vector<size_t> permutation_entries(N);
        std::vector< sparse_bispace_any_order > bispaces(1,lhs.get_bispace());
        bispaces.push_back(rhs.get_bispace());

        expr::label<N> lhs_le = lhs.get_letter_expr();
        expr::label<N> rhs_le = rhs.get_letter_expr();
        for(size_t i = 0; i < N; ++i)
        {
            const letter& a = lhs_le.letter_at(i);
            size_t rhs_idx = rhs_le.index_of(a);
            permutation_entries[i] = rhs_idx;

            //Populate the loop for this index
            block_loop bl(bispaces);
            bl.set_subspace_looped(0,i);
            bl.set_subspace_looped(1,rhs_idx);
            this->m_loops.push_back(bl);
        }

        runtime_permutation perm(permutation_entries);
        m_bpk_ptr = new block_permute_kernel<T>(perm);

        if(rhs.get_data_ptr() == NULL)
        {
            this->m_batch_providers.push_back(rhs.get_batch_provider());
        }

        //Batch providers for direct tensors

        //Deliberately case away the const
        this->m_ptrs.push_back((T*)lhs.get_data_ptr());
        this->m_ptrs.push_back((T*)rhs.get_data_ptr());
    }

    virtual ~permute2_batch_provider() { delete m_bpk_ptr; }
    virtual batch_provider<T>* clone() const { return new permute2_batch_provider(*this); }
};

template<typename T>
const char* permute2_batch_provider<T>::k_clazz = "permute2_batch_provider<T>";

} // namespace libtensor

#endif /* PERMUTE_H */
