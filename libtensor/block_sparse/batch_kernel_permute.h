#ifndef BATCH_KERNEL_PERMUTE_H
#define BATCH_KERNEL_PERMUTE_H

#include "batch_kernel.h"
#include "gen_labeled_btensor.h"
#include "sparse_loop_list.h"
#include "block_permute_kernel.h"

namespace libtensor {

template<typename T>
class batch_kernel_permute : public batch_kernel<T>
{
public:
    static const char* k_clazz; //!< Class name

    template<size_t N>
    batch_kernel_permute(const gen_labeled_btensor<N,T>& lhs,const gen_labeled_btensor<N,T>& rhs);
    virtual void generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches);

    sparse_loop_list* m_sll_ptr;
    block_permute_kernel<T>* m_bpk_ptr;

    ~batch_kernel_permute() { delete m_sll_ptr; delete m_bpk_ptr; }
    batch_kernel_permute(const batch_kernel_permute<T>& rhs) : m_sll_ptr(new sparse_loop_list(*rhs.m_sll_ptr)),m_bpk_ptr(new block_permute_kernel<T>(*rhs.m_bpk_ptr)) {}
    batch_kernel_permute& operator=(const batch_kernel_permute<T>& rhs) { m_sll_ptr = new sparse_loop_list(*rhs.m_sll_ptr); m_bpk_ptr = new block_permute_kernel<T>(*rhs.m_bpk_ptr); }
};

template<typename T>
const char* batch_kernel_permute<T>::k_clazz = "batch_kernel_permute";

template<typename T> template<size_t N>
batch_kernel_permute<T>::batch_kernel_permute(const gen_labeled_btensor<N,T>& lhs,const gen_labeled_btensor<N,T>& rhs)
{
        std::vector<size_t> permutation_entries(N);
        expr::label<N> lhs_le = lhs.get_letter_expr();
        expr::label<N> rhs_le = rhs.get_letter_expr();
        std::vector<sparse_bispace_any_order> bispaces(1,lhs.get_bispace());
        bispaces.push_back(rhs.get_bispace());
        std::vector<block_loop> loops;
        for(size_t i = 0; i < N; ++i)
        {
            const letter& a = lhs_le.letter_at(i);
            size_t rhs_idx = rhs_le.index_of(a);
            permutation_entries[i] = rhs_idx;

            //Populate the loop for this index
            block_loop bl(bispaces);
            bl.set_subspace_looped(0,i);
            bl.set_subspace_looped(1,rhs_idx);
            loops.push_back(bl);
        }

        idx_list direct_tensors;
        if(lhs.get_data_ptr() == NULL)
        {
            direct_tensors.push_back(0);
        }
        if(rhs.get_data_ptr() == NULL)
        {
            direct_tensors.push_back(1);
        }

        runtime_permutation perm(permutation_entries);
        m_bpk_ptr = new block_permute_kernel<T>(perm);
        m_sll_ptr = new sparse_loop_list(loops,bispaces,direct_tensors);
}
    
template<typename T>
void batch_kernel_permute<T>::generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches)
{
    std::map<size_t,idx_pair> loop_batches;
    const std::vector<block_loop>& loops = m_sll_ptr->get_loops();
    for(bispace_batch_map::const_iterator batch_it = batches.begin(); batch_it != batches.end(); ++batch_it)
    {
        size_t bispace_idx = batch_it->first.first;
        size_t subspace_idx = batch_it->first.second;
        for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
        {
            const block_loop& loop = loops[loop_idx];
            if(!loop.is_bispace_ignored(bispace_idx) && loop.get_subspace_looped(bispace_idx) == subspace_idx)
            {
                loop_batches[loop_idx] = batch_it->second;
            }
        }
    }
    if(loop_batches.size() > 1)
    {
        throw bad_parameter(g_ns, k_clazz,"generate_batch(...)",__FILE__, __LINE__,
            "Batching over multiple loops is currently unsupported");
    }
    m_sll_ptr->run(*m_bpk_ptr,ptrs,loop_batches);
}

} // namespace libtensor

#endif /* BATCH_KERNEL_PERMUTE_H */
