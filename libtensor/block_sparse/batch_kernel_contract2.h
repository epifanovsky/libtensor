#ifndef BATCH_KERNEL_CONTRACT2_H
#define BATCH_KERNEL_CONTRACT2_H

#include "batch_kernel.h"
#include "gen_sparse_btensor.h"
#include "sparse_loop_list.h"
#include "block_contract2_kernel.h"

namespace libtensor {

template<typename T>
class batch_kernel_contract2 : public batch_kernel<T>
{
private:
public:
    static const char* k_clazz; //!< Class name

    template<size_t NC,size_t NA,size_t NB>
    batch_kernel_contract2(const gen_sparse_btensor<NC,T>& C,const gen_sparse_btensor<NA,T>& A,const gen_sparse_btensor<NB,T>& B,const std::multimap<size_t,size_t>& contr_map);

    virtual void generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches);

    sparse_loop_list* m_sll_ptr;
    block_contract2_kernel<T>* m_bc2k_ptr;

    ~batch_kernel_contract2() { delete m_sll_ptr; delete m_bc2k_ptr; }
    batch_kernel_contract2(const batch_kernel_contract2<T>& rhs) : m_sll_ptr(new sparse_loop_list(*rhs.m_sll_ptr)),m_bc2k_ptr(new block_contract2_kernel<T>(*rhs.m_bc2k_ptr)) {}
    batch_kernel_contract2& operator=(const batch_kernel_contract2<T>& rhs) { m_sll_ptr = new sparse_loop_list(*rhs.m_sll_ptr); m_bc2k_ptr = new block_contract2_kernel<T>(*rhs.m_bc2k_ptr); }
};

template<typename T>
const char* batch_kernel_contract2<T>::k_clazz = "batch_kernel_contract2";

template<typename T> template<size_t NC,size_t NA,size_t NB>
batch_kernel_contract2<T>::batch_kernel_contract2(const gen_sparse_btensor<NC,T>& C,const gen_sparse_btensor<NA,T>& A,const gen_sparse_btensor<NB,T>& B,const std::multimap<size_t,size_t>& contr_map)
{
    if(NC != NA+NB - 2*contr_map.size())
    {
        throw bad_parameter(g_ns, k_clazz,"batch_kernel_contract2(...)",
                __FILE__, __LINE__, "Invalid tensor orders");
    }

    std::vector<sparse_bispace_any_order> bispaces(1,C.get_bispace());
    bispaces.push_back(A.get_bispace());
    bispaces.push_back(B.get_bispace());

    std::vector<block_loop> contracted_loops;
    std::multimap<size_t,size_t> contr_inv;
    for(std::multimap<size_t,size_t>::const_iterator it = contr_map.begin(); it != contr_map.end(); ++it)
    {
        block_loop bl(bispaces);
        bl.set_subspace_looped(1,it->first);
        bl.set_subspace_looped(2,it->second - NA);
        contracted_loops.push_back(bl);
        contr_inv.insert(idx_pair(it->second,it->first));
    }

    std::multimap<size_t,size_t> uncontr_map;
    size_t m = 0; 
    for(size_t i = 0; i < NA+NB; ++i)
    {
        if(i < NA && (contr_map.find(i) == contr_map.end()))
        {
            uncontr_map.insert(idx_pair(m,i));
            ++m;
        }
        else if(i >= NA && (contr_inv.find(i) == contr_inv.end()))
        {
            uncontr_map.insert(idx_pair(m,i));
            ++m;
        }
    }

    m = 0;
    std::vector<block_loop> uncontracted_loops(NC,block_loop(bispaces));
    for(std::multimap<size_t,size_t>::iterator it = uncontr_map.begin(); it != uncontr_map.end(); ++it)
    {
        uncontracted_loops[m].set_subspace_looped(0,it->first);
        size_t bispace_idx = it->second < NA ? 1 : 2;
        size_t subspace_idx = it->second < NA ? it->second : it->second - NA;
        uncontracted_loops[m].set_subspace_looped(bispace_idx,subspace_idx);
        ++m;
    }

    //Figure out whether we should make the loops over the contracted or uncontracted indices
    //the outer loops based on a crude estimate of their combined size.
    //We wanted contracted indices as outer loops for dot-product like things
    size_t uncontracted_dim = 1;
    for(size_t loop_idx = 0; loop_idx < uncontracted_loops.size(); ++loop_idx)
    {
        const block_loop& loop = uncontracted_loops[loop_idx];
        for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
        {
            if(!loop.is_bispace_ignored(bispace_idx))
            {
                size_t subspace_idx = loop.get_subspace_looped(bispace_idx);
                uncontracted_dim *= bispaces[bispace_idx][subspace_idx].get_dim();
            }
        }
    }
    size_t contracted_dim = 1;
    for(size_t loop_idx = 0; loop_idx < contracted_loops.size(); ++loop_idx)
    {
        const block_loop& loop = contracted_loops[loop_idx];
        for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
        {
            if(!loop.is_bispace_ignored(bispace_idx))
            {
                size_t subspace_idx = loop.get_subspace_looped(bispace_idx);
                contracted_dim *= bispaces[bispace_idx][subspace_idx].get_dim();
            }
        }
    }

    std::vector<block_loop> loops;
    //Fudge factor of 2 for writes being more expensive 
    if(contracted_dim > uncontracted_dim*2)
    {
        loops.insert(loops.end(),contracted_loops.begin(),contracted_loops.end());
        loops.insert(loops.end(),uncontracted_loops.begin(),uncontracted_loops.end());
    }
    else
    {
        loops.insert(loops.end(),uncontracted_loops.begin(),uncontracted_loops.end());
        loops.insert(loops.end(),contracted_loops.begin(),contracted_loops.end());
    }

    std::vector<size_t> direct_tensors;
    if(C.get_data_ptr() == NULL)
    {
        direct_tensors.push_back(0);
    }
    if(A.get_data_ptr() == NULL)
    {
        direct_tensors.push_back(1);
    }
    if(B.get_data_ptr() == NULL)
    {
        direct_tensors.push_back(2);
    }
    m_sll_ptr = new sparse_loop_list(loops,bispaces,direct_tensors);
    m_bc2k_ptr = new block_contract2_kernel<T>(*m_sll_ptr);
}
    
template<typename T>
void batch_kernel_contract2<T>::generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches)
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
    m_sll_ptr->run(*m_bc2k_ptr,ptrs,loop_batches);
}

} // namespace libtensor


#endif /* BATCH_KERNEL_CONTRACT2_H */
