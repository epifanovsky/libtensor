#ifndef BATCH_KERNEL_ADD2_H
#define BATCH_KERNEL_ADD2_H

#include "batch_kernel.h"
#include "gen_sparse_btensor.h"
#include "sparse_loop_list.h"
#include "block_add2_kernel.h"

namespace libtensor {

template<typename T>
class batch_kernel_add2 : public batch_kernel<T>
{
private:
    std::vector<sparse_bispace_any_order> m_bispaces;
    sparse_loop_list* m_sll_add_ptr;
    sparse_loop_list* m_sll_sub_ptr;
    block_add2_kernel<T> m_ba2k_add;
    block_add2_kernel<T> m_ba2k_sub;
public:
    static const char* k_clazz; //!< Class name

    template<size_t NC,size_t NA,size_t NB>
    batch_kernel_add2(const gen_sparse_btensor<NC,T>& C,const gen_sparse_btensor<NA,T>& A,const gen_sparse_btensor<NB,T>& B,double lhs_scalar,double rhs_scalar);

    virtual void generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches);
    virtual void init(const std::vector<T*>& ptrs,const bispace_batch_map& bbm);

    ~batch_kernel_add2();
    batch_kernel_add2(const batch_kernel_add2<T>& rhs);
    batch_kernel_add2& operator=(const batch_kernel_add2<T>& rhs);
};

template<typename T>
batch_kernel_add2<T>::~batch_kernel_add2() 
{ 
    delete m_sll_add_ptr; 
    delete m_sll_sub_ptr; 
}

template<typename T>
batch_kernel_add2<T>::batch_kernel_add2(const batch_kernel_add2<T>& rhs) : m_sll_add_ptr(new sparse_loop_list(*rhs.m_sll_add_ptr)),
                                                                           m_sll_sub_ptr(new sparse_loop_list(*rhs.m_sll_sub_ptr))
{
}

template<typename T>
batch_kernel_add2<T>& batch_kernel_add2<T>::operator=(const batch_kernel_add2<T>& rhs) 
{

    m_sll_add_ptr = new sparse_loop_list(*rhs.m_sll_add_ptr);
    m_sll_sub_ptr = new sparse_loop_list(*rhs.m_sll_sub_ptr);
    return *this;
}

template<typename T>
const char* batch_kernel_add2<T>::k_clazz = "batch_kernel_add2";

template<typename T> template<size_t NC,size_t NA,size_t NB>
batch_kernel_add2<T>::batch_kernel_add2(const gen_sparse_btensor<NC,T>& C,const gen_sparse_btensor<NA,T>& A,const gen_sparse_btensor<NB,T>& B,double lhs_scalar,double rhs_scalar) : m_ba2k_add(1,lhs_scalar),m_ba2k_sub(1,rhs_scalar)
{
    if(NC != NA || NC != NB)
    {
        throw bad_parameter(g_ns, k_clazz,"batch_kernel_add2(...)",
                __FILE__, __LINE__, "Invalid tensor orders");
    }

    m_bispaces.push_back(C.get_bispace());
    m_bispaces.push_back(A.get_bispace());
    m_bispaces.push_back(B.get_bispace());

    //To obtain correct results 
    //We must use separate loops so that sparse fusion does not occur when subtracting
    //sparse from dense
    std::vector<block_loop> loops;
    for(size_t i = 0; i < NC; ++i)
    {
        block_loop bl(m_bispaces);
        bl.set_subspace_looped(0,i);
        bl.set_subspace_looped(1,i);
        bl.set_subspace_looped(2,i);
        loops.push_back(bl);
    }

    //Direct tensors?
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

    //Due to the fact that sparse_loop_list currently fuses sparsity by default, 
    //we must separate adding the first entry from subtracting the second
    //Add the first operand
    std::vector<sparse_bispace_any_order> add_bispaces(2,m_bispaces[0]);
    add_bispaces.push_back(m_bispaces[1]);
    std::vector<sparse_bispace_any_order> sub_bispaces(2,m_bispaces[0]);
    sub_bispaces.push_back(m_bispaces[2]);
    std::vector<block_loop> add_loops;
    std::vector<block_loop> sub_loops;
    for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
    {
        const block_loop& loop = loops[loop_idx];
        block_loop bl_add(add_bispaces);
        bl_add.set_subspace_looped(0,loop.get_subspace_looped(0));
        bl_add.set_subspace_looped(1,loop.get_subspace_looped(0));
        bl_add.set_subspace_looped(2,loop.get_subspace_looped(1));
        add_loops.push_back(bl_add);

        block_loop bl_sub(sub_bispaces);
        bl_sub.set_subspace_looped(0,loop.get_subspace_looped(0));
        bl_sub.set_subspace_looped(1,loop.get_subspace_looped(0));
        bl_sub.set_subspace_looped(2,loop.get_subspace_looped(2));
        sub_loops.push_back(bl_sub);
    }

    idx_list new_direct_tensors;
    if(find(direct_tensors.begin(),direct_tensors.end(),0) != direct_tensors.end())
    {
        new_direct_tensors.push_back(0);
        new_direct_tensors.push_back(1);
    }
    if(find(direct_tensors.begin(),direct_tensors.end(),1) != direct_tensors.end())
    {
        new_direct_tensors.push_back(2);
    }
    m_sll_add_ptr = new sparse_loop_list(add_loops,add_bispaces,new_direct_tensors);

    if(new_direct_tensors.size() > 0 && new_direct_tensors.back() == 2)
    {
        new_direct_tensors.pop_back();
    }
    if(find(direct_tensors.begin(),direct_tensors.end(),2) != direct_tensors.end())
    {
        new_direct_tensors.push_back(2);
    }
    m_sll_sub_ptr = new sparse_loop_list(sub_loops,sub_bispaces,new_direct_tensors);
}

template<typename T>
void batch_kernel_add2<T>::init(const std::vector<T*>& ptrs,const bispace_batch_map& bbm)
{
    size_t output_batch_size = m_bispaces[0].get_nnz();
    for(bispace_batch_map::const_iterator it = bbm.begin(); it != bbm.end(); ++it)
    {
        if(it->first.first == 0)
        {
            output_batch_size = m_bispaces[0].get_batch_size(it->first.second,it->second);
            break;
        }
    }
    memset(ptrs[0],0,output_batch_size*sizeof(T));
}
    
template<typename T>
void batch_kernel_add2<T>::generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches)
{
    //TODO: THIS WILL NOT WORK FOR SIMULTANEOUS PERMUTATION AND SUBTRACTION!!!
    std::map<size_t,idx_pair> loop_batches;
    const std::vector<block_loop>& loops = m_sll_add_ptr->get_loops();
    for(bispace_batch_map::const_iterator batch_it = batches.begin(); batch_it != batches.end(); ++batch_it)
    {
        size_t bispace_idx = batch_it->first.first;
        if(bispace_idx != 2)
        {
            bispace_idx = (bispace_idx  == 0) ? 0  : 2;
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
    }

    std::vector<T*> cur_ptrs(ptrs.size(),ptrs[0]);
    cur_ptrs[2] = ptrs[1];
    m_sll_add_ptr->run(m_ba2k_add,cur_ptrs,loop_batches);
    
    loop_batches.clear();
    for(bispace_batch_map::const_iterator batch_it = batches.begin(); batch_it != batches.end(); ++batch_it)
    {
        size_t bispace_idx = batch_it->first.first;
        if(bispace_idx != 1)
        {
            bispace_idx = (bispace_idx  == 0) ? 0  : 2;
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
    }
    cur_ptrs[2] = ptrs[2]; 
    m_sll_sub_ptr->run(m_ba2k_sub,cur_ptrs,loop_batches);
}

} // namespace libtensor



#endif /* BATCH_KERNEL_ADD2_H */
