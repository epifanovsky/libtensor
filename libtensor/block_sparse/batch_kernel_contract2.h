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
public:
    static const char* k_clazz; //!< Class name

    template<size_t NC,size_t NA,size_t NB>
    batch_kernel_contract2(const gen_sparse_btensor<NC,T>& C,const gen_sparse_btensor<NA,T>& A,const gen_sparse_btensor<NB,T>& B,const std::multimap<size_t,size_t>& contr_map);

    virtual void generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches);

    sparse_loop_list* m_sll_ptr;
    block_contract2_kernel<T>* m_bc2k_ptr;

    //~batch_kernel_contract2() { delete m_sll_ptr; delete m_bc2k_ptr; }
    //batch_kernel_contract2(const batch_kernel_contract2<T>& rhs) : m_sll_ptr(new sparse_loop_list(*rhs.m_sll_ptr)),m_bc2k_ptr(new block_contract2_kernel<T>(*rhs.m_bc2k_ptr)) {}
    //batch_kernel_contract2& operator=(const batch_kernel_contract2<T>& rhs) { m_sll_ptr = new sparse_loop_list(*rhs.m_sll_ptr); m_bc2k_ptr = new block_contract2_kernel<T>(*rhs.m_bc2k_ptr); }
};

template<typename T>
const char* batch_kernel_contract2<T>::k_clazz = "batch_kernel_contract2";

template<typename T> template<size_t NC,size_t NA,size_t NB>
batch_kernel_contract2<T>::batch_kernel_contract2(const gen_sparse_btensor<NC,T>& C,const gen_sparse_btensor<NA,T>& A,const gen_sparse_btensor<NB,T>& B,const std::multimap<size_t,size_t>& contr_map)
{
    m_sll_ptr = NULL;
    m_bc2k_ptr = NULL;
}
    
template<typename T>
void batch_kernel_contract2<T>::generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches)
{
}

} // namespace libtensor


#endif /* BATCH_KERNEL_CONTRACT2_H */
