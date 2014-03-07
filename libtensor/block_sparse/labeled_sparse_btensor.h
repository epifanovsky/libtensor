#ifndef LABELED_SPARSE_BTENSOR_H
#define LABELED_SPARSE_BTENSOR_H

#include "sparse_btensor.h"
#include "gen_labeled_btensor.h"
#include "block_permute_kernel.h"
#include "permute.h"
#include "batch_provider_factory.h"

namespace libtensor {

//Forward declaration for member variable
template<size_t N,typename T>
class sparse_btensor;

template<size_t N,typename T = double> 
class labeled_sparse_btensor : public gen_labeled_btensor<N,T>
{
private:
    sparse_btensor<N,T>& m_tensor; 
    expr::label<N> m_le;

    void run_permutation(const gen_labeled_btensor<N,T>& rhs);
public:

    labeled_sparse_btensor(sparse_btensor<N,T>& tensor,const expr::label<N>& le) : m_tensor(tensor),m_le(le) { };

    expr::label<N> get_letter_expr() const { return m_le; }
    virtual const T* get_data_ptr() const { return m_tensor.get_data_ptr(); }
    virtual batch_provider<T>* get_batch_provider() const { return NULL; } 

    /** \brief Return the sparse_bispace defining this tensor 
     **/
    sparse_bispace<N> get_bispace() const { return m_tensor.get_bispace(); }; 

    //For permutations
    labeled_sparse_btensor<N,T>& operator=(const gen_labeled_btensor<N,T>& rhs) { run_permutation(rhs); return *this; }
    labeled_sparse_btensor<N,T>& operator=(const labeled_sparse_btensor<N,T>& rhs) { run_permutation(rhs); return *this; }

    //Store the result of an expression in this tensor
    labeled_sparse_btensor<N,T>& operator=(const batch_provider_factory<N,T>& factory);
};

//Used for evaluating permutations
template<size_t N,typename T> 
void labeled_sparse_btensor<N,T>::run_permutation(const gen_labeled_btensor<N,T>& rhs)
{
    //TODO - HORRIBLE HACK, FIX 
    permute2_batch_provider<T>(*this,rhs).get_batch((T*)this->m_tensor.get_data_ptr(),std::map<idx_pair,idx_pair>(),1e18);
}

//Used for evaluating contractions, to prevent an unnecessary copy at the end
template<size_t N,typename T>
labeled_sparse_btensor<N,T>& labeled_sparse_btensor<N,T>::operator=(const batch_provider_factory<N,T>& factory)
{
    batch_provider<T>* bp = factory.get_batch_provider(*this);
    bp->get_batch((T*)m_tensor.get_data_ptr());
    delete bp;
    return *this;
}

} // namespace libtensor

#endif /* LABELED_SPARSE_BTENSOR_H */
