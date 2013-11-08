#ifndef LAZY_EVAL_FUNCTOR_
#define LAZY_EVAL_FUNCTOR_

#include <stdlib.h>

namespace libtensor {

//Forward declaration
template<size_t N,typename T> 
class labeled_sparse_btensor;

//Used to erase other size parameters from contract_eval_functor 
template<size_t N,typename T=double>
class lazy_eval_functor {
public:
    virtual void operator()(labeled_sparse_btensor<N,T>& dest) const = 0;
};

} // namespace libtensor

#endif /* LAZY_EVAL_FUNCTOR_ */
