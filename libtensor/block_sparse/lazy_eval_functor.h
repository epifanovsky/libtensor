#ifndef LAZY_EVAL_FUNCTOR_
#define LAZY_EVAL_FUNCTOR_

#include <vector>
#include <stdlib.h>

namespace libtensor {

//Forward declaration
template<size_t N,typename T> 
class labeled_sparse_btensor;

typedef std::vector<size_t> block_list; 

//Used to erase other size parameters from contract_eval_functor 
template<size_t N,typename T=double>
class lazy_eval_functor {
public:
    virtual void operator()(labeled_sparse_btensor<N,T>& dest) const = 0;

    virtual ~lazy_eval_functor() {};
};

} // namespace libtensor

#endif /* LAZY_EVAL_FUNCTOR_ */
