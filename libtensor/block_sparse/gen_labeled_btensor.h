#ifndef GEN_LABELED_BTENSOR_H
#define GEN_LABELED_BTENSOR_H

#include "../iface/letter_expr.h"
#include "batch_provider.h"

namespace libtensor { 

template<size_t N,typename T = double>
class gen_labeled_btensor {
public:
    virtual letter_expr<N> get_letter_expr() const = 0;

    virtual sparse_bispace<N> get_bispace() const = 0;

    virtual const T* get_data_ptr() const = 0;

    virtual batch_provider<T>* get_batch_provider() const = 0; 
};

} // namespace libtensor

#endif /* GEN_LABELED_BTENSOR_H */
