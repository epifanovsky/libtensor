#ifndef LIBTENSOR_BTOD_SET_H
#define LIBTENSOR_BTOD_SET_H

#include <libtensor/core/block_tensor_i.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/tod/tod_set.h>
#include <libtensor/block_tensor/bto/bto_set.h>

namespace libtensor {


struct btod_set_traits {

    typedef double element_type;

    template<size_t N> struct block_tensor_type {
        typedef block_tensor_i<N, double> type;
    };

    template<size_t N> struct block_tensor_ctrl_type {
        typedef block_tensor_ctrl<N, double> type;
    };

    template<size_t N> struct block_type {
        typedef dense_tensor_i<N, double> type;
    };

    template<size_t N> struct to_set_type {
        typedef tod_set<N> type;
    };

    static bool is_zero(double d) {
        return d == 0.0;
    }

};


/** \brief Sets all elements of a block tensor to a value preserving
        the symmetry
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_set : public bto_set<N, btod_set_traits> {
public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    btod_set(double v = 0.0) : bto_set<N, btod_set_traits>(v) { }

};


template<size_t N>
const char *btod_set<N>::k_clazz = "btod_set<N>";


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_H
